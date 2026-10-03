# model_test.py
# checks that the transformer in model.py is wired up correctly. a bug here (attention peeking at
# future tokens, padding leaking into the classifier) wouldn't crash anything - it would just quietly
# make the model worse, or make its scores look better than they really are.

import math
import os
import tempfile
import unittest
import torch

from model import GPT, GPTClassifier, count_parameters
from train import configure_optimizer, save_checkpoint
from tests.helpers import tiny_config, random_tokens


class GPTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.config = tiny_config(vocab_size=100)
        self.model = GPT(self.config).eval()

    def test_output_shapes(self):
        x = random_tokens(3, 10, self.config.vocab_size)
        logits, loss = self.model(x)
        self.assertEqual(tuple(logits.shape), (3, 10, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_causal_attention_never_sees_the_future(self):
        # changing token t must not change the predictions at any position before t
        x = random_tokens(2, 20, self.config.vocab_size)
        changed = x.clone()
        t = 12
        changed[:, t] = (changed[:, t] + 1) % self.config.vocab_size
        with torch.no_grad():
            before, _ = self.model(x)
            after, _ = self.model(changed)
        torch.testing.assert_close(before[:, :t], after[:, :t])
        # ...but it should change the prediction AT t, otherwise the test proves nothing
        self.assertFalse(torch.allclose(before[:, t], after[:, t]))

    def test_untrained_loss_is_close_to_random_guessing(self):
        # small random weights -> roughly uniform predictions -> loss ~= ln(vocab_size)
        x = random_tokens(4, 32, self.config.vocab_size)
        y = random_tokens(4, 32, self.config.vocab_size, seed=1)
        with torch.no_grad():
            _, loss = self.model(x, y)
        self.assertAlmostEqual(loss.item(), math.log(self.config.vocab_size), delta=0.1)

    def test_sequence_longer_than_block_size_is_rejected(self):
        x = random_tokens(1, self.config.block_size + 1, self.config.vocab_size)
        with self.assertRaises(AssertionError):
            self.model(x)

    def test_output_head_is_tied_to_token_embeddings(self):
        self.assertIs(self.model.lm_head.weight, self.model.transformer.wte.weight)
        # and the shared matrix is only counted once
        unique = {id(p): p for p in self.model.parameters()}
        self.assertEqual(count_parameters(self.model), sum(p.numel() for p in unique.values()))

    def test_can_memorize_one_batch(self):
        # the classic smoke test: if a model can't drive the loss to ~0 on a single batch it sees
        # over and over, something in the forward pass, loss or optimizer is broken
        model = GPT(self.config).train()
        x = random_tokens(2, 16, self.config.vocab_size)
        y = random_tokens(2, 16, self.config.vocab_size, seed=1)
        optimizer = configure_optimizer(model, weight_decay=0.0, lr=1e-2, device=torch.device("cpu"))
        for _ in range(150):
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.1)


class GPTClassifierTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.config = tiny_config(vocab_size=100)
        self.model = GPTClassifier(self.config).eval()

    def classify(self, tokens, lengths):
        with torch.no_grad():
            logits, _ = self.model(tokens, torch.tensor(lengths))
        return logits

    def test_output_shape(self):
        x = random_tokens(3, 10, self.config.vocab_size)
        logits = self.classify(x, [10, 5, 1])
        self.assertEqual(tuple(logits.shape), (3, 2))

    def test_padding_after_the_message_is_ignored(self):
        # the score is read at position length - 1, and attention is causal, so whatever comes
        # after the message - and how much of it there is - must not matter
        message = random_tokens(1, 8, self.config.vocab_size)
        short = self.classify(message, [8])
        padded = torch.cat([message, torch.zeros(1, 10, dtype=torch.long)], dim=1)
        junk = torch.cat([message, random_tokens(1, 10, self.config.vocab_size, seed=5)], dim=1)
        torch.testing.assert_close(short, self.classify(padded, [8]))
        torch.testing.assert_close(short, self.classify(junk, [8]))

    def test_batching_with_longer_messages_does_not_change_the_result(self):
        a = random_tokens(1, 6, self.config.vocab_size, seed=1)
        b = random_tokens(1, 20, self.config.vocab_size, seed=2)
        alone = self.classify(a, [6])
        batch = torch.cat([torch.cat([a, torch.zeros(1, 14, dtype=torch.long)], dim=1), b])
        torch.testing.assert_close(alone, self.classify(batch, [6, 20])[:1])

    def test_score_comes_from_the_last_real_token(self):
        # the prediction for length n should depend on token n - 1 ...
        x = random_tokens(1, 10, self.config.vocab_size)
        changed = x.clone()
        changed[0, 6] = (changed[0, 6] + 1) % self.config.vocab_size
        self.assertFalse(torch.allclose(self.classify(x, [7]), self.classify(changed, [7])))
        # ... but not on token n
        changed = x.clone()
        changed[0, 7] = (changed[0, 7] + 1) % self.config.vocab_size
        torch.testing.assert_close(self.classify(x, [7]), self.classify(changed, [7]))

    def test_class_weights_change_the_loss(self):
        x = random_tokens(4, 10, self.config.vocab_size)
        lengths = torch.tensor([10, 10, 10, 10])
        labels = torch.tensor([0, 0, 0, 1])
        with torch.no_grad():
            _, plain = self.model(x, lengths, labels)
            _, weighted = self.model(x, lengths, labels, torch.tensor([1.0, 1.0]))
            _, spam_heavy = self.model(x, lengths, labels, torch.tensor([1.0, 5.0]))
        torch.testing.assert_close(plain, weighted)
        self.assertNotAlmostEqual(plain.item(), spam_heavy.item(), places=4)

    def test_from_pretrained_loads_the_transformer_weights(self):
        gpt = GPT(self.config)
        optimizer = configure_optimizer(gpt, weight_decay=0.1, lr=1e-3, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(gpt, optimizer, self.config, step=0, val_loss=0.0, path=path)
            classifier = GPTClassifier.from_pretrained(path)
        for name, tensor in gpt.state_dict().items():
            torch.testing.assert_close(classifier.gpt.state_dict()[name], tensor, msg=name)


if __name__ == "__main__":
    unittest.main()
