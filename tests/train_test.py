# train_test.py
# checks the pieces of train.py that every training run depends on: the learning rate schedule,
# the batches pulled from the token cache, and which parameters get weight decay.

import os
import tempfile
import unittest
import numpy as np
import torch

from model import GPT
from train import get_lr, load_data, configure_optimizer
from tests.helpers import tiny_config


class LearningRateTest(unittest.TestCase):
    warmup, max_steps, max_lr, min_lr = 10, 100, 1e-3, 1e-4

    def lr(self, step):
        return get_lr(step, self.warmup, self.max_steps, self.max_lr, self.min_lr)

    def test_warmup_ramps_up_to_max(self):
        self.assertAlmostEqual(self.lr(0), self.max_lr / self.warmup)
        self.assertAlmostEqual(self.lr(self.warmup - 1), self.max_lr)
        warmup = [self.lr(s) for s in range(self.warmup)]
        self.assertEqual(warmup, sorted(warmup))

    def test_cosine_decays_from_max_to_min(self):
        self.assertAlmostEqual(self.lr(self.warmup), self.max_lr)
        decay = [self.lr(s) for s in range(self.warmup, self.max_steps)]
        self.assertEqual(decay, sorted(decay, reverse=True))
        self.assertTrue(all(self.min_lr <= lr <= self.max_lr for lr in decay))

    def test_stays_at_min_after_the_last_step(self):
        self.assertAlmostEqual(self.lr(self.max_steps), self.min_lr)
        self.assertAlmostEqual(self.lr(self.max_steps * 2), self.min_lr)


class BatchTest(unittest.TestCase):
    # a fake token cache where every token id equals its position, so it's easy to see
    # exactly where each batch came from
    n_tokens, block_size, batch_size = 1000, 8, 64

    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "tokens.bin")
        np.arange(self.n_tokens, dtype=np.uint16).tofile(path)
        # the corpus path is never read, because the token cache already exists
        self.get_batch = load_data("missing.csv", path, self.batch_size, self.block_size, torch.device("cpu"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_targets_are_inputs_shifted_by_one(self):
        x, y = self.get_batch("train")
        self.assertEqual(tuple(x.shape), (self.batch_size, self.block_size))
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        torch.testing.assert_close(y[:, -1], x[:, -1] + 1)

    def test_train_and_val_never_overlap(self):
        split = int(0.9 * self.n_tokens)
        for _ in range(20):
            x, y = self.get_batch("train")
            self.assertLess(int(y.max()), split)
            x, y = self.get_batch("val")
            self.assertGreaterEqual(int(x.min()), split)


class OptimizerTest(unittest.TestCase):
    def test_weight_decay_only_on_matrices(self):
        model = GPT(tiny_config(vocab_size=100))
        optimizer = configure_optimizer(model, weight_decay=0.1, lr=1e-3, device=torch.device("cpu"))
        decay, no_decay = optimizer.param_groups
        self.assertEqual(decay["weight_decay"], 0.1)
        self.assertEqual(no_decay["weight_decay"], 0.0)
        self.assertTrue(all(p.dim() >= 2 for p in decay["params"]))
        self.assertTrue(all(p.dim() < 2 for p in no_decay["params"]))
        # every parameter is in exactly one group
        grouped = decay["params"] + no_decay["params"]
        self.assertEqual(len(grouped), len(list(model.parameters())))


if __name__ == "__main__":
    unittest.main()
