# eval_lm_test.py
# checks the fixed evaluation windows, the n-gram baselines and the bits-per-byte bookkeeping in eval_lm.py.

import math
import os
import tempfile
import unittest
import numpy as np
import torch

from model import GPT, enc
from eval_lm import (window_starts, windows, model_nll, ngram_baselines, token_byte_lengths, load_prompts)
from tests.helpers import tiny_config


class WindowTest(unittest.TestCase):
    def test_all_windows_tile_the_data_without_overlap(self):
        starts = window_starts(np.arange(1000), 8, None)
        self.assertEqual(len(starts), 999 // 8)
        np.testing.assert_array_equal(np.diff(starts), 8)

    def test_sample_is_spread_evenly_and_repeatable(self):
        data = np.arange(100_000)
        starts = window_starts(data, 10, 1_000)
        self.assertEqual(len(starts), 100)
        self.assertEqual(starts[0], 0)
        self.assertGreater(starts[-1], 90_000)          # reaches the end of the split, not just the start
        self.assertTrue((starts % 10 == 0).all())       # still aligned to non-overlapping windows
        np.testing.assert_array_equal(starts, window_starts(data, 10, 1_000))

    def test_targets_are_inputs_shifted_by_one(self):
        x, y = windows(np.arange(50), np.array([0, 10]), 5)
        np.testing.assert_array_equal(x[1], [10, 11, 12, 13, 14])
        np.testing.assert_array_equal(y[1], [11, 12, 13, 14, 15])

    def test_model_nll_matches_the_model_loss(self):
        torch.manual_seed(0)
        model = GPT(tiny_config(vocab_size=100, block_size=16))
        data = np.random.default_rng(0).integers(0, 100, 500)
        starts = window_starts(data, 16, None)
        nll = model_nll(model, data, starts, 16, torch.device("cpu"), batch_size=7)
        x, y = windows(data, starts, 16)
        with torch.no_grad():
            _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        self.assertAlmostEqual(float(nll.mean()), loss.item(), places=4)


class NgramTest(unittest.TestCase):
    def test_bigram_learns_a_repeating_pattern(self):
        # 0 1 2 3 0 1 2 3 ... : the previous token gives the next one away
        train = np.tile(np.arange(4), 5_000)
        prev = np.tile(np.arange(4), 10).reshape(5, 8)
        targets = (prev + 1) % 4
        rows = ngram_baselines(train, prev, targets, vocab_size=10, tune_tokens=400)
        self.assertAlmostEqual(rows["uniform"].mean(), math.log(10))
        # unigram: four tokens equally common -> about ln(4)
        self.assertAlmostEqual(rows["unigram"].mean(), math.log(4), delta=0.01)
        bigram = [v for k, v in rows.items() if k.startswith("bigram")][0]
        self.assertLess(bigram.mean(), 0.1)


class BytesAndPromptsTest(unittest.TestCase):
    def test_token_byte_lengths_add_up_to_the_text(self):
        text = "Hi Vince, the price is $4.50 – café ☕"
        lengths = token_byte_lengths(enc.n_vocab)
        self.assertEqual(int(lengths[enc.encode_ordinary(text)].sum()), len(text.encode("utf-8")))

    def test_prompts_file_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prompts.txt")
            with open(path, "w") as f:
                f.write("# comment\nHello,\\n\\nThanks\n\nFYI - \n")
            self.assertEqual(load_prompts(path), ["Hello,\n\nThanks", "FYI - "])


if __name__ == "__main__":
    unittest.main()
