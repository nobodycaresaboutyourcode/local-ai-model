# classify_test.py
# runs a whole email through classify.py's path - formatting, tokenizing, scoring - with a tiny
# untrained model. the answers are meaningless; the point is that the plumbing works end to end.

import unittest
import torch

from model import GPTClassifier
from prepare_labeled import LABELS
from classify import classify
from tests.helpers import tiny_config


class ClassifyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.model = GPTClassifier(tiny_config(block_size=32)).eval()
        self.cpu = torch.device("cpu")

    def test_returns_a_label_and_probability(self):
        label, p_spam, truncated = classify(self.model, 0.5, "Lunch?", "Are we still on for Friday?", self.cpu)
        self.assertIn(label, LABELS)
        self.assertTrue(0.0 <= p_spam <= 1.0)
        self.assertFalse(truncated)

    def test_label_follows_the_threshold(self):
        self.assertEqual(classify(self.model, 0.0, "hi", "hello", self.cpu)[0], "spam")
        self.assertEqual(classify(self.model, 1.01, "hi", "hello", self.cpu)[0], "legitimate")

    def test_long_messages_are_flagged_as_truncated(self):
        _, _, truncated = classify(self.model, 0.5, "news", "lots of words " * 50, self.cpu)
        self.assertTrue(truncated)

    def test_empty_message_does_not_crash(self):
        label, _, _ = classify(self.model, 0.5, "", "", self.cpu)
        self.assertIn(label, LABELS)

    def test_case_and_spacing_do_not_matter(self):
        # classify normalizes text the same way as the training data, so these are the same message
        _, a, _ = classify(self.model, 0.5, "WIN BIG", "Click   here NOW!", self.cpu)
        _, b, _ = classify(self.model, 0.5, "win big", "click here now !", self.cpu)
        self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
