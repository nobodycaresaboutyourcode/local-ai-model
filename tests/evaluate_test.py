# evaluate_test.py
# checks the scoring code. if the metrics are wrong, every comparison between models is wrong too.

import os
import tempfile
import unittest
import numpy as np
import torch

from model import GPTClassifier
from evaluate import (binary_metrics, choose_threshold, predict_probs, load_classifier, ranking_metrics,
                      bootstrap, mcnemar, wilson_interval, split_fingerprint)
from train_classifier import save_classifier
from tests.helpers import tiny_config, random_tokens


class BinaryMetricsTest(unittest.TestCase):
    def test_hand_computed_confusion_matrix(self):
        # 3 spam caught, 1 legitimate flagged, 2 spam missed, 4 legitimate let through
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1])
        labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
        m = binary_metrics(probs, labels, threshold=0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (3, 1, 2, 4))
        self.assertAlmostEqual(m["accuracy"], 0.7)
        self.assertAlmostEqual(m["precision"], 3 / 4)
        self.assertAlmostEqual(m["recall"], 3 / 5)
        self.assertAlmostEqual(m["f1"], 2 * 0.75 * 0.6 / (0.75 + 0.6))
        # F0.5 = 1.25 * p * r / (0.25 * p + r)
        self.assertAlmostEqual(m["fbeta"], 1.25 * 0.75 * 0.6 / (0.25 * 0.75 + 0.6))

    def test_threshold_is_inclusive(self):
        m = binary_metrics(np.array([0.5]), np.array([1]), threshold=0.5)
        self.assertEqual(m["tp"], 1)

    def test_fbeta_with_beta_one_is_f1(self):
        probs = np.array([0.9, 0.6, 0.2, 0.7])
        labels = np.array([1, 0, 1, 1])
        m = binary_metrics(probs, labels, beta=1.0)
        self.assertAlmostEqual(m["fbeta"], m["f1"])

    def test_predicting_nothing_as_spam_scores_zero_without_crashing(self):
        m = binary_metrics(np.zeros(4), np.array([1, 0, 0, 0]), threshold=0.5)
        self.assertEqual((m["precision"], m["recall"], m["f1"], m["fbeta"]), (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(m["accuracy"], 0.75)


class ChooseThresholdTest(unittest.TestCase):
    def test_finds_a_cut_that_separates_the_classes(self):
        probs = np.array([0.05, 0.1, 0.2, 0.3, 0.35, 0.36, 0.8, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        threshold = choose_threshold(probs, labels)
        self.assertTrue(0.3 < threshold <= 0.35)
        self.assertEqual(binary_metrics(probs, labels, threshold)["fbeta"], 1.0)

    def test_prefers_precision_over_recall(self):
        # one spam message scores lower than a legitimate one. catching it would cost a false
        # positive; F0.5 values precision more, so the threshold should sit above the legitimate message
        probs = np.array([0.1, 0.2, 0.6, 0.5, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        threshold = choose_threshold(probs, labels)
        self.assertGreater(threshold, 0.6)


    def test_tied_cutoffs_pick_the_middle(self):
        # every cut-off between 0.01 and 0.80 separates the classes perfectly; taking the first one
        # would flag anything with a 1% chance of being spam
        probs = np.array([0.0, 0.0, 0.0, 0.81, 0.9, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(choose_threshold(probs, labels), 0.41)


class RankingMetricsTest(unittest.TestCase):
    def test_perfect_ranking(self):
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
        labels = np.array([0, 0, 0, 1, 1])
        m = ranking_metrics(probs, labels)
        self.assertEqual((m["roc_auc"], m["pr_auc"], m["recall_at_fpr"]), (1.0, 1.0, 1.0))

    def test_reversed_ranking(self):
        m = ranking_metrics(np.array([0.9, 0.8, 0.1]), np.array([0, 0, 1]))
        self.assertEqual(m["roc_auc"], 0.0)
        self.assertEqual(m["recall_at_fpr"], 0.0)

    def test_recall_at_fpr_allows_only_that_many_false_positives(self):
        # 100 legitimate messages: at 1% FPR one of them may outscore spam, but not two
        probs = np.concatenate([np.full(98, 0.1), [0.95, 0.85], [0.9, 0.8]])
        labels = np.concatenate([np.zeros(100), np.ones(2)]).astype(int)
        self.assertEqual(ranking_metrics(probs, labels)["recall_at_fpr"], 0.5)

    def test_calibration(self):
        # a bin of messages all scored 0.8, of which 80% really are spam: perfectly calibrated
        probs = np.full(10, 0.8)
        labels = np.array([1] * 8 + [0] * 2)
        m = ranking_metrics(np.append(probs, 0.0), np.append(labels, 0))
        self.assertAlmostEqual(m["ece"], 0.0)
        # always saying 0.8 when it's spam half the time is off by 0.3
        m = ranking_metrics(probs, np.array([1, 0] * 5))
        self.assertAlmostEqual(m["ece"], 0.3)
        self.assertAlmostEqual(m["brier"], 0.5 * 0.2 ** 2 + 0.5 * 0.8 ** 2)


class UncertaintyTest(unittest.TestCase):
    def test_bootstrap_interval_contains_the_score(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        probs = np.clip(labels * 0.6 + rng.random(200) * 0.5, 0, 1)
        m = binary_metrics(probs, labels, 0.5)
        ci = bootstrap(probs, labels, 0.5, n=200)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            lo, hi = ci[key]
            self.assertLessEqual(lo, hi)
            if key in m:
                self.assertTrue(lo <= m[key] <= hi, key)

    def test_mcnemar_counts_only_disagreements(self):
        a = np.array([True] * 10 + [False] * 2 + [True] * 50)
        b = np.array([False] * 10 + [True] * 2 + [True] * 50)
        a_only, b_only, p = mcnemar(a, b)
        self.assertEqual((a_only, b_only), (10, 2))
        self.assertLess(p, 0.05)
        self.assertEqual(mcnemar(a, a)[2], 1.0)

    def test_wilson_interval(self):
        lo, hi = wilson_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)  # zero mistakes in 10 still allows up to ~28%
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo + hi, 1.0)

    def test_fingerprint_changes_with_the_data(self):
        tokens, lengths, labels = np.ones((3, 4)), np.array([2, 3, 4]), np.array([0, 1, 1])
        self.assertEqual(split_fingerprint(tokens, lengths, labels), split_fingerprint(tokens, lengths, labels))
        self.assertNotEqual(split_fingerprint(tokens, lengths, labels), split_fingerprint(tokens, lengths, labels[::-1]))


class PredictTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.model = GPTClassifier(tiny_config(vocab_size=100))
        self.tokens = random_tokens(10, 32, 100).numpy()
        self.lengths = np.array([32, 5, 12, 1, 30, 7, 8, 20, 32, 2])

    def test_returns_one_probability_per_message(self):
        probs = predict_probs(self.model, self.tokens, self.lengths, torch.device("cpu"))
        self.assertEqual(probs.shape, (10,))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

    def test_batch_size_does_not_change_the_result(self):
        cpu = torch.device("cpu")
        np.testing.assert_allclose(predict_probs(self.model, self.tokens, self.lengths, cpu, batch_size=3),
                                   predict_probs(self.model, self.tokens, self.lengths, cpu, batch_size=64),
                                   rtol=1e-5, atol=1e-6)

    def test_restores_training_mode(self):
        self.model.train()
        predict_probs(self.model, self.tokens, self.lengths, torch.device("cpu"))
        self.assertTrue(self.model.training)

    def test_saved_classifier_gives_the_same_predictions(self):
        cpu = torch.device("cpu")
        expected = predict_probs(self.model, self.tokens, self.lengths, cpu)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "classifier.pt")
            save_classifier(self.model, epoch=1, val_f1=0.5, threshold=0.42, path=path)
            loaded, checkpoint = load_classifier(path, cpu)
        self.assertEqual(checkpoint["threshold"], 0.42)
        self.assertFalse(loaded.training)
        np.testing.assert_allclose(predict_probs(loaded, self.tokens, self.lengths, cpu), expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
