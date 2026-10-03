# compare_test.py
# checks compare.py's gate on hand-made results, and that run_evals.py's results can be saved as json.

import copy
import json
import unittest
import numpy as np

from compare import gate_checks, test_agreement, challenge_changes
from run_evals import jsonable


def make_run(probs, wrong, perplexity=25.0):
    return {"classifier": {"split_fingerprint": "abc", "threshold": 0.5, "probs": probs,
                           "metrics": {"fbeta": 0.99}, "ci": {"fbeta": [0.97, 1.0]}},
            "challenge": {"wrong": wrong},
            "lm": {"perplexity": perplexity}}


def results(checks):
    return {name.split(":")[-1].strip(): passed for name, passed, _ in checks}


class GateTest(unittest.TestCase):
    def setUp(self):
        self.a = make_run([0.1] * 99 + [0.9] * 101, ["x", "y"])

    def test_identical_runs_pass(self):
        self.assertTrue(all(results(gate_checks(self.a, copy.deepcopy(self.a))).values()))

    def test_lower_fbeta_fails(self):
        b = copy.deepcopy(self.a)
        b["classifier"]["metrics"]["fbeta"] = 0.95
        self.assertFalse(results(gate_checks(self.a, b))["test F0.5 within the first run's 95% CI"])

    def test_too_many_changed_predictions_fail(self):
        b = copy.deepcopy(self.a)
        b["classifier"]["probs"][:3] = [0.9, 0.9, 0.9]   # 3 of 200 = 98.5% agreement
        self.assertEqual(test_agreement(self.a, b)["changed"], 3)
        self.assertFalse(results(gate_checks(self.a, b))["test predictions agree >= 99%"])

    def test_different_test_splits_cannot_pass(self):
        b = copy.deepcopy(self.a)
        b["classifier"]["split_fingerprint"] = "other"
        self.assertIsNone(test_agreement(self.a, b))
        self.assertFalse(results(gate_checks(self.a, b))["test predictions agree"])

    def test_same_score_but_different_mistakes_fails(self):
        b = copy.deepcopy(self.a)
        b["challenge"]["wrong"] = ["p", "q"]   # still 2 wrong, but none of the same ones
        self.assertEqual(challenge_changes(self.a, b), {"newly_wrong": ["p", "q"], "newly_right": ["x", "y"]})
        r = results(gate_checks(self.a, b))
        self.assertTrue(r["at most 1 more wrong"])
        self.assertFalse(r["at most 2 labels change"])

    def test_perplexity_rise(self):
        b = copy.deepcopy(self.a)
        b["lm"]["perplexity"] = 25.4   # +1.6%
        self.assertTrue(results(gate_checks(self.a, b))["perplexity rises <= 2%"])
        b["lm"]["perplexity"] = 26.0   # +4%
        self.assertFalse(results(gate_checks(self.a, b))["perplexity rises <= 2%"])

    def test_missing_lm_is_skipped_not_failed(self):
        b = copy.deepcopy(self.a)
        del b["lm"]
        self.assertIsNone(results(gate_checks(self.a, b))["perplexity rises <= 2%"])


class JsonableTest(unittest.TestCase):
    def test_numpy_values_become_plain_json(self):
        data = {"a": np.float32(0.1234567891), "b": np.int64(3), "c": np.array([1.0, 2.0]), "e": "x"}
        out = json.loads(json.dumps(jsonable(data)))
        self.assertEqual(out, {"a": 0.123457, "b": 3, "c": [1.0, 2.0], "e": "x"})


if __name__ == "__main__":
    unittest.main()
