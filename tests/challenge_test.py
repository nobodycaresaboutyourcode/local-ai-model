# challenge_test.py
# checks the challenge set loader and the edits used for the invariance and directional tests.

import os
import tempfile
import unittest
import numpy as np
import torch

from model import GPTClassifier
from evaluate import predict_probs
from challenge import CASES_PATH, load_cases, swap_names, perturb, encode, same_input, run, INVARIANCE
from tests.helpers import tiny_config

SAMPLE = """# a comment before the first case
=== a-1
label: spam
tags: phishing, short
subject: hi: there
---
line one

line two
=== a-2
label: legitimate
tags: work
subject:
---
ok
"""


class LoadCasesTest(unittest.TestCase):
    def test_parses_headers_and_multiline_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cases.txt")
            with open(path, "w") as f:
                f.write(SAMPLE)
            cases = load_cases(path)
        self.assertEqual([c["id"] for c in cases], ["a-1", "a-2"])
        self.assertEqual(cases[0]["label"], 1)
        self.assertEqual(cases[0]["tags"], ["phishing", "short"])
        self.assertEqual(cases[0]["subject"], "hi: there")   # only the first colon splits
        self.assertEqual(cases[0]["body"], "line one\n\nline two")
        self.assertEqual((cases[1]["label"], cases[1]["subject"], cases[1]["body"]), (0, "", "ok"))

    def test_the_real_challenge_set_is_well_formed(self):
        cases = load_cases(CASES_PATH)
        self.assertGreater(len(cases), 50)
        for c in cases:
            self.assertIn(c["label"], (0, 1), c["id"])
            self.assertTrue(c["tags"], c["id"])
            self.assertTrue(c["body"], c["id"])


class PerturbationTest(unittest.TestCase):
    def test_swap_names_replaces_whole_words_only(self):
        self.assertEqual(swap_names("Hi Sarah", "Tom and Tommy"), ("Hi Aisha", "Diego and Tommy"))
        self.assertIsNone(swap_names("hello", "no names here"))

    def test_perturb_respects_labels(self):
        cases = [{"label": 0, "subject": "a", "body": "b"}, {"label": 1, "subject": "c", "body": "d"}]
        pairs = perturb(cases, (1,), lambda s, b: (s, b + "!"))
        self.assertEqual([(i, c["body"]) for i, c in pairs], [(1, "d!")])

    def test_case_and_spacing_edits_are_no_ops_after_normalization(self):
        cases = [{"label": 0, "subject": "Lunch", "body": "See you at noon."}]
        t1, l1 = encode(cases, 32)
        for name, _, edit in INVARIANCE[:2]:
            t2, l2 = encode([c for _, c in perturb(cases, (0,), edit)], 32)
            self.assertTrue(same_input(t1, l1, t2, l2).all(), name)


class RunTest(unittest.TestCase):
    def test_runs_end_to_end_with_a_tiny_model(self):
        torch.manual_seed(0)
        model = GPTClassifier(tiny_config(block_size=64)).eval()
        cases = load_cases(CASES_PATH)[:6]
        result = run("tiny", lambda t, l: predict_probs(model, t, l, torch.device("cpu")), 0.5, cases, 64)
        self.assertEqual(len(result["probs"]), 6)
        self.assertTrue(result["edge_ok"])
        self.assertEqual(len(result["invariance"]), len(INVARIANCE))
        self.assertTrue(((result["probs"] >= 0) & (result["probs"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
