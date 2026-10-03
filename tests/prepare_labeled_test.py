# prepare_labeled_test.py
# checks how the labeled spam data is cleaned, tokenized and split. mistakes here leak between
# train and test, or feed the classifier text in a different shape than it was trained on.

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from model import enc
from prepare_labeled import (EOT, normalize_text, format_message, load_messages,
                             encode_messages, remove_near_duplicates, stratified_split)


class FormattingTest(unittest.TestCase):
    def test_lowercases_and_splits_off_punctuation(self):
        self.assertEqual(normalize_text("Don't PAY $1,000!!"), "don ' t pay $ 1 , 000 ! !")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("  hello\r\n\n  there\t friend "), "hello there friend")

    def test_already_normalized_text_is_unchanged(self):
        # spam_mails.csv is already in this shape, so normalizing it again must be a no-op
        text = normalize_text("Re: Meeting @ 3pm -- can't make it, sorry!")
        self.assertEqual(normalize_text(text), text)

    def test_format_message_matches_the_dataset_layout(self):
        self.assertEqual(format_message("Lunch?", "Friday at noon."), "Subject: lunch ? friday at noon .")
        self.assertEqual(format_message("", "hi"), "Subject: hi")


class LoadMessagesTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spam.csv")
            pd.DataFrame(rows, columns=["text", "spam"]).to_csv(path, index=False)
            return load_messages(path)

    def test_removes_duplicates_and_conflicting_labels(self):
        df = self.load([
            ["Subject: win money now", 1],
            ["Subject: win  money now", 1],     # duplicate once whitespace is collapsed
            ["Subject: meeting notes", 0],
            ["Subject: maybe spam", 0],
            ["Subject: maybe spam", 1],          # same text, both labels - can't be trusted
            ["Subject: ", 0],                    # nothing left after the prefix
        ])
        self.assertEqual(sorted(df["text"]), ["Subject: meeting notes", "Subject: win money now"])
        self.assertEqual(df.set_index("text").loc["Subject: win money now", "spam"], 1)


class EncodeMessagesTest(unittest.TestCase):
    block_size = 16

    def test_short_message_ends_with_eot_then_padding(self):
        ids = enc.encode_ordinary("Subject: hi there")
        tokens, lengths, truncated = encode_messages(["Subject: hi there"], self.block_size)
        self.assertEqual(tokens.shape, (1, self.block_size))
        self.assertEqual(lengths[0], len(ids) + 1)
        self.assertEqual(tokens[0, :len(ids)].tolist(), ids)
        self.assertTrue((tokens[0, len(ids):] == EOT).all())
        self.assertEqual(truncated, 0)

    def test_long_message_keeps_the_start_and_still_ends_with_eot(self):
        text = "Subject: " + "word " * 100
        ids = enc.encode_ordinary(text)
        tokens, lengths, truncated = encode_messages([text], self.block_size)
        self.assertEqual(lengths[0], self.block_size)
        self.assertEqual(tokens[0, :-1].tolist(), ids[:self.block_size - 1])
        self.assertEqual(tokens[0, -1], EOT)
        self.assertEqual(truncated, 1)

    def test_lengths_fit_their_dtypes(self):
        tokens, lengths, _ = encode_messages(["a", "b c"], 256)
        self.assertEqual(tokens.dtype, np.uint16)
        self.assertTrue((lengths >= 2).all() and (lengths <= 256).all())


class RemoveNearDuplicatesTest(unittest.TestCase):
    spam = "Subject: you have won a free cruise click here to claim your prize today"

    def keep(self, texts, labels, block_size=64):
        tokens, lengths, _ = encode_messages(texts, block_size)
        return remove_near_duplicates(tokens, lengths, np.array(labels), ngram=3, threshold=0.7).tolist()

    def test_keeps_the_first_of_each_near_copy(self):
        texts = [self.spam,
                 "Subject: lunch is moved to the cafeteria on the second floor this week",
                 self.spam.replace("today", "now")]
        self.assertEqual(self.keep(texts, [1, 0, 1]), [True, True, False])

    def test_messages_identical_after_truncation_are_copies(self):
        start = "Subject: " + "the market report for this morning is attached " * 10
        texts = [start + "see you at the meeting", start + "call me if anything changes"]
        self.assertEqual(self.keep(texts, [0, 0], block_size=16), [True, False])

    def test_near_copies_with_different_labels_are_all_dropped(self):
        texts = [self.spam, self.spam.replace("today", "now"),
                 "Subject: lunch is moved to the cafeteria on the second floor this week"]
        self.assertEqual(self.keep(texts, [1, 0, 0]), [False, False, True])


class StratifiedSplitTest(unittest.TestCase):
    def setUp(self):
        # 25% spam, like the real dataset
        self.labels = np.array([1] * 250 + [0] * 750)
        self.splits = stratified_split(self.labels, val_frac=0.1, test_frac=0.1, seed=1337)

    def test_every_message_lands_in_exactly_one_split(self):
        all_idx = np.concatenate(list(self.splits.values()))
        self.assertEqual(len(all_idx), len(self.labels))
        self.assertEqual(len(np.unique(all_idx)), len(self.labels))

    def test_sizes_and_spam_ratio(self):
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, {"train": 800, "val": 100, "test": 100})
        for name, idx in self.splits.items():
            self.assertAlmostEqual(self.labels[idx].mean(), 0.25, places=2, msg=name)

    def test_same_seed_gives_same_split(self):
        again = stratified_split(self.labels, val_frac=0.1, test_frac=0.1, seed=1337)
        for name in self.splits:
            np.testing.assert_array_equal(self.splits[name], again[name])


if __name__ == "__main__":
    unittest.main()
