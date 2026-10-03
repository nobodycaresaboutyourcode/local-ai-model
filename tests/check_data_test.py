# check_data_test.py
# checks the duplicate and overlap detection in dedup.py and check_data.py on small hand-made examples.

import unittest
import numpy as np

from prepare_labeled import EOT
from dedup import words, hash_words, shingles, message_shingles, near_duplicates, duplicate_groups
from check_data import exact_duplicates, overlap_fraction, check_integrity


class ShingleTest(unittest.TestCase):
    def test_labeled_and_raw_text_give_the_same_words(self):
        # the labeled data is lowercased with punctuation split off; the Enron bodies are not
        self.assertEqual(words("don ' t pay $ 1 , 000"), words("Don't pay $1,000"))

    def test_one_shingle_per_window(self):
        h = shingles(hash_words("a b c d e".split()), 3)
        self.assertEqual(len(h), 3)
        self.assertEqual(len(np.unique(h)), 3)

    def test_same_words_give_the_same_shingle(self):
        a = shingles(hash_words("x the quick brown fox".split()), 3)
        b = shingles(hash_words("the quick brown fox y".split()), 3)
        self.assertEqual(len(np.intersect1d(a, b)), 2)  # "the quick brown", "quick brown fox"

    def test_word_order_matters(self):
        a = shingles(hash_words("one two three".split()), 3)
        b = shingles(hash_words("three two one".split()), 3)
        self.assertEqual(len(np.intersect1d(a, b)), 0)

    def test_short_message_becomes_one_shingle(self):
        self.assertEqual(len(shingles(hash_words(["hi", "there"]), 5)), 1)
        self.assertEqual(len(shingles(hash_words([]), 5)), 0)

    def test_subject_prefix_is_ignored(self):
        np.testing.assert_array_equal(message_shingles("Subject: lunch on friday", 2),
                                      message_shingles("lunch on friday", 2))


class DuplicateTest(unittest.TestCase):
    def test_near_duplicates_find_the_closest_reference(self):
        reference = ["meeting moved to thursday at three in room b please bring the slides",
                     "you have won a free cruise click here to claim your prize today"]
        query = ["you have won a free cruise click here to claim your prize now",  # one word changed
                 "the quarterly numbers look good and we should talk about hiring"]
        sets = lambda texts: [message_shingles(t, 3) for t in texts]
        sim, match, flagged = near_duplicates(sets(query), sets(reference), threshold=0.7)
        self.assertEqual(match[0], 1)
        self.assertGreater(sim[0], 0.7)
        self.assertEqual(sim[1], 0.0)
        self.assertEqual(flagged.tolist(), [True, False])

    def test_duplicate_groups_keep_the_first_of_each_group(self):
        texts = ["you have won a free cruise click here to claim your prize today",
                 "the quarterly numbers look good and we should talk about hiring",
                 "you have won a free cruise click here to claim your prize now",
                 "you have won a free cruise click here to claim your prize today"]
        group = duplicate_groups([message_shingles(t, 3) for t in texts], threshold=0.7)
        self.assertEqual(group.tolist(), [0, 1, 0, 0])

    def test_exact_duplicates_compare_only_the_real_tokens(self):
        # identical messages with different amounts of padding are still the same input
        def split(rows):
            tokens = np.full((len(rows), 6), EOT, dtype=np.int64)
            for i, r in enumerate(rows):
                tokens[i, :len(r)] = r
            return tokens, np.array([len(r) for r in rows]), np.zeros(len(rows), dtype=np.int64)
        data = {"train": split([[1, 2, EOT], [3, EOT]]), "test": split([[1, 2, EOT], [4, EOT]])}
        dupes = exact_duplicates(data)
        self.assertEqual(dupes["train"].tolist(), [False, False])
        self.assertEqual(dupes["test"].tolist(), [True, False])

    def test_overlap_fraction(self):
        hits = np.array([1, 2, 3], dtype=np.uint64)
        sets = [np.array([1, 2, 9, 10], dtype=np.uint64), np.array([], dtype=np.uint64)]
        np.testing.assert_allclose(overlap_fraction(sets, hits), [0.5, 0.0])


class IntegrityTest(unittest.TestCase):
    def test_flags_a_message_missing_its_end_token(self):
        tokens = np.array([[5, 6, EOT, EOT], [7, 8, 9, EOT]])
        labels = np.array([0, 1])
        self.assertTrue(check_integrity("ok", tokens, np.array([3, 4]), labels))
        bad = tokens.copy()
        bad[1, 3] = 10
        self.assertFalse(check_integrity("bad", bad, np.array([3, 4]), labels))


if __name__ == "__main__":
    unittest.main()
