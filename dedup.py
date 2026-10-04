# dedup.py
# finds messages that are near-copies of each other - the same spam with a different name or link,
# the same form email sent to different people. shared by prepare_labeled.py (which removes them before
# splitting) and check_data.py (which checks that none slipped across splits).
#
# each message is cut into overlapping runs of n words ("shingles"). two messages that share most of
# their shingles are near-duplicates, even if a few words differ. similarity is the Jaccard index:
# shared shingles / all distinct shingles of the pair.

import re
import numpy as np
from scipy.sparse import csr_matrix

PRIME = np.uint64(1099511628211)  # multiplier for the rolling n-gram hash


# ---------------------------------------------------------------------------
# text -> words -> hashed n-grams
# ---------------------------------------------------------------------------
# text is reduced to plain lowercase words with punctuation dropped, so the labeled data
# ("don ' t pay $ 1 , 000") and the raw Enron bodies ("Don't pay $1,000") compare as equal.
def words(text):
    return re.findall(r"\w+", text.lower())


def hash_words(ws):
    # python's string hash is randomized per process, which is fine - everything is compared within one run
    return np.fromiter(map(hash, ws), dtype=np.int64, count=len(ws)).view(np.uint64)


def shingles(word_hashes, n):
    # one hash per run of n consecutive words, computed for all positions at once.
    # a message shorter than n words becomes a single shingle of all its words.
    n = min(n, len(word_hashes))
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    count = len(word_hashes) - n + 1
    acc = np.zeros(count, dtype=np.uint64)
    for k in range(n):
        acc = acc * PRIME + word_hashes[k:k + count]  # uint64 arithmetic simply wraps around
    return acc


def message_shingles(text, n):
    ws = words(text)
    # the labeled data starts every message with "Subject:", which the Enron bodies never contain
    if ws and ws[0] == "subject":
        ws = ws[1:]
    return np.unique(shingles(hash_words(ws), n))


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------
def jaccard(query, reference):
    # Jaccard similarity of every query/reference pair that shares at least one shingle, as a sparse matrix.
    # each message becomes a row of 0s and 1s, one column per distinct shingle; multiplying the two
    # matrices counts the shared shingles of every pair in one go.
    _, columns = np.unique(np.concatenate(query + reference), return_inverse=True)
    n_cols = int(columns.max()) + 1 if len(columns) else 0

    def matrix(sets, offset):
        rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        cols = columns[offset:offset + len(rows)]
        return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sets), n_cols))

    q = matrix(query, 0)
    r = matrix(reference, q.nnz)
    shared = (q @ r.T).tocoo()
    sizes_q = np.array([len(s) for s in query], dtype=np.float32)
    sizes_r = np.array([len(s) for s in reference], dtype=np.float32)
    shared.data = shared.data / (sizes_q[shared.row] + sizes_r[shared.col] - shared.data)
    return shared.tocsr()


def near_duplicates(query, reference, threshold):
    # for every message in query, its most similar message in reference
    sim = jaccard(query, reference).toarray()
    best_match = sim.argmax(axis=1)
    best_sim = sim[np.arange(len(query)), best_match]
    return best_sim, best_match, best_sim >= threshold


def duplicate_groups(sets, threshold):
    # walks the messages in order: the first of each group of near-copies is kept, and every later
    # message at least `threshold` similar to it joins its group. returns, for each message, the index
    # of the message it was grouped under (itself if it was kept).
    sim = jaccard(sets, sets)
    group = np.arange(len(sets))
    for i in range(len(sets)):
        if group[i] != i:
            continue
        row = sim.indices[sim.indptr[i]:sim.indptr[i + 1]]
        vals = sim.data[sim.indptr[i]:sim.indptr[i + 1]]
        later = row[(row > i) & (vals >= threshold)]
        later = later[group[later] == later]   # not already claimed by an earlier group
        group[later] = i
    return group
