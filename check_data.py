# check_data.py
# sanity checks on the labeled spam splits written by prepare_labeled.py. run it after every rebuild
# of the data, before trusting any score from evaluate.py.
#
#   1. integrity     - every saved example is well formed (ends in <|endoftext|>, padded with it, valid token ids)
#   2. leakage       - no message the classifier is tested on also appears in its training data, either as an
#                      exact copy (after truncation to block_size) or as a near-copy (same spam, different name/link)
#   3. contamination - how much of each split the base GPT already read during pretraining on enron_mails.csv.
#                      the legitimate messages in spam_mails.csv come from Enron mailboxes, so the base model
#                      may have seen them before. that doesn't break the classifier, but test scores on those
#                      messages measure "have I seen this?" as much as "is this spam?".
#
# near-copies are found by comparing 5-word sequences ("shingles", see dedup.py). contamination uses
# longer 13-word sequences, which are long enough that matching one by chance is very unlikely.
#
# per-message results are saved to data/data_checks.npz so evaluate.py can report clean subsets.
#
# examples:
#   python check_data.py
#   python check_data.py --skip_corpus     # leave out the (slower) pretraining contamination scan

import os
import sys
import time
import argparse
import numpy as np

from model import enc
from prepare_labeled import EOT, LABELS
from evaluate import load_split, split_fingerprint
from dedup import words, hash_words, shingles, message_shingles, near_duplicates

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# 1. integrity
# ---------------------------------------------------------------------------
def check_integrity(name, tokens, lengths, labels):
    block_size = tokens.shape[1]
    rows = np.arange(len(lengths))
    problems = []
    if not ((lengths >= 2) & (lengths <= block_size)).all():
        problems.append("lengths outside [2, block_size]")
    elif not (tokens[rows, lengths - 1] == EOT).all():
        problems.append("message not ending in <|endoftext|>")
    if (tokens >= enc.n_vocab).any():
        problems.append("token ids outside the GPT-2 vocabulary")
    padding = np.arange(block_size)[None, :] >= lengths[:, None]
    if not (tokens[padding] == EOT).all():
        problems.append("padding that isn't <|endoftext|>")
    if not np.isin(labels, range(len(LABELS))).all():
        problems.append(f"labels outside 0..{len(LABELS) - 1}")
    # the end-of-text token should only appear at the end of a message, never inside it
    body = np.arange(block_size)[None, :] < (lengths - 1)[:, None]
    if (tokens[body] == EOT).any():
        problems.append("<|endoftext|> inside a message body")
    for p in problems:
        print(f"  FAIL {name}: {p}")
    return not problems


def print_stats(data):
    print(f"\n{'split':<8}{'messages':>10}{'spam %':>9}{'truncated':>11}{'median len':>12}{'p90 len':>9}")
    for name, (tokens, lengths, labels) in data.items():
        truncated = (lengths == tokens.shape[1]).mean()
        print(f"{name:<8}{len(labels):>10,}{labels.mean():>9.1%}{truncated:>11.1%}"
              f"{int(np.median(lengths)):>12}{int(np.percentile(lengths, 90)):>9}")


# ---------------------------------------------------------------------------
# 2. leakage between splits
# ---------------------------------------------------------------------------
def exact_duplicates(data):
    # compares exactly what the model reads: two different emails that share their first block_size
    # tokens are identical inputs once truncated
    seen = {}
    dupes = {name: np.zeros(len(data[name][2]), dtype=bool) for name in data}
    for name, (tokens, lengths, _) in data.items():
        for i, (row, n) in enumerate(zip(tokens, lengths)):
            key = row[:n].tobytes()
            if key in seen:
                dupes[name][i] = True
            else:
                seen[key] = name
    return dupes


# ---------------------------------------------------------------------------
# 3. contamination from the pretraining corpus
# ---------------------------------------------------------------------------
def corpus_hits(tokens_path, wanted, n, chunk_tokens=10_000_000):
    # streams the pretraining token cache and returns which of the wanted shingles appear in it,
    # separately for the part train.py trained on (first 90%) and the part it validated on (last 10%)
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    split_at = int(0.9 * len(tokens))
    found = {"pretrain": [], "pretrain_val": []}
    start, t0 = 0, time.time()
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        # end each chunk on a message boundary so no message is cut in half
        if end < len(tokens):
            eots = np.flatnonzero(tokens[start:end] == EOT)
            if len(eots):
                end = start + int(eots[-1]) + 1
        for region, lo, hi in (("pretrain", start, min(end, split_at)), ("pretrain_val", max(start, split_at), end)):
            if lo >= hi:
                continue
            text = enc.decode(tokens[lo:hi].tolist()).replace("<|endoftext|>", "\n")
            h = shingles(hash_words(words(text)), n)
            found[region].append(np.unique(h[np.isin(h, wanted)]))
        print(f"  scanned {end:,} / {len(tokens):,} tokens ({time.time() - t0:.0f}s)", end="\r")
        start = end
    print()
    return {region: np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
            for region, parts in found.items()}


def overlap_fraction(message_shingle_sets, hits):
    # share of each message's shingles that also appear in the corpus
    return np.array([np.isin(s, hits).mean() if len(s) else 0.0 for s in message_shingle_sets])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def preview(text, width=90):
    text = text.removeprefix("Subject: ")
    return text[:width] + ("..." if len(text) > width else "")


def main():
    parser = argparse.ArgumentParser(description="Integrity, leakage and contamination checks for the labeled spam splits")
    parser.add_argument("--data_dir", default="data", help="Where prepare_labeled.py wrote the splits")
    parser.add_argument("--tokens", default="enron_body_tokens.bin", help="Pretraining token cache written by train.py")
    parser.add_argument("--skip_corpus", action="store_true", help="Skip the pretraining contamination scan")
    parser.add_argument("--near_dup_ngram", type=int, default=5, help="Words per shingle for near-duplicate detection")
    parser.add_argument("--near_dup_threshold", type=float, default=0.8, help="Jaccard similarity that counts as a near-duplicate")
    parser.add_argument("--contam_ngram", type=int, default=13, help="Words per shingle for the contamination scan")
    parser.add_argument("--contam_threshold", type=float, default=0.5, help="Share of a message found in pretraining that counts as contaminated")
    parser.add_argument("--examples", type=int, default=3, help="How many example pairs to print")
    args = parser.parse_args()

    data = {name: load_split(args.data_dir, name) for name in SPLITS}
    texts = {name: [enc.decode(row[:n - 1].tolist()) for row, n in zip(tokens, lengths)]
             for name, (tokens, lengths, _) in data.items()}
    # lets evaluate.py confirm these results belong to the splits it's evaluating
    results = {f"fingerprint_{name}": split_fingerprint(*data[name]) for name in SPLITS}
    ok = True

    # 1. integrity
    print("1. Integrity")
    ok &= all(check_integrity(name, *data[name]) for name in SPLITS)
    print("  all splits well formed" if ok else "  integrity problems found")
    print_stats(data)

    # 2. leakage
    print("\n2. Leakage between splits")
    dupes = exact_duplicates(data)
    for name in SPLITS:
        count = int(dupes[name].sum())
        print(f"  exact copies in {name:<5} of an earlier message: {count:,}")
        results[f"exact_dup_{name}"] = dupes[name]
    if dupes["val"].any() or dupes["test"].any():
        print("  FAIL: val/test contain inputs identical to earlier messages")
        ok = False

    near = {name: [message_shingles(t, args.near_dup_ngram) for t in texts[name]] for name in SPLITS}
    print(f"\n  near-duplicates (Jaccard >= {args.near_dup_threshold} on {args.near_dup_ngram}-word shingles):")
    for query, reference in (("val", "train"), ("test", "train"), ("test", "val")):
        sim, match, flagged = near_duplicates(near[query], near[reference], args.near_dup_threshold)
        labels_q, labels_r = data[query][2], data[reference][2]
        by_class = ", ".join(f"{LABELS[c]} {int((flagged & (labels_q == c)).sum())}" for c in range(len(LABELS)))
        disagree = int((flagged & (labels_q != labels_r[match])).sum())
        print(f"    {query:<5} vs {reference:<5}: {int(flagged.sum()):>4} of {len(flagged):,} ({flagged.mean():.1%})"
              f" | {by_class} | label disagrees with its twin: {disagree}")
        results[f"near_dup_{query}_vs_{reference}"] = flagged
        results[f"near_dup_sim_{query}_vs_{reference}"] = sim
        for i in np.flatnonzero(flagged)[np.argsort(-sim[flagged])][:args.examples]:
            print(f"      {sim[i]:.2f}  {query}: {preview(texts[query][i])}")
            print(f"            {reference}: {preview(texts[reference][match[i]])}")

    # 3. contamination
    if not args.skip_corpus:
        print(f"\n3. Contamination from pretraining ({args.tokens}, {args.contam_ngram}-word shingles)")
        if not os.path.exists(args.tokens):
            print(f"  {args.tokens} not found - run train.py once to build it, or pass --skip_corpus")
        else:
            contam = {name: [message_shingles(t, args.contam_ngram) for t in texts[name]] for name in SPLITS}
            wanted = np.unique(np.concatenate([s for name in SPLITS for s in contam[name]]))
            hits = corpus_hits(args.tokens, wanted, args.contam_ngram)
            print(f"  messages with >= {args.contam_threshold:.0%} of their text found in the corpus the base model trained on:")
            print(f"    {'split':<8}" + "".join(f"{label:>14}" for label in LABELS) + f"{'all':>14}{'in pretrain val':>18}")
            for name in SPLITS:
                frac = overlap_fraction(contam[name], hits["pretrain"])
                frac_val = overlap_fraction(contam[name], hits["pretrain_val"])
                flagged = frac >= args.contam_threshold
                labels = data[name][2]
                cells = "".join(f"{f'{int(flagged[labels == c].sum()):,} ({flagged[labels == c].mean():.0%})':>14}"
                                for c in range(len(LABELS)))
                print(f"    {name:<8}{cells}{f'{int(flagged.sum()):,} ({flagged.mean():.0%})':>14}"
                      f"{int((frac_val >= args.contam_threshold).sum()):>18,}")
                results[f"pretrain_overlap_{name}"] = frac
                results[f"pretrain_val_overlap_{name}"] = frac_val

    path = os.path.join(args.data_dir, "data_checks.npz")
    np.savez(path, **results)
    print(f"\nSaved per-message results to {path}")
    if not ok:
        print("Some checks FAILED - fix the data before trusting evaluation scores.")
        sys.exit(1)


if __name__ == "__main__":
    main()
