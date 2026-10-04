# prepare_labeled.py
# prepares the labeled spam dataset for fine-tuning the Enron GPT as a spam classifier.
#
# spam_mails.csv has two columns:
#   text - "Subject: ..." followed by the message body (already lowercased and split on spaces)
#   spam - 1 if the user flagged the message as spam, 0 if it's a legitimate message
#
# each message is BPE-encoded with the same GPT-2 tokenizer the base model was trained on,
# truncated/padded to a fixed length, de-duplicated, and saved as train/val/test splits in data/.

import os
import re
import argparse
import numpy as np
import pandas as pd

from model import enc
from dedup import message_shingles, duplicate_groups

LABELS = ["legitimate", "spam"]
# GPT-2's <|endoftext|> token - marks the end of each message and fills the padding after it
EOT = enc.eot_token
SUBJECT_PREFIX = "Subject:"


# ---------------------------------------------------------------------------
# text formatting - shared by training (this file) and classification (classify.py)
# ---------------------------------------------------------------------------
# the labeled data was lowercased and had its punctuation split off ("don ' t", "$ 1 , 000").
# a real email has to be put into the same shape before classifying it, otherwise the model
# sees text that looks nothing like what it was fine-tuned on.
def normalize_text(text):
    text = re.sub(r"([^\w\s])", r" \1 ", text.lower())
    # collapse runs of whitespace (the dataset is full of double spaces) - saves tokens
    # and makes near-identical messages compare as equal when de-duplicating
    return " ".join(text.split())


# every message is "Subject: <subject> <body>", matching the layout of spam_mails.csv
def format_message(subject, body):
    return f"{SUBJECT_PREFIX} {normalize_text(f'{subject} {body}')}"


def load_messages(path):
    # only read the two columns we need, with compact dtypes
    df = pd.read_csv(path, usecols=["text", "spam"], dtype={"text": "string", "spam": "int8"})
    print(f"Loaded {len(df):,} messages from {path}")

    df = df.dropna(subset=["text", "spam"])
    # the subject and body are already joined in the text column, so it all goes through as the body
    df["text"] = df["text"].str.removeprefix(SUBJECT_PREFIX).map(lambda body: format_message("", body))
    df = df[df["text"] != f"{SUBJECT_PREFIX} "]

    # the same message often appears more than once (spam especially). if copies land in both
    # train and test, the model gets tested on messages it has already memorized, so keep one of each.
    # a message flagged as both spam and legitimate can't be trusted either way, so drop it entirely.
    conflicting = df.groupby("text")["spam"].transform("nunique") > 1
    if conflicting.any():
        print(f"  dropping {df.loc[conflicting, 'text'].nunique():,} messages with conflicting labels")
    df = df[~conflicting]
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  removed {before - len(df):,} duplicates, {len(df):,} unique messages remain")
    return df


def encode_messages(texts, block_size):
    # every example becomes exactly block_size tokens: the message, one <|endoftext|>, then padding.
    # we keep the START of long messages - the subject and opening lines carry the most spam signal.
    # padding goes at the end: attention is causal, so padding can never influence the real tokens before it.
    encoded = enc.encode_ordinary_batch(texts, num_threads=os.cpu_count())
    tokens = np.full((len(encoded), block_size), EOT, dtype=np.uint16)
    lengths = np.empty(len(encoded), dtype=np.int16)
    truncated = 0
    for i, ids in enumerate(encoded):
        if len(ids) >= block_size:
            truncated += 1
        ids = ids[:block_size - 1] + [EOT]
        tokens[i, :len(ids)] = ids
        # the classifier reads its prediction from position length - 1 (the <|endoftext|> token),
        # which is the only position that has attended to the whole message
        lengths[i] = len(ids)
    return tokens, lengths, truncated


def remove_near_duplicates(tokens, lengths, labels, ngram, threshold):
    # load_messages only removes exact text copies. two kinds of copy still get through, and if they
    # land on both sides of the train/test split the test score rewards memorization:
    #   - near-copies: the same spam or form email with a different name, date or garbled character
    #   - truncation copies: different emails whose first block_size tokens are identical, so the model
    #     sees exactly the same input
    # they are compared on the text the model actually reads (the truncated tokens). the first message
    # of each group is kept; a group whose copies disagree on the label is dropped entirely.
    texts = [enc.decode(row[:n - 1].tolist()) for row, n in zip(tokens, lengths)]
    group = duplicate_groups([message_shingles(t, ngram) for t in texts], threshold)
    first_seen = {}
    for i, (row, n) in enumerate(zip(tokens, lengths)):
        key = row[:n].tobytes()
        if group[i] == i and key in first_seen:
            group[i] = first_seen[key]
        first_seen.setdefault(key, group[i])

    conflicting = np.zeros(len(labels), dtype=bool)
    for g in np.unique(group):
        members = group == g
        if len(np.unique(labels[members])) > 1:
            conflicting |= members
    keep = (group == np.arange(len(labels))) & ~conflicting
    print(f"  removed {int((group != np.arange(len(labels))).sum()):,} near-duplicates "
          f"(Jaccard >= {threshold} on {ngram}-word shingles, or identical after truncation)")
    if conflicting.any():
        print(f"  dropped {int(conflicting.sum()):,} messages in near-duplicate groups with conflicting labels")
    print(f"  {int(keep.sum()):,} messages remain")
    return keep


def stratified_split(labels, val_frac, test_frac, seed):
    # split each class separately so train/val/test all keep the same spam ratio
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        splits["test"].append(idx[:n_test])
        splits["val"].append(idx[n_test:n_test + n_val])
        splits["train"].append(idx[n_test + n_val:])
    # shuffle again so the classes are mixed within each split
    return {name: rng.permutation(np.concatenate(parts)) for name, parts in splits.items()}


def main():
    parser = argparse.ArgumentParser(description="Prepare the labeled spam dataset for classifier fine-tuning")
    parser.add_argument("--input", default="spam_mails.csv", help="CSV with 'text' and 'spam' columns")
    parser.add_argument("--out_dir", default="data", help="Where to write the prepared splits")
    parser.add_argument("--block_size", type=int, default=256, help="Tokens per example - must match the base model's block_size")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--near_dup_ngram", type=int, default=5, help="Words per shingle when comparing messages")
    parser.add_argument("--near_dup_threshold", type=float, default=0.8, help="Jaccard similarity that counts as a near-duplicate")
    args = parser.parse_args()

    df = load_messages(args.input)
    tokens, lengths, truncated = encode_messages(df["text"].tolist(), args.block_size)
    labels = df["spam"].to_numpy(dtype=np.int64)
    print(f"  {truncated:,} of {len(df):,} messages ({truncated / len(df):.0%}) truncated to {args.block_size} tokens")
    keep = remove_near_duplicates(tokens, lengths, labels, args.near_dup_ngram, args.near_dup_threshold)
    tokens, lengths, labels = tokens[keep], lengths[keep], labels[keep]

    os.makedirs(args.out_dir, exist_ok=True)
    splits = stratified_split(labels, args.val_frac, args.test_frac, args.seed)
    print(f"\n{'split':<8}{'messages':>10}{'spam':>8}{'legit':>8}{'spam %':>9}")
    for name, idx in splits.items():
        path = os.path.join(args.out_dir, f"spam_{name}.npz")
        np.savez(path, tokens=tokens[idx], lengths=lengths[idx], labels=labels[idx], block_size=args.block_size)
        n_spam = int(labels[idx].sum())
        print(f"{name:<8}{len(idx):>10,}{n_spam:>8,}{len(idx) - n_spam:>8,}{n_spam / len(idx):>9.1%}")
    print(f"\nSaved splits to {args.out_dir}/")


if __name__ == "__main__":
    main()
