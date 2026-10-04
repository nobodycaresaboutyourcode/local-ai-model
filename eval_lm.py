# eval_lm.py
# measures the pretrained language model (checkpoints/ckpt.pt from train.py) on Enron text it didn't train on.
#
# train.py's "val loss" is the average over 50 RANDOM batches, so it moves from run to run and is fine for
# watching training but not for comparing two checkpoints. this script scores a FIXED set of windows from
# the validation split (the last 10% of enron_body_tokens.bin), so the same checkpoint always gets the
# same number and two checkpoints are compared on exactly the same text.
#
# it reports:
#   loss           - average cross entropy per token (nats): how surprised the model is by the real next token
#   perplexity     - e^loss: "as uncertain as choosing between this many equally likely tokens"
#   bits per byte  - the loss spread over the raw text's bytes instead of its tokens. perplexity depends on the
#                    tokenizer; bits per byte doesn't, so it can be compared with the character-level model
#                    or any other tokenizer
#   loss by position - loss at each position in the window. early positions have almost no context, late
#                    ones have up to 255 tokens. if the late ones aren't much better, attention isn't helping.
#
# and the same numbers for simple baselines the model has to beat:
#   uniform  - every token equally likely: ln(50,257). an untrained model scores about this
#   unigram  - each token's frequency in the training split, ignoring context
#   bigram   - next-token frequencies given only the previous token
#
# finally it writes samples from a fixed list of prompts (eval_sets/prompts.txt) with fixed seeds, so
# generations from two checkpoints can be compared side by side.
#
# examples:
#   python eval_lm.py                                              # ckpt.pt on a 2M-token sample of val
#   python eval_lm.py --full                                       # all ~14.5M validation tokens (~8 min on mps)
#   python eval_lm.py checkpoints/ckpt.pt checkpoints/ckpt_before_attn_fix.pt

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from model import enc
from train import TOKENS_PATH, get_device
from generate import load_model, generate

PROMPTS_PATH = "eval_sets/prompts.txt"
# loss is averaged over these ranges of positions in the window (the model sees position + 1 tokens of context)
POSITION_BUCKETS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 128), (128, 256)]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_splits(tokens_path):
    # the same 90/10 split as train.py
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    n = int(0.9 * len(tokens))
    return tokens[:n], tokens[n:]


def window_starts(data, block_size, max_tokens):
    # non-overlapping windows of block_size tokens. a sample is spread evenly across the whole split
    # (not just its start), so it represents all of it, and it's the same windows every run.
    n_windows = (len(data) - 1) // block_size
    if max_tokens and max_tokens // block_size < n_windows:
        idx = np.linspace(0, n_windows - 1, max_tokens // block_size).astype(np.int64)
    else:
        idx = np.arange(n_windows)
    return idx * block_size


def windows(data, starts, block_size):
    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts]).astype(np.int64)
    return x, y


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@torch.no_grad()
def model_nll(model, data, starts, block_size, device, batch_size=32):
    # per-token loss for every target position, shape (n_windows, block_size)
    model.eval()
    out = np.empty((len(starts), block_size), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(starts), batch_size):
        x, y = windows(data, starts[i:i + batch_size], block_size)
        logits, _ = model(torch.from_numpy(x).to(device))
        nll = F.cross_entropy(logits.float().transpose(1, 2), torch.from_numpy(y).to(device), reduction="none")
        out[i:i + len(x)] = nll.cpu().numpy()
        done = min(i + batch_size, len(starts))
        if (i // batch_size) % 50 == 0 or done == len(starts):
            print(f"  {done:,} / {len(starts):,} windows ({time.time() - t0:.0f}s)", end="\r")
    print()
    return out


# ---------------------------------------------------------------------------
# n-gram baselines
# ---------------------------------------------------------------------------
def ngram_baselines(train, prev, targets, vocab_size, tune_tokens=1_000_000):
    # unigram and bigram models counted from the training split, scored on the same target tokens as the model.
    # the bigram mixes in the unigram so a pair never seen in training doesn't get probability zero:
    #   P(b | a) = lam * count(a, b) / count(a) + (1 - lam) * P(b)
    # lam is picked on the last 1M training tokens, which are left out of the counts.
    counts_from, tune = np.asarray(train[:-tune_tokens], dtype=np.int64), np.asarray(train[-tune_tokens:], dtype=np.int64)

    # add-one smoothing: every token gets counted at least once
    unigram = (np.bincount(counts_from, minlength=vocab_size) + 1).astype(np.float64)
    unigram /= unigram.sum()

    print("  counting bigrams...")
    codes, pair_counts = np.unique(counts_from[:-1] * vocab_size + counts_from[1:], return_counts=True)
    first_counts = np.bincount(counts_from[:-1], minlength=vocab_size)

    def bigram_prob(a, b):
        code = a * vocab_size + b
        pos = np.minimum(np.searchsorted(codes, code), len(codes) - 1)
        c_ab = np.where(codes[pos] == code, pair_counts[pos], 0)
        c_a = first_counts[a]
        return np.divide(c_ab, c_a, out=np.zeros(len(a)), where=c_a > 0)

    def bigram_nll(a, b, lam):
        return -np.log(lam * bigram_prob(a, b) + (1 - lam) * unigram[b])

    lam = min(np.arange(0.05, 1.0, 0.05), key=lambda l: bigram_nll(tune[:-1], tune[1:], l).mean())
    return {"uniform": np.full(targets.shape, math.log(vocab_size)),
            "unigram": -np.log(unigram[targets]),
            f"bigram (lam {lam:.2f})": bigram_nll(prev.ravel(), targets.ravel(), lam).reshape(targets.shape)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def token_byte_lengths(vocab_size):
    # how many bytes of raw text each token stands for
    return np.array([len(enc.decode_single_token_bytes(t)) for t in range(vocab_size)], dtype=np.int64)


def print_table(rows, n_bytes):
    print(f"\n{'':<34}{'loss':>8}{'perplexity':>12}{'bits/byte':>11}")
    for name, nll in rows:
        loss = float(nll.mean())
        print(f"{name:<34}{loss:>8.4f}{math.exp(loss):>12,.1f}{nll.sum() / math.log(2) / n_bytes:>11.4f}")


def print_positions(rows):
    print("\nloss by position in the window (more context to the right)")
    print(f"{'':<34}" + "".join(f"{f'{a}-{b - 1}' if b - a > 1 else str(a):>9}" for a, b in POSITION_BUCKETS))
    for name, nll in rows:
        print(f"{name:<34}" + "".join(f"{nll[:, a:b].mean():>9.3f}" for a, b in POSITION_BUCKETS))


def load_prompts(path):
    # one prompt per line; "\n" in a line becomes a real newline. blank lines and # comments are skipped
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").replace("\\n", "\n") for line in f if line.strip() and not line.startswith("#")]


def write_samples(path, checkpoint, prompts, max_new_tokens, seed):
    # sampled on the cpu, one fixed seed per prompt, so the same checkpoint gives the same text every time
    model = load_model(checkpoint, torch.device("cpu"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"samples from {checkpoint} (temperature 0.8, top_k 40, {max_new_tokens} tokens, seed {seed} + prompt number)\n")
        for i, prompt in enumerate(prompts):
            torch.manual_seed(seed + i)
            text = generate(model, prompt, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
            f.write(f"\n{'=' * 80}\nprompt {i}: {prompt!r}\n{'=' * 80}\n{text}\n")
    print(f"Wrote samples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained GPT checkpoints on a fixed slice of the validation split")
    parser.add_argument("checkpoints", nargs="*", default=["checkpoints/ckpt.pt"], help="Checkpoints saved by train.py")
    parser.add_argument("--tokens", default=TOKENS_PATH, help="Token cache written by train.py")
    parser.add_argument("--max_tokens", type=int, default=2_000_000, help="Validation tokens to score (evenly spaced windows)")
    parser.add_argument("--full", action="store_true", help="Score the whole validation split")
    parser.add_argument("--no_baselines", action="store_true", help="Skip the unigram/bigram baselines")
    parser.add_argument("--no_samples", action="store_true", help="Skip writing samples from the fixed prompts")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate per prompt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = get_device()
    train, val = load_splits(args.tokens)
    rows = []
    for path in args.checkpoints:
        model = load_model(path, device)
        block_size = model.config.block_size
        starts = window_starts(val, block_size, None if args.full else args.max_tokens)
        rows.append((os.path.basename(path), model_nll(model, val, starts, block_size, device)))
        del model

    prev, targets = windows(val, starts, block_size)
    n_bytes = int(token_byte_lengths(enc.n_vocab)[targets].sum())
    print(f"\nScored {targets.size:,} validation tokens ({len(starts):,} windows of {block_size}, {n_bytes:,} bytes of text)")
    if not args.no_baselines:
        rows = list(ngram_baselines(train, prev, targets, enc.n_vocab).items()) + rows

    print_table(rows, n_bytes)
    print_positions(rows)

    if not args.no_samples:
        prompts = load_prompts(PROMPTS_PATH)
        print()
        for path in args.checkpoints:
            name = os.path.splitext(os.path.basename(path))[0]
            write_samples(os.path.join("results", f"samples_{name}.txt"), path, prompts, args.max_new_tokens, args.seed)


if __name__ == "__main__":
    main()
