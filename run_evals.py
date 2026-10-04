# run_evals.py
# runs every evaluation against one pair of checkpoints and saves the results to results/<name>.json, so a
# later run (a retrained model, a quantized one) can be compared against it with compare.py.
#
# what gets recorded:
#   meta        - when, which git commit (and whether there were uncommitted changes), which checkpoints:
#                 a hash of each file so you can tell exactly which weights were measured, and its size on disk
#   classifier  - the test split report from evaluate.py: every metric with its 95% bootstrap interval,
#                 P(spam) for every test message, and false positives split by pretraining exposure
#   challenge   - challenge.py on the hand-written emails: rates, results by tag, invariance and directional tests,
#                 and P(spam) for every email
#   speed       - milliseconds to classify one email, and emails per second in batches
#   lm          - eval_lm.py on the fixed validation sample: loss, perplexity, bits per byte, loss by position
#
# examples:
#   python run_evals.py --name baseline
#   python run_evals.py --name int8 --classifier checkpoints/classifier_int8.pt --lm checkpoints/ckpt_int8.pt
#   python run_evals.py --name quick --skip_lm

import os
import json
import time
import hashlib
import argparse
import subprocess
from datetime import datetime
import numpy as np
import torch

from model import enc, count_parameters
from prepare_labeled import format_message, encode_messages
from evaluate import (load_split, load_classifier, predict_probs, all_metrics, bootstrap, load_data_checks,
                      split_fingerprint, CONTAM_THRESHOLD)
from challenge import CASES_PATH, load_cases, run as run_challenge
from eval_lm import load_splits, window_starts, windows, model_nll, token_byte_lengths, POSITION_BUCKETS
from generate import load_model
from train import TOKENS_PATH, get_device

RESULTS_DIR = "results"
# a typical short email for the latency measurement
LATENCY_EMAIL = ("Q3 planning meeting moved to Thursday",
                 "Hi team, the Q3 planning meeting is moving from Wednesday to Thursday at 10am, same room. "
                 "Please bring your draft roadmap and any headcount requests. Thanks, Sarah")


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------
def git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def file_info(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return {"path": path, "sha256": h.hexdigest()[:16], "size_mb": round(os.path.getsize(path) / 1e6, 1)}


def jsonable(x):
    # numpy numbers and arrays -> plain python, so json can write them
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    return x


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def classifier_section(model, threshold, data_dir, device):
    tokens, lengths, labels = load_split(data_dir, "test")
    probs = predict_probs(model, tokens, lengths, device)
    section = {"split_fingerprint": split_fingerprint(tokens, lengths, labels), "threshold": threshold,
               "metrics": all_metrics(probs, labels, threshold),
               "ci": {k: list(v) for k, v in bootstrap(probs, labels, threshold).items()},
               "probs": probs}
    checks = load_data_checks(data_dir, "test", tokens, lengths, labels)
    if checks is not None and "pretrain_overlap_test" in checks:
        seen = checks["pretrain_overlap_test"] >= CONTAM_THRESHOLD
        flagged = probs >= threshold
        section["legit_flagged_by_exposure"] = {
            name: [int(flagged[(labels == 0) & mask].sum()), int(((labels == 0) & mask).sum())]
            for name, mask in (("seen_in_pretraining", seen), ("not_seen", ~seen))}
    return section


def challenge_section(model, threshold, device, cases_path):
    cases = load_cases(cases_path)
    r = run_challenge("transformer", lambda t, l: predict_probs(model, t, l, device), threshold, cases,
                      model.config.block_size)
    labels, pred = r["labels"], r["pred"]
    tags = sorted({t for c in cases for t in c["tags"]})
    return {
        "threshold": threshold,
        "legit_flagged": [int(pred[labels == 0].sum()), int((labels == 0).sum())],
        "spam_caught": [int(pred[labels == 1].sum()), int((labels == 1).sum())],
        "wrong": [c["id"] for c, p, l in zip(cases, pred, labels) if p != l],
        "by_tag": {tag: [int(sum(p == l for c, p, l in zip(cases, pred, labels) if tag in c["tags"])),
                         sum(tag in c["tags"] for c in cases)] for tag in tags},
        "invariance": {t["name"]: {"n": t["n"], "unchanged": t["unchanged"], "flips": len(t["failures"])}
                       for t in r["invariance"]},
        "directional": {t["name"]: {"n": t["n"], "wrong_way": len(t["failures"]), "mean_delta": t["mean_delta"],
                                    "now_spam": t["now_spam"]} for t in r["directional"]},
        "edge_ok": r["edge_ok"],
        "probs": {c["id"]: p for c, p in zip(cases, r["probs"])},
    }


def speed_section(model, device, data_dir, repeats=30):
    # one email at a time - what a mail client would see
    tokens, lengths, _ = encode_messages([format_message(*LATENCY_EMAIL)], model.config.block_size)
    tokens, lengths = tokens.astype(np.int64), lengths.astype(np.int64)
    for _ in range(5):  # warm up: the first calls include one-time setup
        predict_probs(model, tokens, lengths, device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_probs(model, tokens, lengths, device)   # returns a numpy array, so it waits for the device
        times.append(time.perf_counter() - t0)
    # in batches - the whole test split
    x, L, _ = load_split(data_dir, "test")
    t0 = time.perf_counter()
    predict_probs(model, x, L, device)
    batch_seconds = time.perf_counter() - t0
    return {"device": str(device), "single_email_ms": float(np.median(times) * 1000),
            "batched_emails_per_s": len(L) / batch_seconds}


def lm_section(path, tokens_path, max_tokens, device):
    model = load_model(path, device)
    _, val = load_splits(tokens_path)
    block_size = model.config.block_size
    starts = window_starts(val, block_size, max_tokens)
    nll = model_nll(model, val, starts, block_size, device)
    _, targets = windows(val, starts, block_size)
    n_bytes = int(token_byte_lengths(enc.n_vocab)[targets].sum())
    loss = float(nll.mean())
    return {"tokens_scored": int(targets.size), "max_tokens": max_tokens, "loss": loss, "perplexity": float(np.exp(loss)),
            "bits_per_byte": float(nll.sum() / np.log(2) / n_bytes),
            "loss_by_position": {f"{a}-{b - 1}": float(nll[:, a:b].mean()) for a, b in POSITION_BUCKETS},
            "parameters": count_parameters(model)}


def main():
    parser = argparse.ArgumentParser(description="Run every evaluation and save the results for compare.py")
    parser.add_argument("--name", default=None, help="Name for this run (default: a timestamp); saved as results/<name>.json")
    parser.add_argument("--classifier", default="checkpoints/classifier.pt", help="Classifier checkpoint")
    parser.add_argument("--lm", default="checkpoints/ckpt.pt", help="Pretrained language model checkpoint")
    parser.add_argument("--data_dir", default="data", help="Where prepare_labeled.py wrote the splits")
    parser.add_argument("--cases", default=CASES_PATH, help="Challenge set file")
    parser.add_argument("--tokens", default=TOKENS_PATH, help="Pretraining token cache, for the language model evaluation")
    parser.add_argument("--lm_tokens", type=int, default=2_000_000, help="Validation tokens for the language model evaluation")
    parser.add_argument("--skip_lm", action="store_true", help="Skip the language model evaluation")
    args = parser.parse_args()

    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{name}.json")
    device = get_device()
    model, checkpoint = load_classifier(args.classifier, device)
    threshold = checkpoint["threshold"]

    results = {"meta": {"name": name, "created": datetime.now().isoformat(timespec="seconds"), **git_info(),
                        "torch": torch.__version__, "device": str(device),
                        "classifier": {**file_info(args.classifier), "parameters": count_parameters(model),
                                       "epoch": checkpoint.get("epoch")}}}
    print(f"Run '{name}': classifier {args.classifier}")

    print("  classifier on the test split...")
    results["classifier"] = classifier_section(model, threshold, args.data_dir, device)
    print("  challenge set...")
    results["challenge"] = challenge_section(model, threshold, device, args.cases)
    print("  speed...")
    results["speed"] = speed_section(model, device, args.data_dir)
    del model

    if not args.skip_lm:
        print(f"  language model {args.lm}...")
        results["meta"]["lm"] = file_info(args.lm)
        results["lm"] = lm_section(args.lm, args.tokens, args.lm_tokens, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(jsonable(results), f, indent=1)

    m, c = results["classifier"]["metrics"], results["challenge"]
    print(f"\nSaved {out_path}")
    print(f"  test F1 {m['f1']:.4f}, F0.5 {m['fbeta']:.4f}, ROC-AUC {m['roc_auc']:.4f} (threshold {threshold:.2f})")
    print(f"  challenge: legitimate flagged {c['legit_flagged'][0]}/{c['legit_flagged'][1]}, "
          f"spam caught {c['spam_caught'][0]}/{c['spam_caught'][1]}")
    print(f"  speed: {results['speed']['single_email_ms']:.1f} ms per email, "
          f"{results['speed']['batched_emails_per_s']:.0f} emails/s batched on {device}")
    if "lm" in results:
        print(f"  language model: loss {results['lm']['loss']:.4f}, perplexity {results['lm']['perplexity']:.1f}, "
              f"{results['lm']['bits_per_byte']:.4f} bits/byte")


if __name__ == "__main__":
    main()
