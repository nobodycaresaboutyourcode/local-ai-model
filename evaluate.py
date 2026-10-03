# evaluate.py
# measures how well the fine-tuned spam classifier does on messages it never trained on.
#
# accuracy alone is misleading here: ~75% of the messages are legitimate, so a model that calls
# EVERYTHING legitimate scores 75% while catching zero spam. instead we report:
#   precision - of the messages flagged as spam, how many really were spam
#   recall    - of the real spam, how much got flagged
#   F1        - balance of the two
#   F0.5      - like F1, but precision counts twice as much. sending a real email to the spam
#               folder costs more than letting one spam through, so this is what the threshold is tuned for.
#
# those all depend on the spam threshold. these don't, so they're fairer for comparing two models:
#   ROC-AUC          - chance a random spam message scores higher than a random legitimate one
#   PR-AUC           - average precision across every threshold; harsher than ROC-AUC when spam is rare
#   recall @ 1% FPR  - how much spam gets caught if at most 1 in 100 legitimate messages may be flagged
# and these check whether P(spam) can be taken at face value (lower is better):
#   Brier            - mean squared error of P(spam) against the true label
#   ECE              - expected calibration error: does "P(spam) = 0.9" really mean spam 90% of the time?
#
# the test set is small (128 spam messages), so one or two messages move every score noticeably.
# each number comes with a 95% bootstrap confidence interval: the test set is resampled with replacement
# 1,000 times and we report the range the middle 95% of those scores fall in. if two models' intervals
# overlap heavily, the test set can't tell them apart.
#
# examples:
#   python evaluate.py                                      # full report on the test set
#   python evaluate.py --baseline                           # also TF-IDF + logistic regression, and McNemar's test
#   python evaluate.py --errors 10                          # write every mistake (and 10 near misses) to results/
#   python evaluate.py checkpoints/classifier_seed*.pt      # several runs: one line each, plus mean +- std
#   python evaluate.py --tune_threshold                     # re-pick the spam threshold on val and save it

import os
import hashlib
import argparse
import numpy as np
import torch

from model import Config, GPTClassifier, enc
from prepare_labeled import LABELS
from train import get_device

BETA = 0.5          # F-beta weighting used to pick the spam threshold (precision counts 1/BETA = 2x as much)
MAX_FPR = 0.01      # operating point for "recall @ 1% FPR"
N_BOOTSTRAP = 1000
CONTAM_THRESHOLD = 0.5  # share of a message found in pretraining that counts as "seen" (same as check_data.py)


def load_split(data_dir, name):
    # the splits written by prepare_labeled.py
    d = np.load(os.path.join(data_dir, f"spam_{name}.npz"))
    return d["tokens"].astype(np.int64), d["lengths"].astype(np.int64), d["labels"].astype(np.int64)


def split_fingerprint(tokens, lengths, labels):
    # identifies a split's exact contents, so per-message results saved by check_data.py are
    # only used with the split they were computed on
    h = hashlib.sha1()
    for a in (tokens, lengths, labels):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPTClassifier(Config(**checkpoint["config"]), num_classes=len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# prediction and metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, tokens, lengths, device, batch_size=64):
    # returns P(spam) for every message
    was_training = model.training
    model.eval()
    probs = []
    for i in range(0, len(tokens), batch_size):
        L = torch.from_numpy(lengths[i:i + batch_size]).to(device)
        # padding is at the end, so each batch can be cut down to its longest message
        x = torch.from_numpy(tokens[i:i + batch_size, :int(L.max())]).to(device)
        logits, _ = model(x, L)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    model.train(was_training)
    return torch.cat(probs).numpy()


def binary_metrics(probs, labels, threshold=0.5, beta=BETA):
    pred = probs >= threshold
    tp = int((pred & (labels == 1)).sum())    # spam caught
    fp = int((pred & (labels == 0)).sum())    # legitimate mail wrongly flagged as spam
    fn = int((~pred & (labels == 1)).sum())   # spam that got through
    tn = int((~pred & (labels == 0)).sum())   # legitimate mail let through
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "accuracy": (tp + tn) / len(labels), "precision": precision,
            "recall": recall, "f1": f1, "fbeta": fbeta, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ranking_metrics(probs, labels, max_fpr=MAX_FPR, bins=10):
    # metrics that look at the scores themselves rather than one threshold
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, probs)
    # expected calibration error: group messages by P(spam) into equal-width bins, and in each bin
    # compare the average P(spam) with the share that really is spam, weighted by the bin's size
    bin_of = np.minimum((probs * bins).astype(int), bins - 1)
    ece = sum(abs(probs[bin_of == b].mean() - labels[bin_of == b].mean()) * (bin_of == b).mean()
              for b in range(bins) if (bin_of == b).any())
    return {"roc_auc": roc_auc_score(labels, probs),
            "pr_auc": average_precision_score(labels, probs),
            "recall_at_fpr": float(tpr[fpr <= max_fpr].max()),
            "brier": float(((probs - labels) ** 2).mean()),
            "ece": float(ece)}


def all_metrics(probs, labels, threshold):
    return {**binary_metrics(probs, labels, threshold), **ranking_metrics(probs, labels)}


def bootstrap(probs, labels, threshold, n=N_BOOTSTRAP, seed=0):
    # resample the messages with replacement and recompute every metric on each resample.
    # the threshold stays fixed: it was chosen on validation, so it isn't part of what's being measured.
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue  # a resample with only one class has no AUC - vanishingly rare at this size
        samples.append(all_metrics(probs[idx], labels[idx], threshold))
    return {k: np.percentile([s[k] for s in samples], [2.5, 97.5]) for k in samples[0] if k != "threshold"}


def choose_threshold(probs, labels, beta=BETA):
    # try cut-offs from 0.01 to 0.99 and keep the one with the best F-beta on the VALIDATION set.
    # never tune this on the test set - that would make the test scores look better than reality.
    #
    # a confident model often scores the same F-beta over a wide range of cut-offs (everything from
    # 0.01 to 0.6, say). taking the first of them puts the threshold right next to the legitimate
    # messages, so take the middle of the longest tied range instead - the cut-off with the most room
    # on both sides for messages the model hasn't seen.
    candidates = np.round(np.linspace(0.01, 0.99, 99), 2)
    scores = np.array([binary_metrics(probs, labels, t, beta)["fbeta"] for t in candidates])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    runs = np.split(best, np.flatnonzero(np.diff(best) > 1) + 1)
    run = max(runs, key=len)
    return float(candidates[run[len(run) // 2]])


def mcnemar(correct_a, correct_b):
    # McNemar's exact test: only the messages where the two models DISAGREE carry information.
    # if both models were equally good, each disagreement would be a coin flip between them.
    from scipy.stats import binomtest

    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def wilson_interval(k, n, z=1.96):
    # 95% confidence interval for a proportion k / n that behaves well when k or n is small
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def print_report(title, m):
    print(f"\n{title} (threshold {m['threshold']:.2f})")
    print(f"  accuracy  {m['accuracy']:.4f}")
    print(f"  precision {m['precision']:.4f}")
    print(f"  recall    {m['recall']:.4f}")
    print(f"  F1        {m['f1']:.4f}")
    print(f"  F{BETA:g}      {m['fbeta']:.4f}")
    print_confusion(m)


def print_confusion(m):
    print(f"                      predicted {LABELS[0]:<12} predicted {LABELS[1]}")
    print(f"  actual {LABELS[0]:<12} {m['tn']:>12,} {m['fp']:>22,}")
    print(f"  actual {LABELS[1]:<12} {m['fn']:>12,} {m['tp']:>22,}")


ROWS = [("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("F1", "f1"),
        (f"F{BETA:g}", "fbeta"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
        (f"recall @ {MAX_FPR:.0%} FPR", "recall_at_fpr"), ("Brier (lower better)", "brier"),
        ("ECE (lower better)", "ece")]


def print_detailed_report(title, probs, labels, threshold):
    m = all_metrics(probs, labels, threshold)
    ci = bootstrap(probs, labels, threshold)
    print(f"\n{title} (threshold {threshold:.2f})")
    print(f"  {'':<22}{'value':>8}   95% CI")
    for name, key in ROWS:
        lo, hi = ci[key]
        print(f"  {name:<22}{m[key]:>8.4f}   [{lo:.4f}, {hi:.4f}]")
    print_confusion(m)
    return m


def print_contamination_report(probs, labels, threshold, frac_seen):
    # nearly every legitimate test message was in the base model's pretraining data and nearly no
    # spam was, so "seen before" and "legitimate" are tangled together. the false positive rate on
    # legitimate mail the model has NOT seen is the closest this test set gets to new legitimate mail.
    seen = frac_seen >= CONTAM_THRESHOLD
    pred = probs >= threshold
    print(f"\nBy pretraining exposure (>= {CONTAM_THRESHOLD:.0%} of the message found in the pretraining corpus)")
    print(f"  {'':<34}{'messages':>9}{'flagged spam':>14}   95% CI")
    for name, label, mask in (("legitimate, seen in pretraining", 0, seen), ("legitimate, NOT seen", 0, ~seen),
                              ("spam, seen in pretraining", 1, seen), ("spam, NOT seen", 1, ~seen)):
        m = (labels == label) & mask
        k, n = int(pred[m].sum()), int(m.sum())
        lo, hi = wilson_interval(k, n)
        rate = f"{k / n:.1%}" if n else "-"
        print(f"  {name:<34}{n:>9,}{f'{k} ({rate})':>14}   [{lo:.1%}, {hi:.1%}]")


def load_data_checks(data_dir, split, tokens, lengths, labels):
    # per-message results from check_data.py, if they were computed on this exact split
    path = os.path.join(data_dir, "data_checks.npz")
    if not os.path.exists(path):
        return None
    checks = np.load(path)
    key = f"fingerprint_{split}"
    if key not in checks or str(checks[key]) != split_fingerprint(tokens, lengths, labels):
        print(f"\nnote: {path} is out of date for this split - re-run check_data.py for the pretraining exposure report")
        return None
    return checks


def write_errors(path, probs, labels, threshold, texts, frac_seen, near_misses):
    # every mistake, most confident first, plus the correct predictions that came closest to the threshold
    pred = probs >= threshold
    margin = np.abs(probs - threshold)
    groups = [
        ("FALSE POSITIVES - legitimate mail flagged as spam", np.flatnonzero(pred & (labels == 0)), -probs),
        ("FALSE NEGATIVES - spam that got through", np.flatnonzero(~pred & (labels == 1)), probs),
        (f"NEAR MISSES - the {near_misses} correct predictions closest to the threshold",
         np.flatnonzero(pred == (labels == 1)), margin),
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"threshold {threshold:.2f}\n")
        for title, idx, order in groups:
            idx = idx[np.argsort(order[idx])]
            if title.startswith("NEAR"):
                idx = idx[:near_misses]
            f.write(f"\n{'=' * 100}\n{title}: {len(idx)}\n{'=' * 100}\n")
            for i in idx:
                seen = "" if frac_seen is None else f" | {frac_seen[i]:.0%} seen in pretraining"
                f.write(f"\n[#{i}] {LABELS[labels[i]]} | P(spam) {probs[i]:.3f}{seen}\n{texts[i]}\n")
    print(f"\nWrote mistakes and near misses to {path}")


# ---------------------------------------------------------------------------
# baseline: TF-IDF + logistic regression
# ---------------------------------------------------------------------------
def decode_texts(tokens, lengths):
    return [enc.decode(t[:n - 1].tolist()) for t, n in zip(tokens, lengths)]


def fit_baseline(data_dir):
    # trains the baseline on the train split and tunes its threshold on val, the same way as the transformer.
    # returns a function from texts to P(spam), and the threshold.
    # imported here so scikit-learn is only needed when the baseline is requested
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # decode the stored tokens back to text, so the baseline sees exactly the same
    # (truncated) messages as the transformer - a fair comparison
    def texts(name):
        tokens, lengths, labels = load_split(data_dir, name)
        return decode_texts(tokens, lengths), labels

    train_x, train_y = texts("train")
    val_x, val_y = texts("val")

    # word and word-pair frequencies, down-weighting words that appear in nearly every message
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(vectorizer.fit_transform(train_x), train_y)

    def predict(texts):
        return clf.predict_proba(vectorizer.transform(texts))[:, 1]

    return predict, choose_threshold(predict(val_x), val_y)


def run_baseline(data_dir, split):
    predict, threshold = fit_baseline(data_dir)
    tokens, lengths, _ = load_split(data_dir, split)
    return predict(decode_texts(tokens, lengths)), threshold


# ---------------------------------------------------------------------------
# several checkpoints: how much does the score move between training runs?
# ---------------------------------------------------------------------------
def compare_runs(paths, tokens, lengths, labels, device):
    keys = ["precision", "recall", "f1", "fbeta", "roc_auc", "pr_auc"]
    print(f"\n{'checkpoint':<40}{'thresh':>7}" + "".join(f"{k:>10}" for k in keys) + f"{'FP':>5}{'FN':>5}")
    rows = []
    for path in paths:
        model, checkpoint = load_classifier(path, device)
        m = all_metrics(predict_probs(model, tokens, lengths, device), labels, checkpoint["threshold"])
        rows.append(m)
        print(f"{os.path.basename(path):<40}{m['threshold']:>7.2f}" + "".join(f"{m[k]:>10.4f}" for k in keys)
              + f"{m['fp']:>5}{m['fn']:>5}")
    print(f"{'mean':<47}" + "".join(f"{np.mean([r[k] for r in rows]):>10.4f}" for k in keys))
    print(f"{'std':<47}" + "".join(f"{np.std([r[k] for r in rows], ddof=1):>10.4f}" for k in keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the spam classifier on a held-out split")
    parser.add_argument("checkpoints", nargs="*", default=["checkpoints/classifier.pt"],
                        help="Classifier checkpoint(s) saved by train_classifier.py; several give a run-to-run comparison")
    parser.add_argument("--data_dir", default="data", help="Where prepare_labeled.py wrote the splits")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate a TF-IDF + logistic regression baseline (needs scikit-learn)")
    parser.add_argument("--errors", type=int, default=None, metavar="N",
                        help="Write every mistake plus the N closest correct calls to results/errors_<split>.txt")
    parser.add_argument("--tune_threshold", action="store_true",
                        help="Re-pick the spam threshold on the validation set and save it into the checkpoint")
    args = parser.parse_args()

    device = get_device()
    tokens, lengths, labels = load_split(args.data_dir, args.split)
    print(f"Evaluating on {args.split}: {len(labels):,} messages, {int(labels.sum()):,} spam")

    if len(args.checkpoints) > 1:
        compare_runs(args.checkpoints, tokens, lengths, labels, device)
        raise SystemExit

    path = args.checkpoints[0]
    model, checkpoint = load_classifier(path, device)
    print(f"Loaded {path} (epoch {checkpoint['epoch']}, val F1 {checkpoint['val_f1']:.4f})")

    if args.tune_threshold:
        x_val, len_val, y_val = load_split(args.data_dir, "val")
        old = checkpoint["threshold"]
        checkpoint["threshold"] = choose_threshold(predict_probs(model, x_val, len_val, device), y_val)
        torch.save(checkpoint, path)
        print(f"Spam threshold re-tuned on validation: {old:.2f} -> {checkpoint['threshold']:.2f} (saved to {path})")

    threshold = checkpoint["threshold"]
    probs = predict_probs(model, tokens, lengths, device)

    # the threshold chosen on the validation set, and the default 0.5 for comparison
    print_detailed_report(f"Transformer classifier on {args.split}", probs, labels, threshold)
    if threshold != 0.5:
        print_report(f"Transformer classifier on {args.split}", binary_metrics(probs, labels, 0.5))

    checks = load_data_checks(args.data_dir, args.split, tokens, lengths, labels)
    frac_seen = checks[f"pretrain_overlap_{args.split}"] if checks is not None and f"pretrain_overlap_{args.split}" in checks else None
    if frac_seen is not None:
        print_contamination_report(probs, labels, threshold, frac_seen)

    if args.baseline:
        base_probs, base_threshold = run_baseline(args.data_dir, args.split)
        print_detailed_report(f"Baseline TF-IDF + logistic regression on {args.split}", base_probs, labels, base_threshold)
        a_only, b_only, p = mcnemar((probs >= threshold) == (labels == 1), (base_probs >= base_threshold) == (labels == 1))
        print(f"\nMcNemar's test, transformer vs baseline: transformer alone right on {a_only}, baseline alone right on {b_only}"
              f" -> p = {p:.4f}" + (" (a real difference)" if p < 0.05 else " (could be chance)"))

    if args.errors is not None:
        write_errors(os.path.join("results", f"errors_{args.split}.txt"), probs, labels, threshold,
                     decode_texts(tokens, lengths), frac_seen, args.errors)
