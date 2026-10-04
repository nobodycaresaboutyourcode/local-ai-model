# challenge.py
# runs the spam classifier against eval_sets/challenge.txt: hand-written emails that look nothing like
# the Enron data it was trained and tested on. the test split can't tell us how the model does on new
# mail - nearly all of its legitimate messages were in the base model's pretraining data - so this is
# the closest thing we have to "does it work on my inbox?".
#
# three kinds of test, borrowed from the CheckList paper (Ribeiro et al., 2020):
#   labeled      - does each email get the right label? results are grouped by tag
#                  (transactional mail, phishing, obfuscated spam...) so you can see WHERE it breaks
#   invariance   - small edits that shouldn't change the answer: add a signature, swap names, prefix
#                  "Re:", put a friendly opener in front of spam. a flipped label is a failure.
#   directional  - edits that should push the score one way: adding a spammy call-to-action to a
#                  legitimate email should never make it LESS likely to be spam
#
# every email goes through the same formatting and truncation as classify.py, so this tests the
# whole path a real email would take.
#
# examples:
#   python challenge.py
#   python challenge.py --baseline     # also score the TF-IDF + logistic regression baseline

import os
import re
import argparse
import numpy as np

from prepare_labeled import LABELS, format_message, encode_messages
from evaluate import load_classifier, predict_probs, fit_baseline, decode_texts, wilson_interval
from train import get_device

CASES_PATH = "eval_sets/challenge.txt"
FAILURES_PATH = "results/challenge_failures.txt"
DIR_TOLERANCE = 0.05   # a directional test fails if P(spam) moves the wrong way by more than this


# ---------------------------------------------------------------------------
# loading the cases
# ---------------------------------------------------------------------------
def load_cases(path):
    # see the top of challenge.txt for the format
    cases = []
    case = None
    in_body = False
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("=== "):
            case = {"id": line[4:].strip(), "label": None, "tags": [], "subject": "", "body": []}
            cases.append(case)
            in_body = False
        elif case is None:
            continue  # comments before the first case
        elif in_body:
            case["body"].append(line)
        elif line.strip() == "---":
            in_body = True
        elif ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            if key == "label":
                case["label"] = LABELS.index(value)
            elif key == "tags":
                case["tags"] = [t.strip() for t in value.split(",") if t.strip()]
            elif key == "subject":
                case["subject"] = value
    for case in cases:
        case["body"] = "\n".join(case["body"]).strip()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    return cases


def edge_cases():
    # inputs with no right answer - they only have to come back with a sensible probability
    long_body = " ".join(["The quarterly report covers revenue, hiring and the product roadmap."] * 600)
    return [{"id": "edge-empty", "subject": "", "body": ""},
            {"id": "edge-subject-only", "subject": "quick question", "body": ""},
            {"id": "edge-very-long", "subject": "report", "body": long_body},
            {"id": "edge-symbols", "subject": "!!!", "body": "?!?! ... $$$ ### @@@"}]


# ---------------------------------------------------------------------------
# perturbations
# ---------------------------------------------------------------------------
NAME_SWAPS = {"Sarah": "Aisha", "Mike": "Tyrone", "Jen": "Mei", "David": "Rajesh", "Tom": "Diego",
              "Maria": "Olga", "Alex": "Kenji", "Priya": "Hannah"}
SIGNATURE = "\n\nBest,\nJordan Rivera\nOperations Manager | Brightline Logistics\n(555) 014-2290"
DISCLAIMER = ("\n\nCONFIDENTIALITY NOTICE: This email and any attachments are intended solely for the "
              "addressee and may contain confidential information. If you are not the intended recipient, "
              "please delete it and notify the sender.")
FRIENDLY_OPENER = "Hi Mike, it was great catching up at the conference last week. Hope the family is well.\n\n"
SPAM_CTA = ("\n\nCLICK HERE to claim your FREE $500 gift card before it expires tonight: "
            "http://claim-reward.example.net")


def swap_names(subject, body):
    pattern = re.compile(r"\b(" + "|".join(NAME_SWAPS) + r")\b")
    if not pattern.search(subject + " " + body):
        return None  # nothing to swap - test doesn't apply
    swap = lambda text: pattern.sub(lambda m: NAME_SWAPS[m.group(1)], text)
    return swap(subject), swap(body)


# (name, which labels it applies to, edit). an edit returns the new (subject, body), or None to skip
INVARIANCE = [
    ("uppercase everything", (0, 1), lambda s, b: (s.upper(), b.upper())),
    ("extra whitespace", (0, 1), lambda s, b: (s, re.sub(r" ", "  ", b).replace("\n", "\n\n"))),
    ("'Re:' subject prefix", (0, 1), lambda s, b: ("Re: " + s, b)),
    ("swap names", (0, 1), swap_names),
    ("add signature", (0, 1), lambda s, b: (s, b + SIGNATURE)),
    ("add legal disclaimer", (0, 1), lambda s, b: (s, b + DISCLAIMER)),
    ("friendly opener on spam", (1,), lambda s, b: (s, FRIENDLY_OPENER + b)),
]
# (name, which labels it applies to, edit, direction P(spam) must not go against: +1 up, -1 down)
DIRECTIONAL = [
    ("spammy call-to-action on legitimate", (0,), lambda s, b: (s, b + SPAM_CTA), +1),
]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def encode(cases, block_size):
    texts = [format_message(c["subject"], c["body"]) for c in cases]
    tokens, lengths, _ = encode_messages(texts, block_size)
    return tokens.astype(np.int64), lengths.astype(np.int64)


def perturb(cases, labels_allowed, edit):
    # returns (index of original case, edited case) for every case the edit applies to
    out = []
    for i, c in enumerate(cases):
        if c["label"] not in labels_allowed:
            continue
        edited = edit(c["subject"], c["body"])
        if edited is not None:
            out.append((i, {**c, "subject": edited[0], "body": edited[1]}))
    return out


def same_input(tokens_a, lengths_a, tokens_b, lengths_b):
    # an edit that disappears in preprocessing (e.g. uppercase, which normalize_text lowercases again)
    # or gets cut off by truncation can't change the answer, so it's counted separately
    return (lengths_a == lengths_b) & (tokens_a == tokens_b).all(axis=1)


def run(name, score, threshold, cases, block_size):
    tokens, lengths = encode(cases, block_size)
    probs = score(tokens, lengths)
    labels = np.array([c["label"] for c in cases])
    result = {"name": name, "threshold": threshold, "probs": probs, "pred": (probs >= threshold).astype(int),
              "labels": labels, "invariance": [], "directional": []}

    edge = edge_cases()
    edge_probs = score(*encode(edge, block_size))
    result["edge"] = [(c["id"], float(p)) for c, p in zip(edge, edge_probs)]
    result["edge_ok"] = bool(np.isfinite(edge_probs).all() and ((edge_probs >= 0) & (edge_probs <= 1)).all())

    for test_name, allowed, edit in INVARIANCE:
        pairs = perturb(cases, allowed, edit)
        if not pairs:
            result["invariance"].append({"name": test_name, "n": 0, "unchanged": 0, "failures": []})
            continue
        idx = np.array([i for i, _ in pairs])
        t2, l2 = encode([c for _, c in pairs], block_size)
        p2 = score(t2, l2)
        unchanged = same_input(tokens[idx], lengths[idx], t2, l2)
        flipped = ((p2 >= threshold) != (probs[idx] >= threshold)) & ~unchanged
        result["invariance"].append({"name": test_name, "n": len(pairs), "unchanged": int(unchanged.sum()),
                                     "failures": [(cases[i]["id"], float(probs[i]), float(p)) for i, p, f
                                                  in zip(idx, p2, flipped) if f]})

    for test_name, allowed, edit, direction in DIRECTIONAL:
        pairs = perturb(cases, allowed, edit)
        if not pairs:
            result["directional"].append({"name": test_name, "n": 0, "unchanged": 0, "mean_delta": 0.0,
                                          "now_spam": 0, "failures": []})
            continue
        idx = np.array([i for i, _ in pairs])
        t2, l2 = encode([c for _, c in pairs], block_size)
        p2 = score(t2, l2)
        unchanged = same_input(tokens[idx], lengths[idx], t2, l2)
        delta = p2 - probs[idx]
        wrong_way = (direction * delta < -DIR_TOLERANCE) & ~unchanged
        now_spam = (p2 >= threshold) & ~(probs[idx] >= threshold)
        result["directional"].append({"name": test_name, "n": len(pairs), "unchanged": int(unchanged.sum()),
                                      "mean_delta": float(delta[~unchanged].mean()) if (~unchanged).any() else 0.0,
                                      "now_spam": int(now_spam.sum()),
                                      "failures": [(cases[i]["id"], float(probs[i]), float(p)) for i, p, f
                                                   in zip(idx, p2, wrong_way) if f]})
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def fmt_rate(k, n):
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({k / n:.0%}) [{lo:.0%}-{hi:.0%}]" if n else "-"


def print_report(cases, results):
    labels = results[0]["labels"]
    names = [r["name"] for r in results]
    col = 26

    print(f"\nLabeled emails: {len(cases)} ({int((labels == 0).sum())} legitimate, {int((labels == 1).sum())} spam)")
    print(f"  {'':<34}" + "".join(f"{n:>{col}}" for n in names))
    for title, mask in (("legitimate flagged as spam", labels == 0), ("spam caught", labels == 1)):
        print(f"  {title:<34}" + "".join(f"{fmt_rate(int(r['pred'][mask].sum()), int(mask.sum())):>{col}}" for r in results))

    tags = sorted({t for c in cases for t in c["tags"]})
    print(f"\n  correct by tag {'':<19}" + "".join(f"{n:>{col}}" for n in names))
    for tag in tags:
        mask = np.array([tag in c["tags"] for c in cases])
        kinds = "/".join(sorted({LABELS[l][:5] for l in labels[mask]}))
        print(f"  {f'{tag} ({kinds})':<34}" + "".join(
            f"{fmt_rate(int((r['pred'][mask] == labels[mask]).sum()), int(mask.sum())):>{col}}" for r in results))

    print("\nInvariance - the label should NOT change")
    print(f"  {'edit':<34}{'applied':>8}{'no-op':>7}" + "".join(f"{f'flips ({n})':>{col}}" for n in names))
    for k, test in enumerate(results[0]["invariance"]):
        print(f"  {test['name']:<34}{test['n']:>8}{test['unchanged']:>7}"
              + "".join(f"{len(r['invariance'][k]['failures']):>{col}}" for r in results))
    print("  (no-op: the edit vanished in preprocessing or was cut off by truncation, so it can't change anything)")

    print(f"\nDirectional - P(spam) should not move the wrong way by more than {DIR_TOLERANCE}")
    print(f"  {'edit':<34}{'applied':>8}{'no-op':>7}" + "".join(f"{f'{n}: wrong way, mean dP':>{col + 8}}" for n in names))
    for k, test in enumerate(results[0]["directional"]):
        print(f"  {test['name']:<34}{test['n']:>8}{test['unchanged']:>7}" + "".join(
            f"{f'{len(d['failures'])}, {d['mean_delta']:+.3f} ({d['now_spam']} now spam)':>{col + 8}}"
            for d in (r["directional"][k] for r in results)))

    print("\nEdge cases (no right answer - just has to return a probability)")
    for r in results:
        print(f"  {r['name']}: " + ("ok, " if r["edge_ok"] else "FAILED, ")
              + ", ".join(f"{i} {p:.2f}" for i, p in r["edge"]))


def write_failures(path, cases, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{'=' * 100}\n{r['name']} (threshold {r['threshold']:.2f})\n{'=' * 100}\n")
            wrong = np.flatnonzero(r["pred"] != r["labels"])
            f.write(f"\nWRONG LABEL: {len(wrong)}\n")
            for i in wrong:
                c = cases[i]
                f.write(f"\n[{c['id']}] {LABELS[c['label']]} | P(spam) {r['probs'][i]:.3f} | tags: {', '.join(c['tags'])}\n"
                        f"Subject: {c['subject']}\n{c['body']}\n")
            for kind in ("invariance", "directional"):
                for test in r[kind]:
                    if test["failures"]:
                        f.write(f"\n{kind.upper()} FAILURES - {test['name']}: {len(test['failures'])}\n")
                        for case_id, before, after in test["failures"]:
                            f.write(f"  {case_id}: P(spam) {before:.3f} -> {after:.3f}\n")
            f.write("\n")
    print(f"\nWrote failing cases to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spam classifier against the hand-written challenge set")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/classifier.pt", help="Classifier checkpoint saved by train_classifier.py")
    parser.add_argument("--cases", default=CASES_PATH, help="Challenge set file")
    parser.add_argument("--data_dir", default="data", help="Where prepare_labeled.py wrote the splits (for --baseline)")
    parser.add_argument("--baseline", action="store_true", help="Also score the TF-IDF + logistic regression baseline")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    device = get_device()
    model, checkpoint = load_classifier(args.checkpoint, device)
    block_size = model.config.block_size
    print(f"Loaded {args.checkpoint}; {len(cases)} challenge emails from {args.cases}")

    results = [run("transformer", lambda t, l: predict_probs(model, t, l, device), checkpoint["threshold"], cases, block_size)]
    if args.baseline:
        predict, threshold = fit_baseline(args.data_dir)
        results.append(run("baseline", lambda t, l: predict(decode_texts(t, l)), threshold, cases, block_size))

    print_report(cases, results)
    write_failures(FAILURES_PATH, cases, results)
