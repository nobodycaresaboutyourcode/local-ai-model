# compare.py
# compares two runs saved by run_evals.py: what changed, by how much, and whether the change is bigger
# than the test set's own noise.
#
# with --gate it also decides whether the second run is an acceptable replacement for the first - for
# example a quantized model standing in for the full-precision one. it passes only if:
#   - test F0.5 stays inside the first run's 95% confidence interval
#   - at least 99% of test predictions are unchanged
#   - the challenge set gets at most 1 more email wrong, and at most 2 of its emails change label at all
#     (a model can keep the same score while getting different emails wrong - not a drop-in replacement)
#   - language model perplexity rises by no more than 2%
# and exits with status 1 if any check fails, so it can be used in a script. a check that can't run
# (e.g. one run has no language model results) is reported as SKIP and doesn't fail the gate.
#
# examples:
#   python compare.py results/baseline.json results/int8.json
#   python compare.py results/baseline.json results/int8.json --gate

import sys
import json
import argparse
import numpy as np

GATE_MIN_AGREEMENT = 0.99
GATE_MAX_EXTRA_CHALLENGE_WRONG = 1
GATE_MAX_CHALLENGE_CHANGED = 2
GATE_MAX_PERPLEXITY_INCREASE = 0.02

CLASSIFIER_ROWS = [("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"), ("F1", "f1"),
                   ("F0.5", "fbeta"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
                   ("recall @ 1% FPR", "recall_at_fpr"), ("Brier (lower better)", "brier"),
                   ("ECE (lower better)", "ece")]


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# what changed
# ---------------------------------------------------------------------------
def test_agreement(a, b):
    # share of test messages that get the same label from both runs, and how far P(spam) moved.
    # only meaningful if both runs scored exactly the same test split.
    ca, cb = a["classifier"], b["classifier"]
    if ca["split_fingerprint"] != cb["split_fingerprint"]:
        return None
    pa, pb = np.array(ca["probs"]), np.array(cb["probs"])
    same = (pa >= ca["threshold"]) == (pb >= cb["threshold"])
    return {"agreement": float(same.mean()), "changed": int((~same).sum()),
            "mean_abs_dp": float(np.abs(pa - pb).mean()), "max_abs_dp": float(np.abs(pa - pb).max())}


def challenge_changes(a, b):
    # emails whose label changed between the runs, and in which direction.
    # every challenge email has a right answer, so a changed label always means newly wrong or newly right.
    wa, wb = set(a["challenge"]["wrong"]), set(b["challenge"]["wrong"])
    return {"newly_wrong": sorted(wb - wa), "newly_right": sorted(wa - wb)}


def gate_checks(a, b):
    # (check, passed, detail) for every gate condition; passed is None when the check can't run
    checks = []
    lo, _ = a["classifier"]["ci"]["fbeta"]
    fb = b["classifier"]["metrics"]["fbeta"]
    checks.append(("test F0.5 within the first run's 95% CI", fb >= lo, f"{fb:.4f} vs lower bound {lo:.4f}"))

    agree = test_agreement(a, b)
    if agree is None:
        checks.append(("test predictions agree", False, "runs scored different test splits - not comparable"))
    else:
        checks.append((f"test predictions agree >= {GATE_MIN_AGREEMENT:.0%}", agree["agreement"] >= GATE_MIN_AGREEMENT,
                       f"{agree['agreement']:.2%} ({agree['changed']} changed)"))

    extra = len(b["challenge"]["wrong"]) - len(a["challenge"]["wrong"])
    checks.append((f"challenge set: at most {GATE_MAX_EXTRA_CHALLENGE_WRONG} more wrong",
                   extra <= GATE_MAX_EXTRA_CHALLENGE_WRONG,
                   f"{len(a['challenge']['wrong'])} -> {len(b['challenge']['wrong'])} wrong"))
    changes = challenge_changes(a, b)
    changed = len(changes["newly_wrong"]) + len(changes["newly_right"])
    checks.append((f"challenge set: at most {GATE_MAX_CHALLENGE_CHANGED} labels change", changed <= GATE_MAX_CHALLENGE_CHANGED,
                   f"{changed} changed ({len(changes['newly_wrong'])} now wrong, {len(changes['newly_right'])} now right)"))

    if "lm" in a and "lm" in b:
        rise = b["lm"]["perplexity"] / a["lm"]["perplexity"] - 1
        checks.append((f"perplexity rises <= {GATE_MAX_PERPLEXITY_INCREASE:.0%}", rise <= GATE_MAX_PERPLEXITY_INCREASE,
                       f"{a['lm']['perplexity']:.2f} -> {b['lm']['perplexity']:.2f} ({rise:+.2%})"))
    else:
        checks.append((f"perplexity rises <= {GATE_MAX_PERPLEXITY_INCREASE:.0%}", None,
                       "a run has no language model results (--skip_lm)"))
    return checks


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def print_meta(a, b):
    print(f"{'':<24}{a['meta']['name']:>28}{b['meta']['name']:>28}")
    for label, get in (("created", lambda r: r["meta"]["created"]),
                       ("git commit", lambda r: (r["meta"]["commit"] or "?")[:10] + ("+dirty" if r["meta"]["uncommitted_changes"] else "")),
                       ("classifier sha256", lambda r: r["meta"]["classifier"]["sha256"]),
                       ("classifier size (MB)", lambda r: r["meta"]["classifier"]["size_mb"]),
                       ("lm sha256", lambda r: r["meta"].get("lm", {}).get("sha256", "-")),
                       ("lm size (MB)", lambda r: r["meta"].get("lm", {}).get("size_mb", "-")),
                       ("device", lambda r: r["meta"]["device"])):
        print(f"{label:<24}{str(get(a)):>28}{str(get(b)):>28}")


def print_classifier(a, b):
    ma, mb = a["classifier"]["metrics"], b["classifier"]["metrics"]
    print(f"\nTest split{'':<14}{'first':>10}{'second':>10}{'change':>10}   first run's 95% CI")
    print(f"{'threshold':<24}{a['classifier']['threshold']:>10.2f}{b['classifier']['threshold']:>10.2f}")
    for name, key in CLASSIFIER_ROWS:
        lo, hi = a["classifier"]["ci"][key]
        note = "" if lo <= mb[key] <= hi else "   <- outside"
        print(f"{name:<24}{ma[key]:>10.4f}{mb[key]:>10.4f}{mb[key] - ma[key]:>+10.4f}   [{lo:.4f}, {hi:.4f}]{note}")
    for key in ("fp", "fn"):
        print(f"{key.upper() + (' (legit flagged)' if key == 'fp' else ' (spam missed)'):<24}{ma[key]:>10}{mb[key]:>10}{mb[key] - ma[key]:>+10}")
    agree = test_agreement(a, b)
    if agree is None:
        print("  (the runs scored different test splits, so per-message agreement can't be compared)")
    else:
        print(f"same label on {agree['agreement']:.2%} of test messages ({agree['changed']} changed); "
              f"P(spam) moved {agree['mean_abs_dp']:.4f} on average, {agree['max_abs_dp']:.4f} at most")


def print_challenge(a, b):
    ca, cb = a["challenge"], b["challenge"]
    print(f"\nChallenge set{'':<11}{'first':>10}{'second':>10}")
    for label, key in (("legitimate flagged", "legit_flagged"), ("spam caught", "spam_caught")):
        print(f"{label:<24}{f'{ca[key][0]}/{ca[key][1]}':>10}{f'{cb[key][0]}/{cb[key][1]}':>10}")
    print(f"{'wrong':<24}{len(ca['wrong']):>10}{len(cb['wrong']):>10}")
    for tag in sorted(set(ca["by_tag"]) | set(cb["by_tag"])):
        ta, tb = ca["by_tag"].get(tag, [0, 0]), cb["by_tag"].get(tag, [0, 0])
        print(f"  {tag:<22}{f'{ta[0]}/{ta[1]}':>10}{f'{tb[0]}/{tb[1]}':>10}")
    print(f"{'invariance flips':<24}{sum(t['flips'] for t in ca['invariance'].values()):>10}"
          f"{sum(t['flips'] for t in cb['invariance'].values()):>10}")
    changes = challenge_changes(a, b)
    if changes["newly_wrong"]:
        print(f"  now wrong: {', '.join(changes['newly_wrong'])}")
    if changes["newly_right"]:
        print(f"  now right: {', '.join(changes['newly_right'])}")


def print_speed_and_lm(a, b):
    sa, sb = a["speed"], b["speed"]
    print(f"\nSpeed{'':<19}{'first':>10}{'second':>10}{'ratio':>10}")
    print(f"{'ms per email':<24}{sa['single_email_ms']:>10.2f}{sb['single_email_ms']:>10.2f}{sb['single_email_ms'] / sa['single_email_ms']:>9.2f}x")
    print(f"{'emails/s batched':<24}{sa['batched_emails_per_s']:>10.0f}{sb['batched_emails_per_s']:>10.0f}"
          f"{sb['batched_emails_per_s'] / sa['batched_emails_per_s']:>9.2f}x")
    if sa["device"] != sb["device"]:
        print(f"  (measured on different devices: {sa['device']} vs {sb['device']})")
    if "lm" in a and "lm" in b:
        la, lb = a["lm"], b["lm"]
        print(f"\nLanguage model{'':<10}{'first':>10}{'second':>10}{'change':>10}")
        for label, key in (("loss", "loss"), ("perplexity", "perplexity"), ("bits per byte", "bits_per_byte")):
            print(f"{label:<24}{la[key]:>10.4f}{lb[key]:>10.4f}{lb[key] - la[key]:>+10.4f}")
        if la["tokens_scored"] != lb["tokens_scored"]:
            print(f"  (scored different amounts of text: {la['tokens_scored']:,} vs {lb['tokens_scored']:,} tokens)")


def main():
    parser = argparse.ArgumentParser(description="Compare two runs saved by run_evals.py")
    parser.add_argument("first", help="The reference run, e.g. results/baseline.json")
    parser.add_argument("second", help="The run to compare against it")
    parser.add_argument("--gate", action="store_true", help="Pass/fail: is the second run an acceptable replacement?")
    args = parser.parse_args()

    a, b = load(args.first), load(args.second)
    print_meta(a, b)
    print_classifier(a, b)
    print_challenge(a, b)
    print_speed_and_lm(a, b)

    if args.gate:
        checks = gate_checks(a, b)
        print("\nGate")
        for name, passed, detail in checks:
            print(f"  {'SKIP' if passed is None else 'PASS' if passed else 'FAIL'}  {name:<44}{detail}")
        ok = all(passed is not False for _, passed, _ in checks)
        print(f"\n{'PASSED' if ok else 'FAILED'}: {b['meta']['name']} "
              f"{'is' if ok else 'is NOT'} an acceptable replacement for {a['meta']['name']}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
