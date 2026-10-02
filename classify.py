# classify.py
# classifies a single email as spam or legitimate using the fine-tuned classifier from train_classifier.py.
# the message goes through the same formatting and encoding as the training data (prepare_labeled.py),
# so the model sees text in the shape it was trained on.
#
# examples:
#   python classify.py --subject "lunch friday?" --body "are we still on for friday at noon?"
#   python classify.py --subject "you have won" --file message.txt
#   cat message.txt | python classify.py

import sys
import argparse
import numpy as np

from prepare_labeled import LABELS, format_message, encode_messages
from evaluate import load_classifier, predict_probs
from train import get_device


def classify(model, threshold, subject, body, device):
    text = format_message(subject, body)
    tokens, lengths, truncated = encode_messages([text], model.config.block_size)
    p_spam = float(predict_probs(model, tokens.astype(np.int64), lengths.astype(np.int64), device)[0])
    return LABELS[int(p_spam >= threshold)], p_spam, truncated > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an email as spam or legitimate")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/classifier.pt", help="Classifier checkpoint saved by train_classifier.py")
    parser.add_argument("--subject", default="", help="Message subject")
    parser.add_argument("--body", default=None, help="Message body (or use --file, or pipe it in)")
    parser.add_argument("--file", default=None, help="Read the message body from this file")
    parser.add_argument("--threshold", type=float, default=None, help="Override the spam threshold stored in the checkpoint")
    args = parser.parse_args()

    if args.body is not None:
        body = args.body
    elif args.file is not None:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            body = f.read()
    else:
        body = sys.stdin.read()

    device = get_device()
    model, checkpoint = load_classifier(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else checkpoint["threshold"]

    label, p_spam, truncated = classify(model, threshold, args.subject, body, device)
    print(f"{label} (P(spam) = {p_spam:.3f}, threshold {threshold:.2f})")
    if truncated:
        print(f"note: only the first {model.config.block_size} tokens of the message were read")
