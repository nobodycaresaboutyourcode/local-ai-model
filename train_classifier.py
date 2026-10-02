# train_classifier.py
# fine-tunes the Enron GPT (checkpoints/ckpt.pt, from train.py) into a spam / legitimate classifier
# using the labeled splits written by prepare_labeled.py.
#
# differences from pretraining in train.py:
#   - one label per message instead of a next-token target at every position
#   - the dataset is small (~4,400 messages), so it's held in memory and we loop over it in
#     shuffled epochs rather than sampling random windows from a huge token file
#   - spam is the rarer class, so its loss is weighted up
#   - the pretrained transformer gets a much lower learning rate than the brand-new classification
#     head, and is frozen for the first epoch while the head learns - otherwise the random head's
#     gradients would scramble what the transformer already knows
#   - the best checkpoint is picked by validation F1, not loss, and the spam threshold is tuned afterwards

import math
import time
import argparse
import numpy as np
import torch

from model import GPTClassifier
from prepare_labeled import LABELS
from train import get_device, get_lr
from evaluate import BETA, load_split, predict_probs, binary_metrics, choose_threshold, print_report

PRETRAINED_PATH = "checkpoints/ckpt.pt"
DATA_DIR = "data"
CLASSIFIER_PATH = "checkpoints/classifier.pt"

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
batch_size = 32
epochs = 6
freeze_epochs = 1       # epochs at the start where only the classification head trains
body_lr = 1e-4          # pretrained transformer
head_lr = 1e-3          # new classification head
min_lr_frac = 0.1       # cosine decay ends at 10% of each learning rate
warmup_frac = 0.1       # fraction of all steps spent warming up
weight_decay = 0.01
grad_clip = 1.0
log_interval = 20


# ---------------------------------------------------------------------------
# optimizer: separate learning rates for the pretrained body and the new head
# ---------------------------------------------------------------------------
def configure_optimizer(model, device):
    body = list(model.gpt.parameters())
    groups = [
        # weight decay only on 2D weights, same as pretraining
        {"params": [p for p in body if p.dim() >= 2], "base_lr": body_lr, "weight_decay": weight_decay},
        {"params": [p for p in body if p.dim() < 2], "base_lr": body_lr, "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "base_lr": head_lr, "weight_decay": 0.0},
    ]
    for group in groups:
        group["lr"] = group["base_lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")


def set_body_trainable(model, trainable):
    # frozen parameters get no gradients, and AdamW skips parameters without gradients
    for p in model.gpt.parameters():
        p.requires_grad = trainable


def save_classifier(model, epoch, val_f1, threshold, path):
    torch.save({
        "model": model.state_dict(),
        "config": vars(model.config),
        "labels": LABELS,
        "threshold": threshold,
        "epoch": epoch,
        "val_f1": val_f1,
        "pretrained": PRETRAINED_PATH,
    }, path)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train(seed=1337, out_path=CLASSIFIER_PATH):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    print(f"Using device: {device}")

    x_train, len_train, y_train = load_split(DATA_DIR, "train")
    x_val, len_val, y_val = load_split(DATA_DIR, "val")
    print(f"Train: {len(y_train):,} messages ({int(y_train.sum()):,} spam), val: {len(y_val):,} ({int(y_val.sum()):,} spam)")

    model = GPTClassifier.from_pretrained(PRETRAINED_PATH, num_classes=len(LABELS), map_location=device).to(device)
    assert model.config.block_size == x_train.shape[1], "data was prepared with a different block_size than the pretrained model"
    print(f"Loaded pretrained transformer from {PRETRAINED_PATH}")

    # weight each class inversely to how common it is, so both classes contribute equally to the loss
    counts = np.bincount(y_train, minlength=len(LABELS))
    class_weights = torch.tensor(len(y_train) / (len(LABELS) * counts), dtype=torch.float32, device=device)
    print("Class weights: " + ", ".join(f"{name} {w:.2f}" for name, w in zip(LABELS, class_weights.tolist())))

    optimizer = configure_optimizer(model, device)
    steps_per_epoch = math.ceil(len(y_train) / batch_size)
    max_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_frac * max_steps)
    use_amp = device.type == "cuda"

    best_f1 = -1.0
    step = 0
    for epoch in range(epochs):
        frozen = epoch < freeze_epochs
        set_body_trainable(model, not frozen)
        model.train()
        if frozen:
            # a frozen transformer is just a fixed feature extractor, so switch off its dropout.
            # (this also matters on mps: attention with dropout isn't supported there when no gradients are needed)
            model.gpt.eval()
        print(f"\nepoch {epoch + 1}/{epochs}" + (" (transformer frozen, training head only)" if frozen else ""))

        t0 = time.time()
        order = rng.permutation(len(y_train))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            L = torch.from_numpy(len_train[idx]).to(device)
            # padding is at the end, so each batch can be cut down to its longest message
            x = torch.from_numpy(x_train[idx, :int(L.max())]).to(device)
            y = torch.from_numpy(y_train[idx]).to(device)

            # same warmup + cosine shape as pretraining, scaled to each group's own learning rate
            factor = get_lr(step, warmup_steps, max_steps, 1.0, min_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, L, y, class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_interval == 0:
                print(f"  step {step:>5} | loss {loss.item():.4f} | lr x{factor:.2f}")
            step += 1

        # check the model on messages it hasn't trained on, and keep the best version
        val_probs = predict_probs(model, x_val, len_val, device)
        m = binary_metrics(val_probs, y_val)
        print(f"  val | precision {m['precision']:.4f} | recall {m['recall']:.4f} | F1 {m['f1']:.4f} | {time.time() - t0:.0f}s")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            save_classifier(model, epoch + 1, best_f1, 0.5, out_path)
            print(f"  saved checkpoint to {out_path}")

    # reload the best epoch and tune the spam threshold on the validation set
    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_probs = predict_probs(model, x_val, len_val, device)
    threshold = choose_threshold(val_probs, y_val)
    save_classifier(model, checkpoint["epoch"], checkpoint["val_f1"], threshold, out_path)
    print(f"\nBest epoch {checkpoint['epoch']}; spam threshold tuned for F{BETA:g} on validation: {threshold:.2f}")
    print_report("Validation", binary_metrics(val_probs, y_val, threshold))
    print("\nRun evaluate.py for the final numbers on the held-out test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the pretrained GPT into a spam classifier")
    # different seeds change the head's starting weights and the order of the batches. training a few
    # seeds and comparing them (python evaluate.py checkpoints/classifier_seed*.pt) shows how much of a
    # score difference is just luck of the draw
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=CLASSIFIER_PATH, help="Where to save the classifier checkpoint")
    args = parser.parse_args()
    train(args.seed, args.out)
