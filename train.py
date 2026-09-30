# train.py
# borrowed heavily from Angelos Perivolaropoulos (https://github.com/angelos-p/llm-from-scratch)
# trains the multi-head attention GPT from model.py on the enron corpus using GPT-2's
# byte pair encoding (BPE) instead of one token per character.
#
# the corpus is ~1.3GB of text (~460 million BPE tokens), which is far too much to re-encode
# every run. the first run encodes it once and caches the token ids to disk; later runs
# memory-map that cache so batches are pulled straight from the file without loading it all.

import os
import math
import time
import numpy as np
import torch

from model import Config, GPT, enc, count_parameters

CORPUS_PATH = "enron_mails.csv"
TOKENS_PATH = "enron_tokens.bin"    # cached BPE token ids (uint16 - GPT-2's 50,257 ids fit in 16 bits)
CHECKPOINT_PATH = "checkpoints/ckpt.pt"

# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------
# model shape (must match between training and later loading of the checkpoint)
block_size = 256        # context length in tokens
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# training
batch_size = 16          # sequences per micro-batch
grad_accum_steps = 4     # micro-batches per optimizer step (effective batch = 16 * 4 * 256 tokens)
max_steps = 5000
warmup_steps = 200
max_lr = 6e-4
min_lr = 6e-5
weight_decay = 0.1
grad_clip = 1.0

# evaluation / logging
eval_interval = 250      # how often to measure train/val loss
eval_iters = 50          # batches averaged per loss estimate
log_interval = 10


# ---------------------------------------------------------------------------
# data: BPE-encode the corpus once, then memory-map the cached tokens
# ---------------------------------------------------------------------------
def build_token_cache(corpus_path, tokens_path, chunk_chars=8 * 1024 * 1024):
    # encode the corpus in chunks so we never hold all 1.3GB of text (or 460M python ints) in memory.
    # each chunk is extended to the next newline so we don't split a word across two chunks,
    # and the pieces of a chunk are encoded in parallel threads by tiktoken.
    print(f"Encoding {corpus_path} with GPT-2 BPE (one-time, cached to {tokens_path})...")
    tmp_path = tokens_path + ".tmp"
    total = 0
    start = time.time()
    with open(corpus_path, encoding="utf-8", errors="replace") as f, open(tmp_path, "wb") as out:
        while chunk := f.read(chunk_chars):
            chunk += f.readline()
            pieces = chunk.splitlines(keepends=True)
            ids = [t for piece in enc.encode_ordinary_batch(pieces, num_threads=os.cpu_count()) for t in piece]
            np.array(ids, dtype=np.uint16).tofile(out)
            total += len(ids)
            print(f"  {total:,} tokens ({time.time() - start:.0f}s)", end="\r")
    # only rename once finished, so an interrupted run doesn't leave a truncated cache behind
    os.replace(tmp_path, tokens_path)
    print(f"\n  done: {total:,} tokens in {time.time() - start:.0f}s")


def load_data(corpus_path, tokens_path, batch_size, block_size, device):
    if not os.path.exists(tokens_path):
        build_token_cache(corpus_path, tokens_path)

    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    print(f"Dataset size: {len(tokens):,} tokens, vocab_size = {enc.n_vocab:,}")

    # first 90% for training, last 10% for validation
    n = int(0.9 * len(tokens))
    splits = {"train": tokens[:n], "val": tokens[n:]}

    def get_batch(split):
        data = splits[split]
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        # inputs are the tokens, targets are the same tokens shifted one to the right
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    return get_batch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# learning rate schedule: linear warmup, then cosine decay from max_lr to min_lr
# ---------------------------------------------------------------------------
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------
# AdamW with weight decay applied only to 2D weights (linear layers, embeddings).
# biases and layer norm parameters are left alone - decaying them tends to hurt.
def configure_optimizer(model, weight_decay, lr, device):
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # the fused kernel is faster but only available on cuda
    fused = device.type == "cuda"
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


# ---------------------------------------------------------------------------
# evaluation and sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    # average the loss over several batches so the number isn't too noisy
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, prompt, max_new_tokens, device, temperature=0.8, top_k=50):
    # encode the prompt with BPE, repeatedly predict the next token, then decode back to text
    model.eval()
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        # never feed the model more context than it was trained on
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        # only sample from the k most likely tokens to avoid rare garbage tokens
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    model.train()
    return enc.decode(idx[0].tolist())


def save_checkpoint(model, optimizer, config, step, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": vars(config),
        "step": step,
        "val_loss": val_loss,
    }, path)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train():
    torch.manual_seed(1337)
    device = get_device()
    print(f"Using device: {device}")

    get_batch = load_data(CORPUS_PATH, TOKENS_PATH, batch_size, block_size, device)

    config = Config(vocab_size=enc.n_vocab, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=dropout)
    model = GPT(config).to(device)
    print(f"Model parameters: {count_parameters(model):,} (~{count_parameters(model) / 1e6:.2f}M)")

    optimizer = configure_optimizer(model, weight_decay, max_lr, device)
    # bfloat16 autocast speeds things up considerably on cuda; mps/cpu run in full float32
    use_amp = device.type == "cuda"
    tokens_per_step = batch_size * grad_accum_steps * block_size
    best_val_loss = float("inf")

    model.train()
    for step in range(max_steps + 1):
        # periodically check how we're doing on data the model hasn't trained on
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"step {step:>5} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, optimizer, config, step, best_val_loss, CHECKPOINT_PATH)
                print(f"  saved checkpoint to {CHECKPOINT_PATH}")
        if step == max_steps:
            break

        t0 = time.time()
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # gradient accumulation: run several micro-batches and sum their gradients,
        # which simulates a larger batch than fits in memory
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch("train")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        # clip gradients so one bad batch can't blow up the weights
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:>5} | loss {loss_accum:.4f} | lr {lr:.2e} | norm {norm:.2f} | "
                  f"{dt * 1000:.0f}ms | {tokens_per_step / dt:,.0f} tok/s")

    print("\nSample from the trained model:")
    print(generate(model, "Subject: ", max_new_tokens=200, device=device))


if __name__ == "__main__":
    train()
