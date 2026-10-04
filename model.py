# model_cheat.py
# borrowed heavily from Angelos Perivolaropoulos (https://github.com/angelos-p/llm-from-scratch)
# this builds a small GPT-style transformer that uses GPT-2's BPE tokenizer (via tiktoken),
# runs a forward pass on a sample batch of emails, and reports how many parameters the model has.
# nothing is trained here - the weights are randomly initialized. training comes later.

# I used Claude to help me put this together from the original source, 
# to help provide a foundation to work from

import math
import torch
import tiktoken
import torch.nn as nn
import torch.nn.functional as F

CORPUS_PATH = "enron_mails.csv"

# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
# instead of one token per character, we use GPT-2's byte pair encoding (same as tokenizer.py).
# BPE merges frequently-seen byte sequences into single tokens, so common words and word pieces
# become one token each. the vocabulary is fixed (50,257 tokens), so there's no need to scan the corpus.
enc = tiktoken.get_encoding("gpt2")


# grab just enough tokens from the start of the corpus to fill one sample batch.
# we don't know up front how many characters make num_tokens, so keep reading more until we have enough.
def read_sample_tokens(path, num_tokens, chunk_chars=64 * 1024):
    text = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while chunk := f.read(chunk_chars):
            text += chunk
            tokens = enc.encode_ordinary(text)
            if len(tokens) > num_tokens:
                return tokens[:num_tokens]
    raise ValueError(f"{path} has fewer than {num_tokens} tokens")


# ---------------------------------------------------------------------------
# model configuration
# ---------------------------------------------------------------------------
# these are deliberately small so the model runs on a laptop.
# vocab_size is filled in from the tokenizer at runtime.
class Config:
    def __init__(self, vocab_size, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1):
        self.vocab_size = vocab_size  # number of tokens in the BPE vocabulary
        self.block_size = block_size  # max context length the model can look at
        self.n_layer = n_layer        # number of transformer blocks stacked on top of each other
        self.n_head = n_head          # number of attention heads per block
        self.n_embd = n_embd          # size of each token's embedding vector
        self.dropout = dropout


# ---------------------------------------------------------------------------
# causal self attention
# ---------------------------------------------------------------------------
# each token looks at every token that came BEFORE it (never after - that's the "causal" part)
# and decides how much of each one to pull into its own representation.
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # one linear layer produces the query, key and value vectors for all heads at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projects the combined heads back into the embedding space
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # lower-triangular mask: position i may only attend to positions 0..i
        # registered as a buffer so it moves with the model but is NOT a trainable parameter
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding size

        # compute q, k, v and split them out per head: (B, n_head, T, head_size)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # reshape for multi-head (B, T, C) -> (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1,2).contiguous().view(B, T, C)
        # glue the heads back together, then project back into the embedding space
        return self.resid_dropout(self.c_proj(y))
    
        # attention scores: how well each query matches each key, scaled to keep softmax stable
        #att = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        # hide the future by setting those scores to -inf (softmax turns them into 0)
        #att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        #att = F.softmax(att, dim=-1)
        #att = self.attn_dropout(att)

        # weighted sum of the values, then glue the heads back together
        #y = att @ v
        #y = y.transpose(1, 2).contiguous().view(B, T, C)
        #return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# MLP (feed-forward) block
# ---------------------------------------------------------------------------
# attention lets tokens talk to each other; the MLP lets each token "think" about what it heard.
# it expands to 4x the embedding size, applies a non-linearity, and shrinks back down.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ---------------------------------------------------------------------------
# transformer block
# ---------------------------------------------------------------------------
# attention + MLP, each wrapped with a layer norm (pre-norm) and a residual connection.
# the residual "x + ..." lets information flow straight through deep stacks of blocks.
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# the full GPT model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # embeddings:
            #   wte - token embedding: turns each token id into a vector of size n_embd
            #   wpe - position embedding: tells the model WHERE in the sequence each token sits
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # language model head: turns the final vectors into a score (logit) for every token in the vocab
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the input embedding and output head share the same matrix (saves parameters)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # embeddings -> transformer blocks -> final layer norm.
    # shared by next-token prediction (forward) and classification (GPTClassifier)
    def hidden(self, idx):
        B, T = idx.shape
        # assert that the sequence length does not exceed the block size
        assert T <= self.config.block_size, f"sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd) - broadcast across the batch
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)

        return self.transformer.ln_f(x)       # (B, T, n_embd)

    # forward pass: token ids in -> logits (and optionally loss) out
    def forward(self, idx, targets=None):
        x = self.hidden(idx)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross entropy: how surprised the model is by the actual next token
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        return logits, loss


# ---------------------------------------------------------------------------
# spam classifier
# ---------------------------------------------------------------------------
# reuses the pretrained GPT body, but instead of predicting the next token at every position
# it reads ONE vector per message and turns it into a score for each class (legitimate / spam).
class GPTClassifier(nn.Module):
    def __init__(self, config, num_classes=2):
        super().__init__()
        self.config = config
        # the full GPT is kept so pretrained checkpoints load as-is. its lm_head is tied to wte,
        # so it adds no extra parameters - it just goes unused here.
        self.gpt = GPT(config)
        self.dropout = nn.Dropout(config.dropout)
        # classification head: embedding vector -> one score (logit) per class
        self.head = nn.Linear(config.n_embd, num_classes)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    # idx: (B, T) padded token ids, lengths: (B,) number of real tokens (including the final <|endoftext|>)
    def forward(self, idx, lengths, labels=None, class_weights=None):
        x = self.gpt.hidden(idx)  # (B, T, n_embd)
        # attention is causal, so the last real token is the only position that has seen the whole message.
        # its vector summarizes the message; padding after it never influences it.
        last = x[torch.arange(x.size(0), device=x.device), lengths.long() - 1]  # (B, n_embd)
        logits = self.head(self.dropout(last))  # (B, num_classes)

        loss = None
        if labels is not None:
            # class_weights lets the rarer class (spam) count for more, so the model can't just predict "legitimate"
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        return logits, loss

    # build a classifier on top of a GPT checkpoint saved by train.py.
    # the transformer weights come from the checkpoint; only the classification head starts random.
    @classmethod
    def from_pretrained(cls, checkpoint_path, num_classes=2, map_location="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        config = Config(**checkpoint["config"])
        model = cls(config, num_classes)
        model.gpt.load_state_dict(checkpoint["model"])
        return model

# ---------------------------------------------------------------------------
# parameter count
# ---------------------------------------------------------------------------
def count_parameters(model):
    # parameters() de-duplicates shared tensors, so the tied embedding/head is only counted once
    return sum(p.numel() for p in model.parameters())


def print_parameter_breakdown(model):
    config = model.config
    embd = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    per_block = sum(p.numel() for p in model.transformer.blocks[0].parameters())
    attn = sum(p.numel() for p in model.transformer.blocks[0].attn.parameters())
    mlp = sum(p.numel() for p in model.transformer.blocks[0].mlp.parameters())
    final_ln = sum(p.numel() for p in model.transformer.ln_f.parameters())
    total = count_parameters(model)

    print("\nParameter breakdown")
    print("-" * 48)
    print(f"{'token embeddings (wte)':<32}{model.transformer.wte.weight.numel():>16,}")
    print(f"{'position embeddings (wpe)':<32}{model.transformer.wpe.weight.numel():>16,}")
    print(f"{'  attention (per block)':<32}{attn:>16,}")
    print(f"{'  MLP (per block)':<32}{mlp:>16,}")
    print(f"{'transformer block (each)':<32}{per_block:>16,}")
    print(f"{f'transformer blocks (x{config.n_layer})':<32}{per_block * config.n_layer:>16,}")
    print(f"{'final layer norm':<32}{final_ln:>16,}")
    print(f"{'lm head (tied to wte)':<32}{0:>16,}")
    print("-" * 48)
    print(f"{'total':<32}{total:>16,}")
    print(f"\nTotal parameters: {total:,} (~{total / 1e6:.2f}M)")
    assert total == embd + per_block * config.n_layer + final_ln


if __name__ == "__main__":
    torch.manual_seed(1336)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. the vocabulary comes straight from the GPT-2 BPE tokenizer
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size:,} (GPT-2 BPE)")

    # 2. configure and create the model
    config = Config(vocab_size=vocab_size)
    model = GPT(config).to(device)

    # 3. forward pass on a sample batch pulled from the corpus
    # the size of the batch does not matter - 4 was just the count Claude suggested.
    batch_size = 4
    sample = read_sample_tokens(CORPUS_PATH, batch_size * config.block_size + 1)
    data = torch.tensor(sample, dtype=torch.long)
    # inputs are the tokens, targets are the same tokens shifted one to the right
    x = data[:-1].view(batch_size, config.block_size).to(device)
    y = data[1:].view(batch_size, config.block_size).to(device)

    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"\nForward pass on {device}")
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  logits shape: {tuple(logits.shape)}")
    # an untrained model should be roughly uniform over the vocab, so loss ~= ln(vocab_size)
    print(f"  loss:         {loss.item():.4f} (random guessing ~= {math.log(vocab_size):.4f})")

    # 4. parameter count
    print_parameter_breakdown(model)
