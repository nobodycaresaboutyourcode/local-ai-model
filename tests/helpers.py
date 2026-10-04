# helpers.py
# shared setup for the unit tests: a model small enough to build and run in milliseconds.

import torch

from model import Config, enc


# same architecture as the real model, just shrunk. dropout is off so outputs are repeatable.
# vocab_size defaults to the real GPT-2 vocabulary so real tokenized text can be fed in.
def tiny_config(vocab_size=enc.n_vocab, block_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0):
    return Config(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                  n_head=n_head, n_embd=n_embd, dropout=dropout)


def random_tokens(batch_size, length, vocab_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(vocab_size, (batch_size, length), generator=g)
