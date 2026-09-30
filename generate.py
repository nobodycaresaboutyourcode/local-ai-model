# generate.py
# borrowed heavily from Angelos Perivolaropoulos (https://github.com/angelos-p/llm-from-scratch)
# loads a checkpoint saved by train.py and generates text from a prompt.
# the model was trained on GPT-2 byte pair encoding (BPE) tokens, so each step predicts a
# word or word piece rather than a single character. the tokenizer is fixed (tiktoken's "gpt2"),
# so unlike the character-level version there's no stoi/itos vocabulary stored in the checkpoint.
import torch

from model import Config, GPT, enc


@torch.no_grad()
def generate(model, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    # BPE-encode the prompt into GPT-2 token ids
    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        # never feed the model more context than it was trained on
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # only sample from the k most likely tokens to avoid rare garbage tokens
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    # decode the token ids back into text
    return enc.decode(idx[0].tolist())


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(checkpoint_path, device):
    # map_location lets a checkpoint trained on one device (e.g. cuda) load on another (e.g. mps)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # train.py stores the config as a plain dict, so rebuild the Config object from it
    config = Config(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print(f"Loaded {checkpoint_path} (step {checkpoint['step']}, val loss {checkpoint['val_loss']:.4f})")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/ckpt.pt", help="Path to checkpoint file saved by train.py")
    parser.add_argument("--prompt", default="Subject: ", help="Starting text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of BPE tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=40, help="Only sample from top-k most likely tokens")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, get_device())

    output = generate(model, args.prompt,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    print(output)
