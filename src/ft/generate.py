import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ft.model import build_transformer
from ft.tokenization import decode_tokens, encode_text


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: list[int],
    tokenizer_meta: dict,
    max_new_tokens: int,
    sequence_length: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    eos_token_id = tokenizer_meta.get("eos_token_id")
    ids = list(input_ids)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-sequence_length:]], dtype=torch.long, device=device)
        logits = model(context)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id == eos_token_id:
            break
        ids.append(next_id)
    return decode_tokens(ids, tokenizer_meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="To be or not to")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = build_transformer(config, vocab_size=checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    tokenizer_meta = checkpoint["tokenizer"]

    # Condition on <bos> just like every document seen during training.
    bos_token_id = tokenizer_meta.get("bos_token_id")
    input_ids = encode_text(args.prompt, tokenizer_meta)
    if bos_token_id is not None:
        input_ids = [bos_token_id, *input_ids]
    text = generate(
        model,
        input_ids,
        tokenizer_meta,
        args.max_new_tokens,
        config["sequence_length"],
        args.temperature,
        device,
    )
    print(text)


if __name__ == "__main__":
    main()
