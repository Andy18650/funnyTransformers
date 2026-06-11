from pathlib import Path

import torch


def get_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= sequence_length + 1:
        raise ValueError("Dataset split is too small for the configured sequence_length.")

    # Training uses random fixed-length windows rather than sequential passes over the corpus.
    token_ids = token_ids.to(torch.long)
    max_start = len(token_ids) - sequence_length - 1
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([token_ids[start : start + sequence_length] for start in starts])
    y = torch.stack([token_ids[start + 1 : start + sequence_length + 1] for start in starts])
    return x.to(device), y.to(device)


def load_processed_data(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
