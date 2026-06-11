import argparse
from pathlib import Path

import requests
import torch

from ft.tokenization import encode_text, tokenizer_vocab_size, train_bpe_tokenizer


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HF_DATASETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "splits": {"train": "train", "val": "validation"},
        "text_column": "text",
    },
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "text_column": "text",
    },
}
DEFAULT_SAFE_MAX_CHARS = 5_000_000


def limit_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def split_text(text: str, train_ratio: float, val_ratio: float) -> dict[str, str]:
    train_end = int(train_ratio * len(text))
    val_end = int((train_ratio + val_ratio) * len(text))
    return {
        "train": text[:train_end],
        "val": text[train_end:val_end],
        "test": text[val_end:],
    }


def read_local_text(path: Path, max_chars: int | None) -> dict[str, str]:
    return {"all": limit_text(path.read_text(encoding="utf-8"), max_chars)}


def read_shakespeare(raw_dir: Path, max_chars: int | None) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "shakespeare.txt"

    if path.exists():
        return read_local_text(path, max_chars)

    response = requests.get(SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return {"all": limit_text(response.text, max_chars)}


def collect_hf_split(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    text_column: str,
    max_chars: int | None,
) -> str:
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, dataset_name, split=split, streaming=max_chars is not None)
    parts = []
    char_count = 0
    for row in dataset:
        text = str(row[text_column]).strip()
        if not text:
            continue
        text = text + "\n\n"
        if max_chars is not None and char_count + len(text) > max_chars:
            remaining = max_chars - char_count
            if remaining > 0:
                parts.append(text[:remaining])
            break
        parts.append(text)
        char_count += len(text)
    return "".join(parts)


def read_huggingface_dataset(dataset: str, max_chars: int | None) -> dict[str, str]:
    spec = HF_DATASETS[dataset]
    # Keep validation/test small when preparing a subset; this is enough for model comparison
    # and avoids WSL memory spikes from materializing full Hugging Face splits.
    eval_max_chars = max(1, max_chars // 20) if max_chars is not None else None
    split_limits = {
        "train": max_chars,
        "val": eval_max_chars,
        "test": eval_max_chars,
    }

    texts = {}
    for output_split, hf_split in spec["splits"].items():
        texts[output_split] = collect_hf_split(
            dataset_path=spec["path"],
            dataset_name=spec.get("name"),
            split=hf_split,
            text_column=spec["text_column"],
            max_chars=split_limits[output_split],
        )
    return texts


def read_or_download_dataset(dataset: str, raw_dir: Path, max_chars: int | None) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{dataset}.txt"

    if path.exists():
        return read_local_text(path, max_chars)
    if dataset == "shakespeare":
        return read_shakespeare(raw_dir, max_chars)
    if dataset in HF_DATASETS:
        return read_huggingface_dataset(dataset, max_chars)

    raise ValueError(f"Unsupported dataset: {dataset}")


def prepare_bpe_data(
    texts: dict[str, str],
    dataset: str,
    output_path: Path,
    train_ratio: float,
    val_ratio: float,
    lowercase: bool,
    vocab_size: int,
) -> None:
    if "all" in texts:
        texts = split_text(texts["all"], train_ratio=train_ratio, val_ratio=val_ratio)

    # Train the tokenizer only on training text, then apply the same vocabulary to all splits.
    tokenizer_meta = train_bpe_tokenizer(texts["train"], lowercase=lowercase, vocab_size=vocab_size)
    encoded = {
        split: torch.tensor(encode_text(text, tokenizer_meta), dtype=torch.long)
        for split, text in texts.items()
    }
    actual_vocab_size = tokenizer_vocab_size(tokenizer_meta)

    payload = {
        "dataset": dataset,
        "level": "bpe",
        "tokenizer": tokenizer_meta,
        "vocab_size": actual_vocab_size,
        "train": encoded["train"],
        "val": encoded["val"],
        "test": encoded.get("test", torch.empty(0, dtype=torch.long)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {dataset} BPE data to {output_path}")
    print(
        "Tokens: "
        f"train={len(payload['train']):,}, "
        f"val={len(payload['val']):,}, "
        f"test={len(payload['test']):,}; "
        f"vocabulary={actual_vocab_size:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BPE-tokenized language data.")
    parser.add_argument(
        "--dataset",
        choices=["shakespeare", "tinystories", "wikitext2"],
        default="shakespeare",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size for BPE.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional character limit. For Hugging Face datasets this limits the train split; val/test receive smaller limits.",
    )
    parser.add_argument(
        "--allow-full-dataset",
        action="store_true",
        help="Allow processing a full Hugging Face dataset without --max-chars. This can use a very large amount of memory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1.")
    if args.dataset in HF_DATASETS and args.max_chars is None and not args.allow_full_dataset:
        raise ValueError(
            f"Preparing the full {args.dataset} Hugging Face dataset can exhaust memory. "
            f"Pass --max-chars {DEFAULT_SAFE_MAX_CHARS} for a laptop-sized subset, or pass "
            "--allow-full-dataset if you intentionally want the full corpus."
        )

    texts = read_or_download_dataset(args.dataset, Path(args.raw_dir), args.max_chars)
    output_path = Path(args.output_dir) / f"{args.dataset}_bpe.pt"
    prepare_bpe_data(
        texts,
        args.dataset,
        output_path,
        args.train_ratio,
        args.val_ratio,
        args.lowercase,
        args.vocab_size,
    )


if __name__ == "__main__":
    main()
