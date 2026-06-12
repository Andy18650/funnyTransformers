import argparse
from pathlib import Path

import torch

from ft.tokenization import encode_text, tokenizer_vocab_size, train_bpe_tokenizer

HF_DATASETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "splits": {"train": "train", "val": "validation"},
        "text_column": "text",
    },
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "splits": {"train": "train", "val": "validation"},
        "text_column": "text",
    },
    # Datasets below ship only a train stream; "val" is carved out of it.
    "tinystories_clean": {
        "path": "karpathy/tinystories-gpt4-clean",
        "splits": {"train": "train"},
        "text_column": "text",
    },
    "fineweb_edu": {
        "path": "karpathy/fineweb-edu-100b-shuffle",
        "splits": {"train": "train"},
        "text_column": "text",
    },
    "ultrafineweb": {
        "path": "openbmb/Ultra-FineWeb",
        # The English data lives in the "en" split; "zh"/"cn" is irrelevant here.
        "splits": {"train": "en"},
        "text_column": "content",
    },
}


def stream_hf_documents(spec: dict, split: str):
    """Yield non-empty documents (with an EOS marker) from one Hugging Face split."""
    from datasets import load_dataset

    dataset = load_dataset(spec["path"], spec.get("name"), split=split, streaming=True)
    text_column = spec["text_column"]
    for row in dataset:
        text = str(row[text_column]).strip()
        if not text:
            continue
        yield text + "<|EOS|>"


def take_chars(documents, max_chars: int) -> str:
    """Consume documents from an iterator until the char budget is reached."""
    parts = []
    char_count = 0
    for text in documents:
        if char_count + len(text) > max_chars:
            remaining = max_chars - char_count
            if remaining > 0:
                parts.append(text[:remaining])
            break
        parts.append(text)
        char_count += len(text)
    return "".join(parts)


def read_huggingface_dataset(dataset: str, max_chars: int) -> dict[str, str]:
    spec = HF_DATASETS[dataset]
    splits = spec["splits"]
    # Keep validation small when preparing a subset; this is enough for model comparison
    # and avoids WSL memory spikes from materializing full Hugging Face splits.
    eval_max_chars = max(1, max_chars // 10)

    if "val" in splits:
        # Dedicated validation split: read each stream independently.
        return {
            "train": take_chars(stream_hf_documents(spec, splits["train"]), max_chars),
            "val": take_chars(stream_hf_documents(spec, splits["val"]), eval_max_chars),
        }

    # No validation split: carve one from the head of the train stream, then keep
    # pulling from the same iterator for train so the two never overlap.
    documents = stream_hf_documents(spec, splits["train"])
    val_text = take_chars(documents, eval_max_chars)
    train_text = take_chars(documents, max_chars)
    return {"train": train_text, "val": val_text}


def prepare_bpe_data(
    texts: dict[str, str],
    dataset: str,
    output_path: Path,
    vocab_size: int,
) -> None:
    # Train the tokenizer only on training text, then apply the same vocabulary to all splits.
    tokenizer_meta = train_bpe_tokenizer(texts["train"], vocab_size=vocab_size)
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
        "test": torch.empty(0, dtype=torch.long),
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
        choices=[*HF_DATASETS],
        required=True,
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size for BPE, DEFAULTS TO 8000.")
    parser.add_argument(
        "--max-chars",
        type=int,
        required=True,
        help="Character limit. (5,000,000 should be safe, usually 4 times token number) "
        "This limits the train split; val receives a smaller limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = read_huggingface_dataset(args.dataset, args.max_chars)
    output_path = Path(args.output_dir) / f"{args.dataset}_bpe.pt"
    prepare_bpe_data(
        texts,
        args.dataset,
        output_path,
        args.vocab_size,
    )


if __name__ == "__main__":
    main()
