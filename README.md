# Funny Transformers

Goofing around with Transformer architectures. This repo is a small,
self-contained playground for training and comparing language-model
architectures on Hugging Face text datasets, starting from a baseline
Transformer (ALiBi attention, pre-norm blocks) and meant to grow with more
experimental architectures over time.

## Features

- BPE tokenization with `<|startoftext|>` / `<|endoftext|>` document markers.
- Streaming data preparation from Hugging Face datasets, with validation
  carved out of the train stream when a dataset ships no validation split.
- Baseline Transformer with ALiBi positional biases.
- Optional **intra-document masking**: tokens never attend across document
  boundaries within a packed training window (toggle in the config).
- Training with Weights & Biases / SwanLab logging.

## Setup

```bash
uv sync
```

Commands below use `uv run`; alternatively activate the environment and call
the `ft-prepare` / `ft-train` / `ft-generate` entry points directly.

## Data preparation

Prepare a BPE-tokenized subset of `fineweb-edu`:

```bash
uv run ft-prepare \
  --dataset fineweb_edu \
  --max-chars 5000000 \
  --vocab-size 8000 \
  --output-dir data/processed
```

`--max-chars` bounds the training text (≈4 characters per token, so 5,000,000
is a laptop-friendly start); the validation split gets a smaller slice. The
result is written to `data/processed/fineweb_edu_bpe.pt`.

Available datasets: `tinystories`, `wikitext2`, `tinystories_clean`,
`fineweb_edu`, `ultrafineweb`.

## Training

Train the baseline Transformer on the prepared data:

```bash
uv run ft-train \
  --config configs/transformer.yaml \
  --dataset fineweb_edu \
  --wandb-project funny-transformers
```

Model and training hyperparameters live in `configs/transformer.yaml`,
including `intra_doc_masking`. Checkpoints and the resolved config are written
to `checkpoints/<dataset>/<timestamp-id>/`. To run without logging, add
`--no-wandb` (or `--wandb-mode disabled`).

## Generation

Sample from a trained checkpoint:

```bash
uv run ft-generate \
  --checkpoint checkpoints/fineweb_edu/20260614-153012-a1b2c3/best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```
