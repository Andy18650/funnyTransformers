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
- Configurable feedforward `activation` (gelu, relu, tanh, sigmoid, silu,
  softplus, sqrt_softplus).
- Optional gated FFN via `ffn_gate` (`none` or `linear`) to build GLU variants
  such as SwiGLU (`activation: silu` + `ffn_gate: linear`).
- Optional **intra-document masking**: tokens never attend across document
  boundaries within a packed training window (toggle in the config).
- Training with Weights & Biases / SwanLab logging.
- **Multi-GPU training** via Distributed Data Parallel (DDP).

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

Train the Transformer on the prepared data:

```bash
uv run ft-train --config configs/transformer.yaml
```

Everything is one flat config file (`configs/transformer.yaml`): the dataset,
model architecture, all training hyperparameters, precision, and wandb settings
are top-level keys. Any key can be overridden on the command line with
`-o key=value`, repeatable, which is handy for quick sweeps or smoke tests:

```bash
uv run ft-train --config configs/transformer.yaml \
  -o dataset=fineweb_edu -o learning_rate=1e-4 -o steps=20000
```

A few ergonomic flags exist for things that change run-to-run without being part
of the experiment definition: `--precision`, `--note`, `--output-dir`, `--num-gpus`,
and `--no-wandb` (disable logging for a smoke test). Override values are coerced as
Python literals; an unknown key is written through silently and only surfaces as
a `KeyError` where it would be used, so typos fail loudly at the right place.

Checkpoints and the resolved `config.yaml` are written to
`checkpoints/<dataset>/<timestamp-id>/`, and `checkpoints/latest` is a symlink
updated to point at the most recent run.

The wandb run name can be templated via the `run_name` config key, a
`str.format` string over any config key plus `{param_count}` and `{note}` —
e.g. `"{activation}_{ffn_gate}_{param_count}"` to name a run after whatever you
are comparing. If unset, it defaults to `transformer_<dataset>_<param_count>[_note]`.

Training requires CUDA. Precision is selected with `precision` (config or
`--precision`):

- `fp16` (default): autocast with a `GradScaler`; works on all CUDA GPUs.
- `bf16`: autocast in bfloat16 (requires an Ampere+ GPU; exits with an error
  otherwise).
- `fp32`: full precision, for highest-precision sanity checks.

### Multi-GPU Training

For multi-GPU training using Distributed Data Parallel (DDP), use the `--num-gpus` flag:

```bash
# Train on 4 GPUs
uv run ft-train --config configs/transformer.yaml --num-gpus 4
```

DDP training is fully integrated with all existing features:
- Gradient synchronization across GPUs
- Rank-0 checkpoint saving and logging
- Progress bars and metrics only displayed on rank 0
- Compatible with mixed precision training (fp16/bf16)
- Works with all config overrides and flags

Single-GPU training remains the default behavior (no `--num-gpus` flag needed).

## Generation

Sample from the most recent run (the `latest` symlink), a handy fixed command
for a quick "does it generate English" sanity check:

```bash
uv run ft-generate \
  --checkpoint checkpoints/latest/best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```
