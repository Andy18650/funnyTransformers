import argparse
import os
import secrets
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from ft.data import get_batch, load_processed_data
from ft.models import build_model
from ft.prepare_data import HF_DATASETS
from ft.utils import count_parameters, load_yaml, perplexity, save_json, set_seed


def generate_run_id() -> str:
    # wandb-style unique id: sortable timestamp plus random suffix so that runs
    # never collide, regardless of how similar their configurations are.
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def update_latest_link(output_dir: Path, link_path: Path = Path("checkpoints/latest")) -> None:
    """Point a stable 'latest' symlink at this run's directory for easy reuse."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a relative target so the link survives the tree being moved/copied.
    target = Path(os.path.relpath(output_dir.resolve(), link_path.parent.resolve()))
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target, target_is_directory=True)
    except OSError as error:
        # Symlinks may be unavailable (e.g. some Windows setups); not fatal.
        print(f"warning: could not update {link_path} -> {target}: {error}")


def format_run_note(note: str | None) -> str:
    if not note:
        return ""
    normalized = "_".join(note.strip().split())
    return f"_{normalized}" if normalized else ""


def build_experiment_config(
    config: dict,
    dataset: str,
    data_dir: str,
    output_dir: str | None,
    wandb_project: str,
    wandb_mode: str,
    swanlab_mode: str,
    compile_model: bool,
    note: str | None,
    precision: str = "fp16",
) -> dict:
    model_config = dict(config["model"])
    resolved_output_dir = output_dir or str(Path("checkpoints") / dataset / generate_run_id())
    return {
        "name": f"{model_config['type'].lower()}_{dataset}",
        "dataset": dataset,
        "tokenizer": "bpe",
        "data_path": str(Path(data_dir) / f"{dataset}_bpe.pt"),
        "output_dir": resolved_output_dir,
        "model": model_config,
        "training": dict(config["training"]),
        "compile": compile_model,
        "precision": precision,
        "note": note,
        "wandb": {
            "project": wandb_project,
            "mode": wandb_mode,
            "swanlab_mode": swanlab_mode,
            "group": dataset,
            "tags": [dataset, "bpe", model_config["type"].lower()],
        },
    }


def maybe_init_wandb(config: dict, enabled: bool):
    if not enabled:
        return None

    wandb_config = config["wandb"]
    swanlab_mode = wandb_config.get("swanlab_mode", "disabled")
    if swanlab_mode != "disabled":
        import swanlab

        # SwanLab monkey-patches W&B logging, so this must happen before wandb.init().
        swanlab.sync_wandb(mode=swanlab_mode)

    import wandb

    return wandb.init(
        project=wandb_config["project"],
        name=config["name"],
        group=wandb_config["group"],
        tags=wandb_config["tags"],
        config=config,
        mode=wandb_config["mode"],
    )


def validate_data_metadata(data: dict) -> dict:
    if data.get("level") != "bpe" or data.get("tokenizer", {}).get("type") != "bpe":
        raise ValueError("Expected BPE processed data. Re-run ft.prepare_data.")
    return data


def language_model_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    targets = y
    if eos_token_id is not None:
        # Every document is followed by <bos>, so the target after an <eos> is always
        # <bos> -- a trivial constant prediction. Ignore those positions in the loss.
        targets = y.masked_fill(x == eos_token_id, -100)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    eval_iters: int,
    seed: int,
    eos_token_id: int | None,
    amp_dtype: torch.dtype | None = None,
    use_amp: bool = False,
) -> float:
    model.eval()
    # Use a local generator so validation loss is comparable across evaluation points.
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(token_ids, batch_size, sequence_length, device, generator=generator)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = language_model_loss(logits, x, y, eos_token_id)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict,
    data_meta: dict,
    step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "model_state": checkpoint_model.state_dict(),
            "config": config,
            "tokenizer": data_meta["tokenizer"],
            "vocab_size": data_meta["vocab_size"],
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def train(config: dict, disable_wandb: bool = False) -> None:
    training = config["training"]
    set_seed(training.get("seed", 42))

    # Training assumes CUDA; there is no point training these models on CPU.
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is required for training but is not available.")
    device = torch.device("cuda")

    # --- Precision setup -------------------------------------------------
    precision = config.get("precision", "fp16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "error: bf16 precision requested but this GPU does not support it. "
            "Use --precision fp16 or fp32."
        )
    use_amp = precision in ("bf16", "fp16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    print(f"precision: {precision} (autocast={use_amp})")

    data = validate_data_metadata(load_processed_data(config["data_path"]))

    eos_token_id = data.get("eos_token_id")
    model_config = dict(config["model"])
    if model_config["type"].lower() == "transformer":
        model_config.setdefault("max_sequence_length", training["sequence_length"])
        model_config["bos_token_id"] = data.get("bos_token_id")

    try:
        model = build_model(model_config, vocab_size=data["vocab_size"]).to(device)
    except RuntimeError as error:
        raise SystemExit(f"error: failed to initialize model on CUDA: {error}")
    param_count = count_parameters(model)
    config["name"] = (
        f"{model_config['type'].lower()}_{config['dataset']}_{param_count}"
        f"{format_run_note(config.get('note'))}"
    )
    if config.get("compile", False):
        model = torch.compile(model)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")
    update_latest_link(output_dir)

    run = maybe_init_wandb(config, enabled=not disable_wandb)
    if run is not None:
        run.summary["parameters"] = param_count
        run.summary["device"] = str(device)
        run.summary["precision"] = precision

    total_steps = training["steps"]
    train_log_interval = training.get("train_log_interval", 50)
    eval_interval = training.get("eval_interval", 500)
    eval_iters = training.get("eval_iters", 20)
    best_val_loss = float("inf")

    progress = trange(1, total_steps + 1, desc=config.get("name", "train"))
    for step in progress:
        log_row = {}
        x, y = get_batch(
            data["train"],
            training["batch_size"],
            training["sequence_length"],
            device,
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = language_model_loss(logits, x, y, eos_token_id)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        grad_clip = training.get("grad_clip")
        if grad_clip is not None:
            # Unscale before clipping so the threshold applies to real gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % train_log_interval == 0 or step == total_steps:
            log_row["train_loss"] = loss.item()
            progress.set_postfix(train_loss=f"{loss.item():.3f}")

        if step % eval_interval == 0 or step == total_steps:
            val_loss = evaluate(
                model,
                data["val"],
                training["batch_size"],
                training["sequence_length"],
                device,
                eval_iters,
                seed=training.get("seed", 42),
                eos_token_id=eos_token_id,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )
            log_row["val_loss"] = val_loss
            log_row["val_perplexity"] = perplexity(val_loss)
            progress.set_postfix(train_loss=f"{loss.item():.3f}", val_loss=f"{val_loss:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "best.pt", model, config, data, step, val_loss)

        if run is not None and log_row:
            run.log(log_row, step=step)

    save_checkpoint(output_dir / "last.pt", model, config, data, total_steps, best_val_loss)
    if run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE-tokenized language model.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[*HF_DATASETS],
        help="Prepared dataset name.",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to checkpoints/<dataset>/<timestamp-id>.",
    )
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--swanlab-mode",
        default="disabled",
        choices=["online", "local", "offline", "disabled"],
        help="Sync W&B logs to SwanLab. SwanLab sync is disabled by default.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile.")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp16",
        help="Mixed-precision autocast dtype. Defaults to fp16 (with GradScaler).",
    )
    parser.add_argument("--note", default=None, help="Optional suffix for the run name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_experiment_config(
        load_yaml(args.config),
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        swanlab_mode=args.swanlab_mode,
        compile_model=args.compile,
        note=args.note,
        precision=args.precision,
    )
    train(config, disable_wandb=args.no_wandb or args.wandb_mode == "disabled")


if __name__ == "__main__":
    main()
