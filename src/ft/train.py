import argparse
import secrets
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from ft.data import get_batch, load_processed_data
from ft.model import build_transformer
from ft.telemetry import collect_forward_telemetry, gradient_stats
from ft.utils import (
    apply_overrides,
    count_parameters,
    load_config,
    perplexity,
    save_yaml,
    set_seed,
    update_latest_link,
    render_run_name,
)    


def maybe_init_wandb(config: dict):
    if config["wandb_mode"] == "disabled" or config.get("no_wandb"):
        return None

    if config["swanlab_mode"] != "disabled":
        import swanlab

        # SwanLab monkey-patches W&B logging, so this must happen before wandb.init().
        swanlab.sync_wandb(mode=config["swanlab_mode"])

    import wandb

    return wandb.init(
        project=config["wandb_project"],
        name=config["name"],
        group=config["dataset"],
        tags=[config["dataset"], "bpe", "transformer"],
        config=config,
        mode=config["wandb_mode"],
    )


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
def forward_telemetry(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    use_amp: bool,
) -> dict[str, float]:
    """Run one eager forward on the uncompiled model to collect activation and
    attention stats. Eager because the attention-score recompute can't run inside
    the fused SDPA path that torch.compile uses on the hot path."""
    base = getattr(model, "_orig_mod", model)
    was_training = base.training
    base.eval()
    with collect_forward_telemetry(base) as stats:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            base(x)
    if was_training:
        base.train()
    return stats


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


def train(config: dict) -> None:
    set_seed(config.get("seed", 42))

    # Training assumes CUDA; there is no point training these models on CPU.
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is required for training but is not available.")
    device = torch.device("cuda")

    # --- Precision setup -------------------------------------------------
    precision = config["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "error: bf16 precision requested but this GPU does not support it. "
            "Use precision fp16 or fp32."
        )
    use_amp = precision in ("bf16", "fp16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    print(f"precision: {precision} (autocast={use_amp})")

    data_path = str(Path(config["data_dir"]) / f"{config['dataset']}_bpe.pt")
    data = load_processed_data(data_path)
    eos_token_id = data.get("eos_token_id")

    model = build_transformer(
        config,
        vocab_size=data["vocab_size"],
        bos_token_id=data.get("bos_token_id"),
    ).to(device)
    config["param_count"] = count_parameters(model)
    config["name"] = render_run_name(config)
    config["data_path"] = data_path
    model = torch.compile(model)

    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    # wandb-style unique id: sortable timestamp plus random suffix so that runs
    # never collide, regardless of how similar their configurations are.
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    output_dir = Path(config.get("output_dir") or Path("checkpoints") / config["dataset"] / run_id)
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    update_latest_link(output_dir)

    run = maybe_init_wandb(config)
    if run is not None:
        run.summary["parameters"] = config["param_count"]
        run.summary["device"] = str(device)
        run.summary["precision"] = precision

    total_steps = config["steps"]
    train_log_interval = config.get("train_log_interval", 50)
    eval_interval = config.get("eval_interval", 500)
    eval_iters = config.get("eval_iters", 20)
    telemetry_interval = config.get("telemetry_interval", eval_interval)
    best_val_loss = float("inf")

    progress = trange(1, total_steps + 1, desc=config["name"])
    for step in progress:
        log_row = {}
        collect_telemetry = telemetry_interval and (step % telemetry_interval == 0 or step == total_steps)
        x, y = get_batch(data["train"], config["batch_size"], config["sequence_length"], device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = language_model_loss(logits, x, y, eos_token_id)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        grad_clip = config.get("grad_clip")
        # Unscale before clipping (so the threshold applies to real gradients) and
        # before reading gradient telemetry (so the stats reflect true gradients).
        if grad_clip is not None or collect_telemetry:
            scaler.unscale_(optimizer)
        if collect_telemetry:
            log_row.update(gradient_stats(model))
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if collect_telemetry:
            log_row.update(forward_telemetry(model, x, device, amp_dtype, use_amp))

        if step % train_log_interval == 0 or step == total_steps:
            log_row["train_loss"] = loss.item()
            progress.set_postfix(train_loss=f"{loss.item():.3f}")

        if step % eval_interval == 0 or step == total_steps:
            val_loss = evaluate(
                model,
                data["val"],
                config["batch_size"],
                config["sequence_length"],
                device,
                eval_iters,
                seed=config.get("seed", 42),
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
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--note", default=None, help="Optional suffix for the run name.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this run.")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        metavar="KEY=VALUE",
        help="Override any config key, e.g. -o steps=2000 -o learning_rate=1e-4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    # Named flags override the config only when explicitly provided.
    if args.precision is not None:
        config["precision"] = args.precision
    if args.note is not None:
        config["note"] = args.note
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.no_wandb:
        config["no_wandb"] = True
    train(config)


if __name__ == "__main__":
    main()
