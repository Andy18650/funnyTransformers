"""Opt-in training telemetry: activation magnitudes, ALiBi attention behaviour,
and gradient extremes. All of it runs only on telemetry steps so normal steps
keep the fast fused-attention path and pay no overhead."""

from contextlib import contextmanager

import torch
from torch import nn

from ft.model import AlibiSelfAttention, FeedForward


def _activation_stats(tensor: torch.Tensor) -> dict[str, float]:
    absolute = tensor.detach().abs().float()
    return {
        "abs_max": absolute.max().item(),
        "abs_min": absolute.min().item(),
        "abs_mean": absolute.mean().item(),
    }


@contextmanager
def collect_forward_telemetry(model: nn.Module):
    """Within this context, attention modules record post-softmax attention
    summaries and every FFN activation records its magnitude stats. Yields a dict
    that is populated when the model is run inside the context.

    Usage:
        with collect_forward_telemetry(model) as stats:
            model(x)
        # stats now holds the aggregated telemetry
    """
    base = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    stats: dict[str, float] = {}
    handles = []
    attention_modules = []

    # Per-layer activation magnitudes, captured straight off the activation module
    # so we see the post-nonlinearity tensor regardless of FFN gating.
    activation_records: list[dict[str, float]] = []

    def make_hook():
        def hook(_module, _inputs, output):
            activation_records.append(_activation_stats(output))

        return hook

    for module in base.modules():
        if isinstance(module, AlibiSelfAttention):
            module.collect_attn_stats = True
            module.attn_stats = None
            attention_modules.append(module)
        elif isinstance(module, FeedForward):
            handles.append(module.activation.register_forward_hook(make_hook()))

    try:
        yield stats
    finally:
        for handle in handles:
            handle.remove()
        for module in attention_modules:
            module.collect_attn_stats = False

        # Aggregate activation magnitudes across all layers.
        if activation_records:
            stats["activation/abs_max"] = max(r["abs_max"] for r in activation_records)
            stats["activation/abs_min"] = min(r["abs_min"] for r in activation_records)
            stats["activation/abs_mean"] = sum(r["abs_mean"] for r in activation_records) / len(
                activation_records
            )

        # Aggregate attention summaries across all layers/heads.
        attn = [m.attn_stats for m in attention_modules if m.attn_stats is not None]
        if attn:
            stats["attention/first_token_attn_mean"] = sum(
                a["first_token_attn_mean"] for a in attn
            ) / len(attn)
            stats["attention/first_token_attn_min"] = min(a["first_token_attn_min"] for a in attn)
            stats["attention/distance_mean"] = sum(a["attn_distance_mean"] for a in attn) / len(attn)
            stats["attention/distance_max"] = max(a["attn_distance_max"] for a in attn)
        for module in attention_modules:
            module.attn_stats = None


@torch.no_grad()
def gradient_stats(model: nn.Module) -> dict[str, float]:
    """Absolute max/min/mean over all accumulated gradients. Call after backward
    (and after unscaling, if a GradScaler is used) to judge GradScaler health:
    abs_max near the fp16 ceiling or abs_min collapsing to 0 signals trouble."""
    abs_max = 0.0
    abs_min = float("inf")
    total = 0.0
    count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        absolute = parameter.grad.detach().abs().float()
        abs_max = max(abs_max, absolute.max().item())
        abs_min = min(abs_min, absolute.min().item())
        total += absolute.sum().item()
        count += absolute.numel()
    if count == 0:
        return {}
    return {
        "grad/abs_max": abs_max,
        "grad/abs_min": abs_min,
        "grad/abs_mean": total / count,
    }
