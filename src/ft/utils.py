import math
import random
from ast import literal_eval
from pathlib import Path
from typing import Any
import os

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Patch flat config keys from 'key=value' strings. Values are parsed as
    Python literals (int/float/bool/list/...) when possible, else kept as str.
    Unknown keys are written through without complaint -- a typo simply surfaces
    later as a KeyError where the value is actually used."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        try:
            value = literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        config[key] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def update_latest_link(output_dir: Path, link_path: Path = Path("checkpoints/latest")) -> None:
    """Point a stable 'latest' symlink at this run's directory for easy reuse."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    # Relative target so the link survives the tree being moved/copied.
    target = Path(os.path.relpath(output_dir.resolve(), link_path.parent.resolve()))
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)


def render_run_name(config: dict) -> str:
    """Build the run name. If run_name is set it is a str.format template over the
    flat config (e.g. '{activation}_{ffn_gate}_{param_count}'); a bad field raises
    KeyError. Otherwise fall back to transformer_<dataset>_<param_count>[_note]."""
    template = config.get("run_name")
    if template:
        name = template.format(**config)
    else:
        name = f"transformer_{config['dataset']}_{config['param_count']}"
        if config.get("note"):
            name = f"{name}_{config['note']}"
    return "_".join(name.strip().split())  # normalize whitespace
