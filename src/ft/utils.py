import math
import random
from ast import literal_eval
from pathlib import Path
from typing import Any

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
