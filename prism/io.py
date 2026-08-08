"""I/O helpers for PRISM experiments."""

import json
from pathlib import Path
from typing import Any

import torch


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_torch_state(path: Path, map_location=None):
    """Load a state_dict checkpoint with PyTorch safe tensor-only loading."""
    return torch.load(path, map_location=map_location, weights_only=True)
