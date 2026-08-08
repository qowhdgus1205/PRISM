from pathlib import Path
import json
from typing import Dict, Tuple
import pandas as pd

import torch
from torch.utils.data import Subset

from prism import config


def _split_counts(n_total: int) -> Tuple[int, int, int]:
    n_val = int(n_total * config.VAL_RATIO)
    n_test = int(n_total * config.TEST_RATIO)
    n_train = n_total - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Dataset split sizes became non-positive. Adjust val/test ratios.")
    return n_train, n_val, n_test


def get_split_path(seed: int | None = None) -> Path:
    split_seed = config.SEED if seed is None else seed
    split_dir = Path(getattr(config, "SPLIT_DIR", "./splits"))
    return split_dir / f"seed_{split_seed}_split.json"


def _validate_split(split: Dict[str, list], n_total: int) -> None:
    if int(split.get("n_total", -1)) != int(n_total):
        raise ValueError(
            f"Split file n_total={split.get('n_total')} does not match dataset size {n_total}."
        )

    indices = split.get("indices", {})
    required = ("train", "val", "test")
    missing = [name for name in required if name not in indices]
    if missing:
        raise ValueError(f"Split file is missing keys: {missing}")

    merged = []
    for name in required:
        merged.extend(indices[name])

    if len(merged) != n_total:
        raise ValueError("Split file does not cover the full dataset exactly once.")
    if len(set(merged)) != n_total:
        raise ValueError("Split file contains duplicate indices.")
    if min(merged) < 0 or max(merged) >= n_total:
        raise ValueError("Split file contains out-of-range indices.")


def load_or_create_split(n_total: int, seed: int | None = None):
    split_seed = config.SEED if seed is None else seed
    split_path = get_split_path(split_seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        _validate_split(split, n_total)
        return split, split_path

    n_train, n_val, n_test = _split_counts(n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(split_seed)).tolist()
    split = {
        "seed": split_seed,
        "n_total": n_total,
        "counts": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
        },
        "ratios": {
            "val": config.VAL_RATIO,
            "test": config.TEST_RATIO,
        },
        "indices": {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        },
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split, split_path


def split_dataset(dataset, seed: int | None = None):
    split, split_path = load_or_create_split(len(dataset), seed=seed)
    train_ds = Subset(dataset, split["indices"]["train"])
    val_ds = Subset(dataset, split["indices"]["val"])
    test_ds = Subset(dataset, split["indices"]["test"])
    return train_ds, val_ds, test_ds, split_path


def split_numpy_arrays(*arrays, seed: int | None = None):
    if not arrays:
        raise ValueError("split_numpy_arrays requires at least one array.")
    n_total = len(arrays[0])
    if any(len(arr) != n_total for arr in arrays):
        raise ValueError("All arrays must have the same length.")

    split, split_path = load_or_create_split(n_total, seed=seed)
    idx = split["indices"]

    def _slice(arr):
        return arr[idx["train"]], arr[idx["val"]], arr[idx["test"]]

    return [_slice(arr) for arr in arrays], split_path


def split_metadata_from_subsets(train_ds, val_ds, test_ds, split_path: Path) -> dict:
    return {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
        "split_file": str(split_path),
    }

def create_ood_split(df: pd.DataFrame, column: str, threshold: float,
                     mode: str = "greater", seed: int | None = None):
    """
    Creates an OOD split where the test set is samples where df[column] satisfies mode vs threshold.
    mode: "greater" or "less".
    seed: controls train/val shuffle; defaults to config.SEED.
    Returns train_indices, val_indices, test_indices.
    """
    import random
    split_seed = config.SEED if seed is None else seed

    if mode == "greater":
        test_mask = df[column] > threshold
    else:
        test_mask = df[column] < threshold

    test_idx      = df.index[test_mask].tolist()
    train_val_idx = df.index[~test_mask].tolist()

    rng = random.Random(split_seed)
    rng.shuffle(train_val_idx)

    n_val = int(len(train_val_idx) * (config.VAL_RATIO / (1 - config.TEST_RATIO)))
    val_idx   = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    return train_idx, val_idx, test_idx
