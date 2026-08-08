# data_loader.py

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from prism import config


def check_required_columns(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {name} columns in CSV: {missing}")


def create_acp_derived_features(
    x_np: np.ndarray,
    acp_np: np.ndarray,
    input_cols: List[str],
    target_cols: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build optional ACP-derived supervision features.

    The current models use raw ACP targets by default. When enabled, these
    simple combustion timing spans give the ACP head extra shape information
    without depending on target labels.
    """
    del x_np, input_cols, target_cols
    if acp_np.shape[1] < 4:
        raise ValueError("ACP-derived features require MFB10, MFB50, MFB90, and p_max.")

    mfb10 = acp_np[:, 0]
    mfb50 = acp_np[:, 1]
    mfb90 = acp_np[:, 2]
    derived = np.stack(
        [
            mfb50 - mfb10,
            mfb90 - mfb50,
            mfb90 - mfb10,
        ],
        axis=1,
    ).astype(np.float32)
    names = ["CA10_50", "CA50_90", "CA10_90"]
    return derived, names


def get_data_loaders():
    # Load data
    csv_path = config.DATA_PATH
    if not pd.io.common.file_exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    input_cols = config.INPUT_COLS
    acp_cols = config.ACP_COLS
    target_cols = config.TARGET_COLS

    # Column checks
    check_required_columns(df, input_cols, "input")
    check_required_columns(df, acp_cols, "ACP")
    check_required_columns(df, target_cols, "target")

    # Numpy arrays (float32)
    X_np = df[input_cols].values.astype(np.float32)
    Y_acp_np = df[acp_cols].values.astype(np.float32)

    # Derived features from ACP and inputs
    if config.USE_DERIVED_ACP_FEATURES:
        acp_derived_np, _ = create_acp_derived_features(
            X_np, Y_acp_np, input_cols=input_cols, target_cols=target_cols
        )
    else:
        acp_derived_np = np.empty((X_np.shape[0], 0), dtype=np.float32)

    Y_full_np = np.concatenate([Y_acp_np, acp_derived_np], axis=1)

    # Tensors
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_full_tensor = torch.tensor(Y_full_np, dtype=torch.float32)
    y_tensor = torch.tensor(df[target_cols].values.astype(np.float32), dtype=torch.float32)

    return X_tensor, Y_full_tensor, y_tensor, df
