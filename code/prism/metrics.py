"""Metric helpers for PRISM experiments."""

import torch
import torch.nn.functional as F


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Return MSE, MAPE, RMSE, MAE, and R2 per target."""
    mse = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
    rmse = torch.sqrt(mse)
    mape = (torch.abs((y_true - y_pred) / (y_true.abs() + 1e-6))).mean(dim=0)
    mae = torch.abs(y_pred - y_true).mean(dim=0)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot.clamp(min=1e-9)
    return mse, mape, rmse, mae, r2
