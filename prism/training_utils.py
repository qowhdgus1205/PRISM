"""Shared training and evaluation loops."""

import copy

import torch
import torch.nn.functional as F

from prism.losses import monotonicity_loss
from prism.metrics import compute_metrics


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer,
    target_cols,
    epochs: int,
    patience: int,
    mono_pairs: list | None = None,
    mono_lambda: float = 0.0,
):
    """Train with early stopping on validation MSE."""
    best_state = None
    best_mse = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb, reduction="none")
            loss[:, 0] *= 2.0
            loss = loss.mean()
            if mono_pairs and mono_lambda > 0:
                loss = loss + mono_lambda * monotonicity_loss(xb, pred, mono_pairs)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                all_preds.append(model(xb).cpu())
                all_labels.append(yb.cpu())

        mse_each, _, _, _, _ = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))
        mse_mean = mse_each.mean().item()

        print(
            f"\r[Epoch {epoch:04}] MSE: "
            + " | ".join(f"{n}: {m.item():.5f}" for n, m in zip(target_cols, mse_each))
            + f" || Mean: {mse_mean:.5f}"
            + f" || Patience: {no_improve}/{patience}",
            end="", flush=True,
        )

        if mse_mean < best_mse:
            best_mse = mse_mean
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best: {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_on_test(model, test_loader, target_cols):
    """Evaluate on test set and return a metrics dictionary."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb).cpu())
            all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    mse_each, mape_each, rmse_each, _, _ = compute_metrics(y_true, y_pred)
    mse_mean = mse_each.mean().item()

    print("\n[TEST]")
    print("MSE : " + " | ".join(f"{n}: {m.item():.5f}" for n, m in zip(target_cols, mse_each)))
    print("MAPE: " + " | ".join(f"{n}: {m.item():.5f}" for n, m in zip(target_cols, mape_each)))
    print("RMSE: " + " | ".join(f"{n}: {m.item():.5f}" for n, m in zip(target_cols, rmse_each)))
    print(f"Mean MSE: {mse_mean:.5f}")
    return {
        "MSE": {n: m.item() for n, m in zip(target_cols, mse_each)},
        "MAPE": {n: m.item() for n, m in zip(target_cols, mape_each)},
        "RMSE": {n: m.item() for n, m in zip(target_cols, rmse_each)},
        "Mean_MSE": mse_mean,
    }
