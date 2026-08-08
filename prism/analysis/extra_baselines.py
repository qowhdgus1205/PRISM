"""Classical ML and alternative DL backbones as comparison baselines.

By default these baselines consume the *feasible* input ``X`` (no ACP), so they
sit alongside ``simple_mlp`` / ``prism`` in the same leave-one-condition-out
table with **identical splits, standardization, and metrics**.  When
standardized ACP arrays are provided, the same model classes are also evaluated
as privileged-input diagnostics on ``[X, true ACP] -> Y`` with ``*_x_acp`` names.

    - ML  (RandomForest / XGBoost / LightGBM): does PRISM beat classical
           tabular regressors, not just other MLPs?
    - DL  (CNN / FT-Transformer backbones):    is PRISM's gain from the physics
           representation learning, or just from a fancier architecture?

Estimators that are not installed (xgboost, lightgbm, tab-transformer-pytorch)
are skipped with a warning instead of failing the whole run.

Row format matches the analysis scripts: ``{"model": name, **metric_fn(...)}``.
"""

from __future__ import annotations

import copy
import warnings
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism.models import build_encoder, build_target_head

try:
    from tqdm.auto import tqdm, trange
except ImportError:  # pragma: no cover - progress bars are optional
    tqdm = None
    trange = None

DEFAULT_ML_MODELS: tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")
DEFAULT_DL_BACKBONES: tuple[str, ...] = ("cnn", "ftt")


# ---------------------------------------------------------------------------
# Classical ML (scikit-learn / boosting)
# ---------------------------------------------------------------------------
def _build_sklearn(name: str, output_dim: int, seed: int):
    """Return an unfitted estimator, or ``None`` if the package is missing."""
    key = name.lower().strip()
    if key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        # RandomForest handles multi-output natively.
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)

    # Boosting libraries are pinned to n_jobs=1: nesting their internal thread
    # pool inside MultiOutputRegressor (multi-target Y) deadlocks in some
    # container/CPU-affinity setups. Datasets here are small, so single-threaded
    # fits cost a couple of seconds.
    if key == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return None
        base = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, verbosity=0,
        )
        return _wrap_multioutput(base, output_dim)

    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            return None
        base = LGBMRegressor(
            n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=1, verbose=-1,
        )
        return _wrap_multioutput(base, output_dim)

    raise ValueError(f"Unknown ML baseline '{name}'. Choices: {DEFAULT_ML_MODELS}")


def _wrap_multioutput(estimator, output_dim: int):
    """Boosting regressors are single-target; wrap when predicting >1 target."""
    if output_dim > 1:
        from sklearn.multioutput import MultiOutputRegressor

        return MultiOutputRegressor(estimator)
    return estimator


def _fit_predict_sklearn(estimator, X_tr, y_tr, X_te, output_dim: int) -> np.ndarray:
    """Fit on standardized arrays and return predictions shaped ``(n, output_dim)``."""
    if output_dim == 1:
        estimator.fit(X_tr, y_tr.ravel())
        return estimator.predict(X_te).reshape(-1, 1)
    estimator.fit(X_tr, y_tr)
    return np.asarray(estimator.predict(X_te)).reshape(-1, output_dim)


# ---------------------------------------------------------------------------
# Alternative DL backbones (supervised X -> Y, no ACP)
# ---------------------------------------------------------------------------
def _dl_regressor(backbone: str, input_dim: int, output_dim: int, latent_dim: int) -> nn.Module:
    """encoder(X -> z) + TargetMLP(z -> Y) as a plain supervised regressor."""
    encoder = build_encoder(backbone, input_dim=input_dim, latent_dim=latent_dim)
    head = build_target_head("mlp", input_dim=latent_dim, output_dim=output_dim)
    return nn.Sequential(encoder, head)


def _loader(*arrays, batch_size: int, shuffle: bool, drop_last: bool = False):
    tensors = [torch.tensor(a.astype(np.float32)) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _train_supervised(model, X_tr, y_tr, X_val, y_val, *, epochs, patience, lr, batch_size, device, desc="dl"):
    """Adam + MSE with early stopping on validation loss (mirrors train_mlp)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _loader(X_tr, y_tr, batch_size=batch_size, shuffle=True, drop_last=len(X_tr) > 1)
    val_loader = _loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    best_state, best_val, no_improve = None, float("inf"), 0
    if trange is not None:
        epoch_iter = trange(1, epochs + 1, desc=desc, leave=False, dynamic_ncols=True)
    else:
        epoch_iter = range(1, epochs + 1)
    for _ in epoch_iter:
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu())
                labels.append(yb)
        val = F.mse_loss(torch.cat(preds), torch.cat(labels)).item()
        if val < best_val:
            best_val, best_state, no_improve = val, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if trange is not None:
            epoch_iter.set_postfix(
                train=f"{float(np.mean(train_losses)):.4f}",
                val=f"{val:.4f}",
                best=f"{best_val:.4f}",
                patience=f"{no_improve}/{patience}",
            )
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_dl(model, X_te, device) -> np.ndarray:
    model.eval()
    loader = _loader(X_te, batch_size=4096, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def extra_baseline_predictions(
    *,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    acp_tr: np.ndarray | None = None,
    acp_val: np.ndarray | None = None,
    acp_te: np.ndarray | None = None,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    output_dim: int,
    latent_dim: int,
    device,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    seed: int,
    ml_models: Sequence[str] = DEFAULT_ML_MODELS,
    dl_backbones: Sequence[str] = DEFAULT_DL_BACKBONES,
    include_x_acp: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Fit ML + DL baselines and return ``(model_name, predictions)`` pairs.

    Predictions are over ``X_te`` in **raw units** (de-standardized with
    ``y_mean`` / ``y_std``), shaped ``(len(X_te), output_dim)``. Missing
    estimators are skipped with a warning. Used by both LOCO folds (whole fold
    is the test set) and stratified splits (predictions indexed per condition).
    """
    preds: list[tuple[str, np.ndarray]] = []
    has_privileged = include_x_acp and acp_tr is not None and acp_val is not None and acp_te is not None
    if has_privileged:
        X_tr_priv = np.concatenate([X_tr, acp_tr], axis=1)
        X_val_priv = np.concatenate([X_val, acp_val], axis=1)
        X_te_priv = np.concatenate([X_te, acp_te], axis=1)
    tasks = [*ml_models, *(f"{backbone}_mlp" for backbone in dl_backbones)]
    task_iter = tqdm(tasks, desc="extra_baselines", leave=False, dynamic_ncols=True) if tqdm is not None else tasks

    # --- Classical ML ---
    for name in ml_models:
        if tqdm is not None:
            task_iter.set_description(f"extra/{name}")
        estimator = _build_sklearn(name, output_dim, seed)
        if estimator is None:
            warnings.warn(f"[extra_baselines] '{name}' not installed; skipping.")
            if tqdm is not None:
                task_iter.update(1)
            continue
        pred = _fit_predict_sklearn(estimator, X_tr, y_tr, X_te, output_dim) * y_std + y_mean
        preds.append((name, pred))
        if has_privileged:
            estimator_priv = _build_sklearn(name, output_dim, seed)
            pred_priv = _fit_predict_sklearn(estimator_priv, X_tr_priv, y_tr, X_te_priv, output_dim) * y_std + y_mean
            preds.append((f"{name}_x_acp", pred_priv))
        if tqdm is not None:
            task_iter.update(1)

    # --- Alternative DL backbones ---
    # cuDNN is disabled here: CNNEncoder's Conv1d over a length-2..N "sequence"
    # can hit CUDNN_STATUS_NOT_INITIALIZED on some GPUs, and these tiny baselines
    # do not benefit from cuDNN acceleration anyway. FT-Transformer (attention)
    # is unaffected either way.
    with torch.backends.cudnn.flags(enabled=False):
        for backbone in dl_backbones:
            model_name = f"{backbone}_mlp"
            if tqdm is not None:
                task_iter.set_description(f"extra/{model_name}")
            try:
                model = _dl_regressor(backbone, X_tr.shape[1], output_dim, latent_dim)
            except ImportError as exc:
                warnings.warn(f"[extra_baselines] backbone '{backbone}' unavailable: {exc}; skipping.")
                if tqdm is not None:
                    task_iter.update(1)
                continue
            try:
                _train_supervised(
                    model, X_tr, y_tr, X_val, y_val,
                    epochs=epochs, patience=patience, lr=lr, batch_size=batch_size, device=device,
                    desc=f"extra/{model_name}",
                )
                pred = _predict_dl(model, X_te, device) * y_std + y_mean
            except RuntimeError as exc:
                warnings.warn(f"[extra_baselines] backbone '{backbone}' failed at runtime: {exc}; skipping.")
                if tqdm is not None:
                    task_iter.update(1)
                continue
            preds.append((model_name, pred))
            if has_privileged:
                try:
                    model_priv = _dl_regressor(backbone, X_tr_priv.shape[1], output_dim, latent_dim)
                    _train_supervised(
                        model_priv, X_tr_priv, y_tr, X_val_priv, y_val,
                        epochs=epochs, patience=patience, lr=lr, batch_size=batch_size, device=device,
                        desc=f"extra/{model_name}_x_acp",
                    )
                    pred_priv = _predict_dl(model_priv, X_te_priv, device) * y_std + y_mean
                    preds.append((f"{model_name}_x_acp", pred_priv))
                except RuntimeError as exc:
                    warnings.warn(f"[extra_baselines] privileged backbone '{backbone}' failed at runtime: {exc}; skipping.")
            if tqdm is not None:
                task_iter.update(1)

    if tqdm is not None:
        task_iter.close()

    return preds


def extra_baseline_rows(
    *,
    y_true_te: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], dict],
    **kwargs,
) -> list[dict]:
    """Convenience wrapper: predictions scored into ``{"model", **metrics}`` rows.

    For LOCO folds where the whole fold is the test set. Accepts the same keyword
    arguments as :func:`extra_baseline_predictions`.
    """
    return [
        {"model": name, **metric_fn(y_true_te, pred)}
        for name, pred in extra_baseline_predictions(**kwargs)
    ]
