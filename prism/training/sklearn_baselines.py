"""
sklearn_baselines.py
====================
Train tree-based baselines for the PRISM paper.

Models:
  - xgboost
  - lightgbm
  - random_forest

Feature sets:
  - x        : feasible direct baseline, X -> Y
  - oracle   : infeasible upper bound, [X, true ACP] -> Y
  - two_stage: feasible two-stage baseline, X -> ACP_hat then [X, ACP_hat] -> Y

Examples:
  python train_sklearn.py --model xgboost --feature-set x --seed 1 --output-dir ../results/seed_1
  python train_sklearn.py --model xgboost --feature-set oracle --seed 1 --output-dir ../results/seed_1
  python train_sklearn.py --model xgboost --feature-set two_stage --seed 1 --output-dir ../results/seed_1
"""

import argparse
import time
from pathlib import Path

import numpy as np

from prism import config
from prism.data_loader import get_data_loaders
from prism.split_utils import split_numpy_arrays
from prism.utils import save_json, set_seed


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6)), axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-9)
    return mse, rmse, mae, mape, r2


def build_model(model_name: str, seed: int):
    if model_name == "xgboost":
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor

        base = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=4,
        )

    raise ValueError(f"Unknown model: {model_name}. Choose from: xgboost, lightgbm, random_forest")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["xgboost", "lightgbm", "random_forest"])
    p.add_argument(
        "--feature-set",
        default="x",
        choices=["x", "oracle", "two_stage"],
        help="x: X->Y, oracle: [X,true ACP]->Y, two_stage: X->ACP_hat then [X,ACP_hat]->Y",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", default="../results/seed_1")
    return p.parse_args()


def prepare_features(args, X_fit, acp_fit, X_test, acp_test):
    if args.feature_set == "x":
        return X_fit, X_test, None

    if args.feature_set == "oracle":
        return np.concatenate([X_fit, acp_fit], axis=1), np.concatenate([X_test, acp_test], axis=1), None

    if args.feature_set == "two_stage":
        acp_model = build_model(args.model, args.seed)
        acp_model.fit(X_fit, acp_fit)
        acp_fit_hat = acp_model.predict(X_fit).astype(np.float32)
        acp_test_hat = acp_model.predict(X_test).astype(np.float32)
        return np.concatenate([X_fit, acp_fit_hat], axis=1), np.concatenate([X_test, acp_test_hat], axis=1), acp_model

    raise ValueError(f"Unknown feature-set: {args.feature_set}")


def main():
    args = parse_args()
    set_seed(args.seed)

    subdir = args.model if args.feature_set == "x" else f"{args.model}_{args.feature_set}"
    out_dir = Path(args.output_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[seed {args.seed}] Training {args.model} ({args.feature_set}) ...")

    X_tensor, _, y_tensor, df = get_data_loaders()
    X_np = X_tensor.numpy()
    acp_np = df[config.ACP_COLS].values.astype(np.float32)
    y_np = y_tensor.numpy()

    splits, _ = split_numpy_arrays(X_np, acp_np, y_np, seed=args.seed)
    X_train, X_val, X_test = splits[0]
    acp_train, acp_val, acp_test = splits[1]
    y_train, y_val, y_test = splits[2]

    X_fit = np.concatenate([X_train, X_val], axis=0)
    acp_fit = np.concatenate([acp_train, acp_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    t0 = time.time()
    X_model_fit, X_model_test, acp_model = prepare_features(args, X_fit, acp_fit, X_test, acp_test)
    model = build_model(args.model, args.seed)
    model.fit(X_model_fit, y_fit)
    elapsed = time.time() - t0

    y_pred = model.predict(X_model_test)
    mse, rmse, mae, mape, r2 = compute_metrics_np(y_test, y_pred)
    names = config.TARGET_COLS

    summary = {
        "model": args.model,
        "feature_set": args.feature_set,
        "seed": args.seed,
        "train_sec": round(elapsed, 2),
        "uses_true_acp_at_inference": args.feature_set == "oracle",
        "has_acp_predictor": acp_model is not None,
        "Mean_MSE": float(mse.mean()),
        "Mean_RMSE": float(rmse.mean()),
        "Mean_MAE": float(mae.mean()),
        "Mean_R2": float(r2.mean()),
        "MSE": {n: float(v) for n, v in zip(names, mse)},
        "RMSE": {n: float(v) for n, v in zip(names, rmse)},
        "MAE": {n: float(v) for n, v in zip(names, mae)},
        "MAPE": {n: float(v) for n, v in zip(names, mape)},
        "R2": {n: float(v) for n, v in zip(names, r2)},
    }

    save_json(summary, out_dir / "summary.json")
    print(f"  {args.model} ({args.feature_set})  Mean MSE: {summary['Mean_MSE']:.6f}  ({elapsed:.1f}s)")
    print(f"  Saved to {out_dir}/summary.json")


if __name__ == "__main__":
    main()
