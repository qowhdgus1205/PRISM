#!/usr/bin/env python
"""Measure how well feasible inputs X can recover privileged ACP variables.

This diagnostic is used to support the Case2 interpretation: oracle models can
benefit from true ACP, but feasible PRISM variants are limited when X does not
contain enough information to reconstruct ACP at deployment time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from prism.encoder import build_role_model
from prism.reproducibility import set_seed
from run_experiments import (
    DATASETS,
    encode_categorical_features,
    loader,
    make_arrays,
    numeric_role_columns,
    predict,
    split_diagnostics,
    split_indices,
    train_supervised,
)


GROUPS = {
    "Main": [
        "airfoil_self_noise",
        "energy_efficiency",
        "hydraulic_systems",
        "household_power_consumption",
        "concrete_slump_test",
    ],
    "Case1": [
        "wine_quality",
        "superconductivity",
        "auto_mpg",
        "combined_cycle_power_plant",
    ],
    "Case2": [
        "air_quality",
        "parkinsons_telemonitoring",
        "real_estate_valuation",
        "student_performance",
    ],
}


def score_acp(a_true_std, a_pred_std, acp_cols):
    err = a_pred_std - a_true_std
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((a_true_std - a_true_std.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "macro_r2": float(np.mean(r2)),
        "macro_std_rmse": float(np.mean(rmse)),
        "macro_std_mae": float(np.mean(mae)),
        "acp_r2_json": json.dumps({k: float(v) for k, v in zip(acp_cols, r2)}, sort_keys=True),
        "acp_std_rmse_json": json.dumps({k: float(v) for k, v in zip(acp_cols, rmse)}, sort_keys=True),
    }


def prepare_dataset(spec, seed, args):
    raw = encode_categorical_features(pd.read_csv(spec.path))
    x_cols, acp_cols, y_cols, dropped = numeric_role_columns(raw, spec)
    needed = x_cols + acp_cols + y_cols
    df = raw.dropna(subset=needed).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"{spec.name}: not enough complete rows after dropping NaNs ({len(df)})")
    if not x_cols or not acp_cols:
        raise ValueError(f"{spec.name}: missing role columns x={len(x_cols)} acp={len(acp_cols)}")

    train_idx, val_idx, test_idx = split_indices(df, spec, seed, args.test_size, args.val_size)
    diag = split_diagnostics(df, spec, train_idx, val_idx, test_idx, y_cols)
    X_tr, x_scaler = make_arrays(df, x_cols, train_idx, fit=True)
    X_val = make_arrays(df, x_cols, val_idx, scaler=x_scaler)
    X_te = make_arrays(df, x_cols, test_idx, scaler=x_scaler)
    A_tr, a_scaler = make_arrays(df, acp_cols, train_idx, fit=True)
    A_val = make_arrays(df, acp_cols, val_idx, scaler=a_scaler)
    A_te = make_arrays(df, acp_cols, test_idx, scaler=a_scaler)
    return df, x_cols, acp_cols, y_cols, dropped, diag, X_tr, X_val, X_te, A_tr, A_val, A_te


def fit_predict(model_name, X_tr, X_val, X_te, A_tr, A_val, args, seed, device, dataset):
    if model_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-4, 4, 17))
        model.fit(np.vstack([X_tr, X_val]), np.vstack([A_tr, A_val]))
        return model.predict(X_te)
    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        model.fit(np.vstack([X_tr, X_val]), np.vstack([A_tr, A_val]))
        return model.predict(X_te)
    if model_name == "mlp":
        batch = min(args.batch_size, max(8, len(X_tr)))
        model = build_role_model(
            "acp_predictor",
            input_dim=X_tr.shape[1],
            output_dim=A_tr.shape[1],
            profile=args.model_profile,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)
        train_supervised(
            model,
            loader(X_tr, A_tr, batch_size=batch, shuffle=True, drop_last=len(X_tr) > 1),
            loader(X_val, A_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{dataset}/s{seed}/x_to_acp",
        )
        return predict(model, X_te, device, batch_size=args.batch_size)
    raise ValueError(f"unknown recoverability model: {model_name}")


def oracle_gain_table(results_path: Path):
    if not results_path.exists():
        return {}
    df = pd.read_csv(results_path)
    simple = df[df["model"] == "simple_mlp"].groupby("dataset")["mean_rmse"].mean()
    oracle = df[df["model"] == "oracle_mlp"].groupby("dataset")["mean_rmse"].mean()
    out = {}
    for dataset in sorted(set(simple.index) & set(oracle.index)):
        if simple[dataset] > 0:
            out[dataset] = float((simple[dataset] - oracle[dataset]) / simple[dataset] * 100.0)
    return out


def aggregate(seed_results: pd.DataFrame):
    rows = []
    for (dataset, model), g in seed_results.groupby(["dataset", "recoverability_model"]):
        rows.append(
            {
                "dataset": dataset,
                "recoverability_model": model,
                "macro_r2_mean": g["macro_r2"].mean(),
                "macro_r2_min": g["macro_r2"].min(),
                "macro_r2_max": g["macro_r2"].max(),
                "macro_r2_std": g["macro_r2"].std(),
                "macro_std_rmse_mean": g["macro_std_rmse"].mean(),
                "macro_std_rmse_min": g["macro_std_rmse"].min(),
                "macro_std_rmse_max": g["macro_std_rmse"].max(),
                "macro_std_rmse_std": g["macro_std_rmse"].std(),
                "macro_std_mae_mean": g["macro_std_mae"].mean(),
                "macro_std_mae_std": g["macro_std_mae"].std(),
                "oracle_gain_pct": g["oracle_gain_pct"].mean(),
                "n_seeds": g["seed"].nunique(),
                "n_rows": g["n_rows"].iloc[0],
                "n_train": g["n_train"].iloc[0],
                "n_test": g["n_test"].iloc[0],
                "n_features": g["n_features"].iloc[0],
                "n_intermediates": g["n_intermediates"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "recoverability_model"])


def write_markdown(agg, out_path: Path, primary_model: str):
    lines = [
        "# X-to-ACP Recoverability Diagnostic",
        "",
        "This diagnostic measures whether deployable inputs X contain enough information to reconstruct privileged ACP variables. It is separate from target prediction: models are trained as X -> ACP and evaluated on the ACP test set using the same dataset preprocessing and split policy as `run_experiments.py`.",
        "",
        f"- Primary recoverability model in the tables below: `{primary_model}`.",
        "- `macro R2`: mean ACP-descriptor R2 across ACP dimensions; higher is better.",
        "- `standardized RMSE`: mean ACP RMSE after train-only ACP standardization; lower is better.",
        "- `Oracle gain`: relative RMSE reduction of `oracle_mlp` over `simple_mlp` from the target-prediction experiment.",
        "- Case2 evidence: high Oracle gain plus low X->ACP macro R2 indicates that ACP is target-relevant but difficult to recover from feasible inputs.",
        "",
    ]
    for group, datasets in GROUPS.items():
        sub = agg[(agg["dataset"].isin(datasets)) & (agg["recoverability_model"] == primary_model)]
        if sub.empty:
            continue
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| Dataset | X->ACP macro R2 | X->ACP standardized RMSE | Oracle gain | Interpretation |")
        lines.append("|---|---:|---:|---:|---|")
        for dataset in datasets:
            row = sub[sub["dataset"] == dataset]
            if row.empty:
                continue
            r = row.iloc[0]
            if group == "Case2":
                interp = "High oracle gain with limited ACP recoverability; supports Case2."
            elif group == "Case1":
                interp = "Oracle gain is small; ACP adds limited target-relevant information."
            else:
                interp = "Main-group target result; recoverability is diagnostic, not the sole grouping rule."
            lines.append(
                f"| `{dataset}` | {r['macro_r2_mean']:.4f} +/- {r['macro_r2_std']:.4f} | "
                f"{r['macro_std_rmse_mean']:.4f} +/- {r['macro_std_rmse_std']:.4f} | "
                f"{r['oracle_gain_pct']:.1f}% | {interp} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_scatter(agg, out_path: Path, primary_model: str):
    group_by_dataset = {
        dataset: group
        for group, datasets in GROUPS.items()
        for dataset in datasets
    }
    colors = {"Main": "#2563eb", "Case1": "#16a34a", "Case2": "#dc2626"}
    markers = {"Main": "o", "Case1": "s", "Case2": "^"}
    sub = agg[agg["recoverability_model"] == primary_model].copy()
    sub["group"] = sub["dataset"].map(group_by_dataset)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for group in ["Main", "Case1", "Case2"]:
        g = sub[sub["group"] == group]
        if g.empty:
            continue
        ax.scatter(
            g["macro_r2_mean"],
            g["oracle_gain_pct"],
            s=58,
            c=colors[group],
            marker=markers[group],
            label=group,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in g.iterrows():
            label = row["dataset"].replace("_", " ")
            ax.annotate(label, (row["macro_r2_mean"], row["oracle_gain_pct"]), xytext=(4, 3), textcoords="offset points", fontsize=7)

    ax.axvline(0.30, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(30.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("X-to-ACP macro R2")
    ax.set_ylabel("Oracle gain over Simple MLP (%)")
    ax.set_title("ACP Recoverability vs. Oracle Gain")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=sum(GROUPS.values(), []), choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=["random_forest", "ridge"], choices=["mlp", "ridge", "random_forest"])
    parser.add_argument("--primary-model", default=None, choices=["mlp", "ridge", "random_forest"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", "--bottleneck-dim", dest="latent_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--model-profile", default="shared_encoder")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--target-results", type=Path, default=Path("../results/uci_external_experiments/processed_extra_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("../results/uci_external_experiments"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gains = oracle_gain_table(args.target_results)

    rows = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for seed in args.seeds:
            set_seed(seed)
            prepared = prepare_dataset(spec, seed, args)
            df, x_cols, acp_cols, y_cols, dropped, diag, X_tr, X_val, X_te, A_tr, A_val, A_te = prepared
            print(
                f"[RUN] dataset={dataset} seed={seed} X={len(x_cols)} ACP={len(acp_cols)} "
                f"train={len(X_tr)} test={len(X_te)} device={device}",
                flush=True,
            )
            for model_name in args.models:
                pred = fit_predict(model_name, X_tr, X_val, X_te, A_tr, A_val, args, seed, device, dataset)
                scores = score_acp(A_te, pred, acp_cols)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "recoverability_model": model_name,
                        "n_rows": len(df),
                        "n_train": len(X_tr),
                        "n_val": len(X_val),
                        "n_test": len(X_te),
                        "n_features": len(x_cols),
                        "n_intermediates": len(acp_cols),
                        "n_targets": len(y_cols),
                        "oracle_gain_pct": gains.get(dataset, np.nan),
                        "dropped_all_nan_cols": ",".join(dropped),
                        **diag,
                        **scores,
                    }
                )

    seed_results = pd.DataFrame(rows)
    agg = aggregate(seed_results)
    seed_path = args.output_dir / "x_to_acp_recoverability_results.csv"
    agg_path = args.output_dir / "x_to_acp_recoverability_aggregate.csv"
    md_path = args.output_dir / "x_to_acp_recoverability_diagnostic.md"
    fig_path = args.output_dir / "x_to_acp_recoverability_scatter.png"
    seed_results.to_csv(seed_path, index=False)
    agg.to_csv(agg_path, index=False)
    primary_model = args.primary_model or args.models[0]
    write_markdown(agg, md_path, primary_model)
    write_scatter(agg, fig_path, primary_model)

    print("\n[AGGREGATE]")
    print(
        agg[
            [
                "dataset",
                "recoverability_model",
                "macro_r2_mean",
                "macro_std_rmse_mean",
                "oracle_gain_pct",
                "n_seeds",
            ]
        ].to_string(index=False)
    )
    print(f"\n[DONE] wrote {seed_path}")
    print(f"[DONE] wrote {agg_path}")
    print(f"[DONE] wrote {md_path}")
    print(f"[DONE] wrote {fig_path}")


if __name__ == "__main__":
    main()
