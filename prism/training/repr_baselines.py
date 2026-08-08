"""
train_repr_baselines.py
========================
Train neural baseline models for the PRISM paper:
  1. simple_mlp      — Direct MLP: X -> Y  (feasible, no ACP)
  2. two_stage_mlp   — X -> ACP_hat, then [X, ACP_hat] -> Y (feasible)
  3. oracle_mlp      — [X, true ACP] -> Y  (infeasible upper bound; used as PRISM teacher)

oracle_mlp.pt is required by train_distilled_joint.py.

Outputs (saved to {output_dir}/seed_{seed}/):
  simple_mlp.pt          simple_mlp_summary.json
  two_stage_mlp_acp.pt    two_stage_mlp_summary.json
  two_stage_mlp_y.pt
  oracle_mlp.pt          oracle_mlp_summary.json

Usage:
  python train_repr_baselines.py --seed 1 --output-dir ../results/seed_1_baselines
  python train_repr_baselines.py --seed 1 --output-dir ../results/seed_1_baselines --epochs 50
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism import config
from prism.data_loader import get_data_loaders
from prism.encoder import build_role_model
from prism.models import import_model_modules
from prism.split_utils import split_dataset
from prism.utils import compute_metrics, save_json, set_seed


def train_mlp(model, train_loader, val_loader, epochs, patience, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mse, best_state, no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu())
                labels.append(yb)
        val_mse = F.mse_loss(torch.cat(preds), torch.cat(labels)).item()

        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d} | val_mse {val_mse:.6f} | patience {no_improve}")
        if no_improve >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, device, input_transform=None):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            xb = batch[0].to(device)
            yb = batch[-1]
            if input_transform:
                xb = input_transform(xb)
            preds.append(model(xb).cpu())
            labels.append(yb)
    y_pred = torch.cat(preds)
    y_true = torch.cat(labels)
    mse, mape, rmse, mae, r2 = compute_metrics(y_true, y_pred)
    return {
        "Mean_MSE":  mse.mean().item(),
        "Mean_RMSE": rmse.mean().item(),
        "Mean_MAE":  mae.mean().item(),
        "Mean_R2":   r2.mean().item(),
        "MSE":  {n: m.item() for n, m in zip(config.TARGET_COLS, mse)},
        "RMSE": {n: m.item() for n, m in zip(config.TARGET_COLS, rmse)},
        "MAE":  {n: m.item() for n, m in zip(config.TARGET_COLS, mae)},
        "R2":   {n: m.item() for n, m in zip(config.TARGET_COLS, r2)},
        "MAPE": {n: m.item() for n, m in zip(config.TARGET_COLS, mape)},
    }


def predict_tensor(model, x_tensor, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(x_tensor), batch_size=config.BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu())
    return torch.cat(preds, dim=0)


def metric_dict(y_true, y_pred):
    mse, mape, rmse, mae, r2 = compute_metrics(y_true, y_pred)
    return {
        "Mean_MSE": mse.mean().item(),
        "Mean_RMSE": rmse.mean().item(),
        "Mean_MAE": mae.mean().item(),
        "Mean_R2": r2.mean().item(),
        "MSE": {n: m.item() for n, m in zip(config.TARGET_COLS, mse)},
        "RMSE": {n: m.item() for n, m in zip(config.TARGET_COLS, rmse)},
        "MAE": {n: m.item() for n, m in zip(config.TARGET_COLS, mae)},
        "R2": {n: m.item() for n, m in zip(config.TARGET_COLS, r2)},
        "MAPE": {n: m.item() for n, m in zip(config.TARGET_COLS, mape)},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=1)
    p.add_argument("--output-dir", default="../results/seed_1_baselines")
    p.add_argument("--epochs",     type=int, default=config.MLP_EPOCHS)
    p.add_argument("--patience",   type=int, default=config.PATIENCE)
    p.add_argument("--model-profile", default="shared_encoder", help="Registered model family to use")
    p.add_argument("--model-module", nargs="*", default=[], help="Optional modules that register custom model families")
    p.add_argument("--bottleneck-dim", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    import_model_modules(args.model_module)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    X_tensor, _, y_cpu, df = get_data_loaders()
    acp_cpu = torch.tensor(df[config.ACP_COLS].values.astype("float32"))
    dataset   = TensorDataset(X_tensor, acp_cpu, y_cpu)
    train_ds, val_ds, test_ds, _ = split_dataset(dataset, seed=args.seed)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds,   False)
    test_loader  = make_loader(test_ds,  False)

    # Helper: XY loaders (strips ACP from batch)
    def xy_loader(ds, shuffle):
        indices = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
        xy_ds = TensorDataset(X_tensor[indices], y_cpu[indices])
        return DataLoader(xy_ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    xy_train = xy_loader(train_ds, True)
    xy_val   = xy_loader(val_ds,   False)

    # ── 1. Simple MLP: X → Y ────────────────────────────────────────────────
    print(f"\n[seed {args.seed}] Training Simple MLP (X → Y) ...")
    simple = build_role_model(
        "simple_mlp",
        input_dim=len(config.INPUT_COLS),
        output_dim=len(config.TARGET_COLS),
        profile=args.model_profile,
        latent_dim=args.bottleneck_dim,
    ).to(device)
    train_mlp(simple, xy_train, xy_val, args.epochs, args.patience, device)

    results_simple = evaluate(simple, test_loader, device,
                              input_transform=lambda x: x)
    results_simple.update({
        "model": "simple_mlp",
        "seed": args.seed,
        "model_profile": args.model_profile,
        "uses_true_acp_at_inference": False,
    })
    torch.save(simple.state_dict(), out_dir / "simple_mlp.pt")
    save_json(results_simple, out_dir / "simple_mlp_summary.json")
    print(f"  Simple MLP  Mean MSE: {results_simple['Mean_MSE']:.6f}")

    # ── 2. Two-stage MLP: X -> ACP_hat, [X, ACP_hat] -> Y ───────────────────
    print(f"\n[seed {args.seed}] Training Two-stage MLP (X -> ACP_hat -> Y) ...")

    def x_acp_loader(ds, shuffle):
        indices = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
        return DataLoader(
            TensorDataset(X_tensor[indices], acp_cpu[indices]),
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
        )

    acp_model = build_role_model(
        "acp_predictor",
        input_dim=len(config.INPUT_COLS),
        output_dim=len(config.ACP_COLS),
        profile=args.model_profile,
        latent_dim=args.bottleneck_dim,
    ).to(device)
    train_mlp(acp_model, x_acp_loader(train_ds, True), x_acp_loader(val_ds, False),
              args.epochs, args.patience, device)

    train_idx = train_ds.indices if hasattr(train_ds, "indices") else list(range(len(train_ds)))
    val_idx = val_ds.indices if hasattr(val_ds, "indices") else list(range(len(val_ds)))
    test_idx = test_ds.indices if hasattr(test_ds, "indices") else list(range(len(test_ds)))

    acp_hat_train = predict_tensor(acp_model, X_tensor[train_idx], device)
    acp_hat_val = predict_tensor(acp_model, X_tensor[val_idx], device)
    acp_hat_test = predict_tensor(acp_model, X_tensor[test_idx], device)

    two_stage_train = DataLoader(
        TensorDataset(torch.cat([X_tensor[train_idx], acp_hat_train], dim=1), y_cpu[train_idx]),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    two_stage_val = DataLoader(
        TensorDataset(torch.cat([X_tensor[val_idx], acp_hat_val], dim=1), y_cpu[val_idx]),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    two_stage_y = build_role_model(
        "two_stage_mlp",
        input_dim=len(config.INPUT_COLS) + len(config.ACP_COLS),
        output_dim=len(config.TARGET_COLS),
        profile=args.model_profile,
        latent_dim=args.bottleneck_dim,
    ).to(device)
    train_mlp(two_stage_y, two_stage_train, two_stage_val, args.epochs, args.patience, device)

    two_stage_y.eval()
    with torch.no_grad():
        two_stage_test_x = torch.cat([X_tensor[test_idx], acp_hat_test], dim=1).to(device)
        y_pred_two_stage = two_stage_y(two_stage_test_x).cpu()
    results_two_stage = metric_dict(y_cpu[test_idx], y_pred_two_stage)
    results_two_stage.update({
        "model": "two_stage_mlp",
        "seed": args.seed,
        "uses_true_acp_at_inference": False,
        "stage1": "X -> ACP_hat",
        "stage2": "[X, ACP_hat] -> Y",
        "model_profile": args.model_profile,
    })
    torch.save(acp_model.state_dict(), out_dir / "two_stage_mlp_acp.pt")
    torch.save(two_stage_y.state_dict(), out_dir / "two_stage_mlp_y.pt")
    save_json(results_two_stage, out_dir / "two_stage_mlp_summary.json")
    print(f"  Two-stage MLP  Mean MSE: {results_two_stage['Mean_MSE']:.6f}")

    # ── 3. Oracle MLP: [X, ACP] -> Y  (infeasible, teacher for PRISM) ────────
    print(f"\n[seed {args.seed}] Training Oracle MLP ([X, ACP] → Y) ...")

    def xacp_loader(ds, shuffle):
        indices = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
        xacp_ds = TensorDataset(
            torch.cat([X_tensor[indices], acp_cpu[indices]], dim=1),
            y_cpu[indices]
        )
        return DataLoader(xacp_ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    xacp_train = xacp_loader(train_ds, True)
    xacp_val   = xacp_loader(val_ds,   False)

    oracle = build_role_model(
        "oracle_mlp",
        input_dim=len(config.INPUT_COLS) + len(config.ACP_COLS),
        output_dim=len(config.TARGET_COLS),
        profile=args.model_profile,
        latent_dim=args.bottleneck_dim,
    ).to(device)
    train_mlp(oracle, xacp_train, xacp_val, args.epochs, args.patience, device)

    # Evaluate oracle on test set using true ACP (infeasible setting)
    def oracle_test_loader(ds):
        indices = ds.indices if hasattr(ds, "indices") else list(range(len(ds)))
        xacp_ds = TensorDataset(
            torch.cat([X_tensor[indices], acp_cpu[indices]], dim=1),
            y_cpu[indices]
        )
        return DataLoader(xacp_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    results_oracle = evaluate(oracle, oracle_test_loader(test_ds), device)
    results_oracle.update({
        "model": "oracle_mlp",
        "seed": args.seed,
        "model_profile": args.model_profile,
        "uses_true_acp_at_inference": True,
    })
    torch.save(oracle.state_dict(), out_dir / "oracle_mlp.pt")
    save_json(results_oracle, out_dir / "oracle_mlp_summary.json")
    print(f"  Oracle MLP  Mean MSE: {results_oracle['Mean_MSE']:.6f}")

    print(f"\n[DONE] Baselines saved to {out_dir}")


if __name__ == "__main__":
    main()
