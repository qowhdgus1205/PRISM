"""
one_stage.py
============
One-Stage PRISM with Stop-Gradient Representation Learning.

Motivation:
  V01 (plain MSE) RMSE=0.0346.
  V08 (joint KD+ACP+Con, no stop-grad) RMSE=0.0390 — WORSE due to gradient conflict.
  Key question: can one-stage + physics repr. learning beat V01?

Solution — Stop-Gradient (SG):
  Same architecture as V01: Encoder(X→h) + TargetMLP([X,h]→Y)
  Add auxiliary heads: ACPHead(h→ACP), KD_proj(h→h_kd), Con_proj(h→z)
  BUT: Y loss uses h.detach() — encoder ONLY receives gradients from ACP/KD/Con.

Result: Encoder learns physics-aligned representations from aux. losses,
        TargetMLP learns Y from those representations.
        No gradient conflict because Y loss never reaches encoder.

Ablation variants:
  OS_sg_acp   : stop-grad + ACP only
  OS_sg_kd    : stop-grad + ACP + KD
  OS_sg_full  : stop-grad + ACP + KD + Contrastive  ← main proposed method

Baselines (existing results reused):
  V01_Plain_MLP : no repr. learning                          RMSE≈0.035
  V08_PRISM_v3  : joint (no stop-grad) + ACP + KD + Mono    RMSE≈0.039 (worse!)
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism import config
from prism.data_loader import get_data_loaders
from prism.models import Encoder, ACPHead, ProjectionHead, TargetMLP
from prism.split_utils import split_dataset
from prism.losses import supervised_physics_contrastive_loss
from prism.utils import set_seed, compute_metrics, save_json
from prism.training.encoder_pretrain import load_oracle_teacher, get_oracle_penultimate

TEACHER_PENULTIMATE_DIM = 64


# ---------------------------------------------------------------------------
# One-Stage model
# ---------------------------------------------------------------------------

class OneStagePRISM(nn.Module):
    """One-stage PRISM with optional stop-gradient repr. learning.

    Architecture (same as V01 / PRISMv3):
      Encoder: X(8) → 64 → 256 → h(32)
      TargetMLP: [X(8), h(32)] → 256 → 128 → 64 → Y(4)

    Additions:
      ACPHead(h → ACP_hat) — physics supervision for encoder
      KD_proj(h → h_kd)   — knowledge distillation from oracle
      Con_proj(h → z)      — supervised physics contrastive

    Stop-gradient mode (use_stop_grad=True):
      Y loss uses h.detach() → encoder only updated by ACP/KD/Con
      No gradient conflict!

    Joint mode (use_stop_grad=False):
      Y loss flows through h → same as V08 (gradient conflict)
    """

    def __init__(
        self,
        input_dim: int,
        acp_dim: int,
        target_dim: int,
        latent_dim: int = 32,
        use_stop_grad: bool = True,
        use_kd: bool = False,
        use_con: bool = False,
    ):
        super().__init__()
        self.use_stop_grad = use_stop_grad
        self.use_kd  = use_kd
        self.use_con = use_con

        self.encoder     = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.target_head = TargetMLP(input_dim=input_dim + latent_dim, output_dim=target_dim)
        self.acp_head    = ACPHead(latent_dim=latent_dim, acp_dim=acp_dim)

        if use_kd:
            self.kd_proj = nn.Linear(latent_dim, TEACHER_PENULTIMATE_DIM)
        if use_con:
            self.con_proj = ProjectionHead(input_dim=latent_dim, proj_dim=latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        h_for_y = h.detach() if self.use_stop_grad else h
        y_pred  = self.target_head(torch.cat([x, h_for_y], dim=-1))
        acp_hat = self.acp_head(h)
        h_kd = self.kd_proj(h) if self.use_kd  else None
        z    = self.con_proj(h) if self.use_con else None
        return y_pred, acp_hat, h_kd, z

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = self.encoder(x)
            return self.target_head(torch.cat([x, h], dim=-1))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OneStagePRISMTrainer:
    def __init__(
        self,
        seed: int,
        out_dir: Path,
        variant: str,
        use_stop_grad: bool = True,
        use_kd: bool = False,
        use_con: bool = False,
        acp_lambda: float = 0.5,
        kd_lambda: float = 0.05,
        con_lambda: float = 0.2,
        latent_dim: int = 32,
        epochs: int = 1000,
        patience: int = 500,
        oracle_path: Path | None = None,
        device: torch.device | None = None,
    ):
        self.seed         = seed
        self.out_dir      = out_dir
        self.variant      = variant
        self.use_stop_grad = use_stop_grad
        self.use_kd       = use_kd
        self.use_con      = use_con
        self.acp_lambda   = acp_lambda
        self.kd_lambda    = kd_lambda
        self.con_lambda   = con_lambda
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.patience     = patience
        self.oracle_path  = oracle_path
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> dict:
        result_file = self.out_dir / f"ablation/seed_{self.seed}/{self.variant}/summary.json"
        if result_file.exists():
            print(f"  [SKIP] {self.variant} seed={self.seed}")
            with open(result_file) as f:
                return json.load(f)

        set_seed(self.seed)
        device = self.device

        # Data
        X_tensor, _, y_cpu, df = get_data_loaders()
        acp_tensor = torch.tensor(df[config.ACP_COLS].values.astype("float32"))
        dataset = TensorDataset(X_tensor, acp_tensor, y_cpu)
        train_ds, val_ds, test_ds, _ = split_dataset(dataset, seed=self.seed)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

        # Oracle teacher for KD
        teacher = None
        if self.use_kd and self.oracle_path is not None:
            teacher = load_oracle_teacher(self.oracle_path, device)
            print(f"[{self.variant}] Teacher loaded: {self.oracle_path}")

        # Model
        model = OneStagePRISM(
            input_dim=len(config.INPUT_COLS),
            acp_dim=len(config.ACP_COLS),
            target_dim=len(config.TARGET_COLS),
            latent_dim=self.latent_dim,
            use_stop_grad=self.use_stop_grad,
            use_kd=self.use_kd,
            use_con=self.use_con,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR)

        sg_tag  = "SG" if self.use_stop_grad else "joint"
        kd_tag  = "+KD" if self.use_kd  else ""
        con_tag = "+Con" if self.use_con else ""
        print(f"\n[{self.variant}] seed={self.seed}  {sg_tag}+ACP{kd_tag}{con_tag}")

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            for xb, acpb, yb in train_loader:
                xb, acpb, yb = xb.to(device), acpb.to(device), yb.to(device)
                optimizer.zero_grad()

                y_pred, acp_hat, h_kd, z = model(xb)

                loss_y   = F.mse_loss(y_pred, yb)
                loss_acp = F.mse_loss(acp_hat, acpb)

                loss_kd = torch.tensor(0.0, device=device)
                if h_kd is not None and teacher is not None:
                    h_oracle = get_oracle_penultimate(teacher, xb, acpb)
                    loss_kd  = F.mse_loss(h_kd, h_oracle)

                loss_con = torch.tensor(0.0, device=device)
                if z is not None:
                    loss_con = supervised_physics_contrastive_loss(
                        z, acpb, temperature=0.07, sigma=0.5
                    )

                loss = (loss_y
                        + self.acp_lambda * loss_acp
                        + self.kd_lambda  * loss_kd
                        + self.con_lambda * loss_con)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for xb, _, yb in val_loader:
                    preds.append(model.predict(xb.to(device)).cpu())
                    labels.append(yb)
            mse_each, *_ = compute_metrics(torch.cat(labels), torch.cat(preds))
            val_mse = mse_each.mean().item()

            if val_mse < best_val:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 100 == 0:
                print(f"  Epoch {epoch:04d} | val_mse={val_mse:.6f} patience={no_improve}")

            if no_improve >= self.patience:
                print(f"  [Early stop] epoch {epoch}")
                break

        # Test evaluation
        model.load_state_dict(best_state)
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, _, yb in test_loader:
                preds.append(model.predict(xb.to(device)).cpu())
                labels.append(yb)
        mse, mape, rmse, mae, r2 = compute_metrics(torch.cat(labels), torch.cat(preds))

        results = {
            "variant":   self.variant,
            "seed":      self.seed,
            "Mean_MSE":  mse.mean().item(),
            "Mean_RMSE": rmse.mean().item(),
            "Mean_MAE":  mae.mean().item(),
            "Mean_R2":   r2.mean().item(),
            "MSE":  {n: mse[i].item()  for i, n in enumerate(config.TARGET_COLS)},
            "RMSE": {n: rmse[i].item() for i, n in enumerate(config.TARGET_COLS)},
            "MAE":  {n: mae[i].item()  for i, n in enumerate(config.TARGET_COLS)},
            "R2":   {n: r2[i].item()   for i, n in enumerate(config.TARGET_COLS)},
        }
        result_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, result_file)
        torch.save(model.state_dict(), result_file.parent / "model.pt")
        print(f"  [DONE] RMSE={results['Mean_RMSE']:.5f}  R2={results['Mean_R2']:.4f}")
        return results
