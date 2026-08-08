"""
encoder_pretrain.py
===================
Stage 2: Physics-aware encoder pretraining with KD + Contrastive + ACP.

The core insight: joint training of KD/Contrastive with Y prediction causes gradient
conflict that hurts V01 → V08/V14 are *worse* than plain V01. Solution: pretrain the
encoder representation without Y supervision, then freeze it for Stage 3.

Architecture trained here:
  X(8) → Encoder(8→64→256→32) → h(32)
           ├── ACPHead(32→16→4)       → ACP_hat     [ACP loss]
           ├── DistillProj(32→64)     → h_distill   [KD loss]
           └── ContrastiveProj(32→32) → z           [SupCon loss]

  Loss = λ_acp * MSE(ACP_hat, ACP_true)
       + λ_kd  * MSE(h_distill, oracle_penultimate([X, ACP_true]))
       + λ_con * SupervisedPhysicsContrastive(z, ACP_true)

NO Y prediction loss — keeps Y gradient from polluting the physics representation.

Outputs saved to `output_dir/`:
  encoder.pt    — Encoder state dict
  acp_head.pt   — ACPHead state dict
  summary.json  — training metrics
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
from prism.models import Encoder, ACPHead, ProjectionHead
from prism.encoder import TargetMLP
from prism.split_utils import split_dataset
from prism.losses import supervised_physics_contrastive_loss
from prism.utils import set_seed, save_json

TEACHER_PENULTIMATE_DIM = 64  # oracle TargetMLP(12→256→128→64→4) penultimate = 64


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------

def load_oracle_teacher(path: Path, device: torch.device) -> TargetMLP:
    """Load Oracle MLP (X+ACP → Y); we use its penultimate layer as KD target."""
    model = TargetMLP(
        input_dim=len(config.INPUT_COLS) + len(config.ACP_COLS),
        output_dim=len(config.TARGET_COLS),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_oracle_penultimate(teacher: TargetMLP, x: torch.Tensor, acp: torch.Tensor) -> torch.Tensor:
    """Extract 64-dim penultimate activations from oracle MLP."""
    with torch.no_grad():
        x_acp = torch.cat([x, acp], dim=1)
        # TargetMLP: model.model = [Linear, LN, SiLU, Drop, Linear, LN, SiLU, Drop, Linear, LN, SiLU, Linear]
        # Penultimate = after third block (index :-1 = everything except last Linear)
        h = teacher.model[:-1](x_acp)
    return h  # (B, 64)


# ---------------------------------------------------------------------------
# Student model (encoder + auxiliary heads)
# ---------------------------------------------------------------------------

class EncoderWithHeads(nn.Module):
    """Encoder + ACP head + KD projection + Contrastive projection.
    Only used during Stage 2 pretraining. After training, encoder and acp_head
    are saved separately and used in Stage 3.
    """
    def __init__(self, input_dim: int, latent_dim: int, acp_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.acp_head = ACPHead(latent_dim=latent_dim, acp_dim=acp_dim)
        self.distill_proj = nn.Linear(latent_dim, TEACHER_PENULTIMATE_DIM)
        self.contrastive_proj = ProjectionHead(input_dim=latent_dim, proj_dim=latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        acp_hat = self.acp_head(h)
        h_distill = self.distill_proj(h)
        z = self.contrastive_proj(h)
        return h, acp_hat, h_distill, z


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class EncoderPretrainer:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device

    def train(self):
        set_seed(self.args.seed)
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Data ──────────────────────────────────────────────────────────
        X_tensor, _, y_cpu, df = get_data_loaders()
        acp_cpu = torch.tensor(df[config.ACP_COLS].values.astype(np.float32))
        dataset = TensorDataset(X_tensor, acp_cpu, y_cpu)
        train_ds, val_ds, _, _ = split_dataset(dataset, seed=self.args.seed)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)

        # ── Teacher ───────────────────────────────────────────────────────
        teacher = load_oracle_teacher(Path(self.args.oracle_path), self.device)
        print(f"[Stage2] Teacher loaded: {self.args.oracle_path}")

        # ── Student ───────────────────────────────────────────────────────
        student = EncoderWithHeads(
            input_dim=len(config.INPUT_COLS),
            latent_dim=self.args.latent_dim,
            acp_dim=len(config.ACP_COLS),
        ).to(self.device)

        optimizer = torch.optim.Adam(student.parameters(), lr=self.args.lr)

        lam_acp = self.args.acp_lambda
        lam_kd  = self.args.kd_lambda
        lam_con = self.args.contrastive_lambda

        print(f"[Stage2] λ_acp={lam_acp}  λ_kd={lam_kd}  λ_con={lam_con}")
        print(f"[Stage2] latent_dim={self.args.latent_dim}  epochs={self.args.epochs}")

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.args.epochs + 1):
            student.train()
            losses = []

            for xb, acpb, _ in train_loader:
                xb   = xb.to(self.device)
                acpb = acpb.to(self.device)

                _, acp_hat, h_distill, z = student(xb)

                # ACP regression
                loss_acp = F.mse_loss(acp_hat, acpb)

                # Knowledge Distillation (representation matching)
                h_oracle = get_oracle_penultimate(teacher, xb, acpb)
                loss_kd  = F.mse_loss(h_distill, h_oracle)

                # Supervised Physics Contrastive
                loss_con = supervised_physics_contrastive_loss(
                    z, acpb, temperature=0.07, sigma=0.5
                )

                loss = lam_acp * loss_acp + lam_kd * loss_kd + lam_con * loss_con

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append({
                    "total": loss.item(),
                    "acp":   loss_acp.item(),
                    "kd":    loss_kd.item(),
                    "con":   loss_con.item(),
                })

            # ── Validation: monitor ACP MSE as proxy for representation quality ──
            student.eval()
            val_acp_losses = []
            with torch.no_grad():
                for xb, acpb, _ in val_loader:
                    xb = xb.to(self.device); acpb = acpb.to(self.device)
                    _, acp_hat, _, _ = student(xb)
                    val_acp_losses.append(F.mse_loss(acp_hat, acpb).item())
            val_acp = float(np.mean(val_acp_losses))

            if val_acp < best_val:
                best_val = val_acp
                best_state = copy.deepcopy(student.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 50 == 0:
                avg = {k: np.mean([b[k] for b in losses]) for k in losses[0]}
                print(f"  Epoch {epoch:04d} | "
                      f"total={avg['total']:.5f} "
                      f"acp={lam_acp*avg['acp']:.5f} "
                      f"kd={lam_kd*avg['kd']:.5f} "
                      f"con={lam_con*avg['con']:.5f} | "
                      f"val_acp={val_acp:.5f} patience={no_improve}")

            if no_improve >= self.args.patience:
                print(f"[Stage2] Early stop at epoch {epoch}")
                break

        # ── Save ──────────────────────────────────────────────────────────
        student.load_state_dict(best_state)
        student.eval()

        # Validate final ACP R² on val set
        all_acp_hat, all_acp_true = [], []
        with torch.no_grad():
            for xb, acpb, _ in val_loader:
                xb = xb.to(self.device); acpb = acpb.to(self.device)
                _, acp_hat, _, _ = student(xb)
                all_acp_hat.append(acp_hat.cpu())
                all_acp_true.append(acpb.cpu())
        acp_pred = torch.cat(all_acp_hat)
        acp_true = torch.cat(all_acp_true)
        r2s = []
        for i in range(acp_pred.shape[1]):
            ss_res = ((acp_pred[:, i] - acp_true[:, i]) ** 2).sum().item()
            ss_tot = ((acp_true[:, i] - acp_true[:, i].mean()) ** 2).sum().item()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        acp_r2 = float(np.nanmean(r2s))

        torch.save(student.encoder.state_dict(),   out_dir / "encoder.pt")
        torch.save(student.acp_head.state_dict(),  out_dir / "acp_head.pt")
        summary = {
            "seed":          self.args.seed,
            "latent_dim":    self.args.latent_dim,
            "acp_lambda":    lam_acp,
            "kd_lambda":     lam_kd,
            "contrastive_lambda": lam_con,
            "best_val_acp_mse": best_val,
            "val_acp_r2_mean":  acp_r2,
            "val_acp_r2_per_dim": {
                n: r2s[i] for i, n in enumerate(config.ACP_COLS)
            },
        }
        save_json(summary, out_dir / "summary.json")
        print(f"[Stage2] Done. val_acp_r2={acp_r2:.4f}  saved → {out_dir}")
        return student.encoder, student.acp_head, summary


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Encoder pretraining (KD+Contrastive+ACP)")
    p.add_argument("--oracle-path",      required=True)
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--output-dir",       default="../results/ablation/seed_1/EncoderPretrain")
    p.add_argument("--latent-dim",       type=int,   default=32)
    p.add_argument("--acp-lambda",       type=float, default=0.5)
    p.add_argument("--kd-lambda",        type=float, default=0.05)
    p.add_argument("--contrastive-lambda", type=float, default=0.2)
    p.add_argument("--lr",               type=float, default=5e-4)
    p.add_argument("--epochs",           type=int,   default=500)
    p.add_argument("--patience",         type=int,   default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = EncoderPretrainer(args, device)
    trainer.train()
