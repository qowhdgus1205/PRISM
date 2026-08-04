# train_distilled_joint.py
#
# PRISM: Privileged Representation Integrated Synergistic Modeling
# Advanced Technical Approach:
# 1. Teacher-Student Distillation: Student latent 'h' mimics Oracle latent 'h_oracle'.
# 2. Physics Constraint: Monotonicity loss for [IT -> p_max] and [IT -> eta_i].
#
# Usage:
#   python train_distilled_joint.py --oracle-path runs/repr_baselines/seed_1/oracle_mlp.pt --seed 1

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism import config
from prism.data_loader import get_data_loaders
from prism.encoder import TargetMLP, build_role_model
from prism.models import import_model_modules
from prism.split_utils import split_dataset, split_metadata_from_subsets
from prism.utils import set_seed, save_json, compute_metrics, monotonicity_loss, perturbation_monotonicity_loss
from prism.losses import local_neighborhood_mixup
from prism.losses_physics import (
    wiebe_ordering_loss,
    wiebe_shape_loss,
    energy_consistency_loss,
    estimate_energy_coefficients,
)

TEACHER_PENULTIMATE_DIM = 64  # TargetMLP penultimate layer size

# Physics-based ACP head monotonicity pairs: (feat_idx, acp_idx, direction)
ACP_MONO_PAIRS = [
    (7, 0, -1),  # IT  → MFB10  (−)  r=−0.69
    (7, 1, -1),  # IT  → MFB50  (−)  r=−0.44
    (7, 2, -1),  # IT  → MFB90  (−)  r=−0.29
    (7, 3, +1),  # IT  → p_max  (+)  r=+0.13
    (2, 0, +1),  # EGR → MFB10  (+)  r=+0.23
    (2, 1, +1),  # EGR → MFB50  (+)  r=+0.34
    (2, 2, +1),  # EGR → MFB90  (+)  r=+0.35
    (2, 3, -1),  # EGR → p_max  (−)  r=−0.20
    (1, 3, +1),  # p_int→ p_max  (+)  r=+0.77
]

# Extended Y monotonicity pairs (applied from epoch 0 for v4)
EXT_MONO_PAIRS_Y = [
    (7, 0, +1),  # IT   → eta_i  (+)  r=+0.69
    (1, 3, +1),  # p_int → IMEP   (+)  r=+0.72
    (1, 0, +1),  # p_int → eta_i  (+)  r=+0.30
    (2, 0, -1),  # EGR  → eta_i   (−)  r=−0.31
    (2, 3, -1),  # EGR  → IMEP    (−)  r=−0.22
    (0, 2, +1),  # N    → T_exh   (+)  higher RPM → higher exhaust temp
]


class PhysicsFloor(nn.Module):
    """
    Methodology 3: Physics-Residual Floor.
    Provides a simple linear baseline based on domain knowledge.
    E.g., IMEP is strongly correlated with p_int.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Initialize as a simple linear layer
        # In a real scenario, this could be a fixed analytic formula
        self.floor = nn.Linear(input_dim, output_dim)
        # Weak initialization to let MLP take lead in IID, but keep floor stable
        nn.init.xavier_uniform_(self.floor.weight, gain=0.1)

    def forward(self, x):
        return self.floor(x)



class PRISMv4Model(nn.Module):
    """
    PRISM v4: Physics-Harmony training framework with additive correction.

    Ŷ = Ŷ_base + ΔŶ
      Ŷ_base = α·f_direct(X) + (1-α)·f_floor(X)           X-only, physics-stable
      ΔŶ     = σ(f_conf(X)) · f_corr(g(X)⊙h(X), ACP_hat)  ACP-informed, gated
               (use_gate=False → g(X)⊙h(X) replaced by h(X))
      α      = σ(learnable scalar), init → 0.5
    """
    def __init__(self, input_dim, bottleneck_dim, acp_dim, target_dim, dropout=0.1, use_gate=True, use_confidence_blend=False, teacher_repr_dim=TEACHER_PENULTIMATE_DIM):
        super().__init__()
        from prism.encoder import Encoder, ACPHead, ProjectionHead

        self.use_gate = use_gate
        self.use_confidence_blend = use_confidence_blend
        self.encoder = Encoder(input_dim=input_dim, latent_dim=bottleneck_dim, dropout=dropout)
        self.acp_head = ACPHead(latent_dim=bottleneck_dim, acp_dim=acp_dim)
        self.distill_proj = nn.Linear(bottleneck_dim, teacher_repr_dim)
        self.contrastive_proj = ProjectionHead(input_dim=bottleneck_dim, proj_dim=bottleneck_dim)

        # Physics gate: g(X) ∈ (0,1)^bottleneck_dim — ablatable via use_gate=False
        if use_gate:
            gate_hidden = 64
            self.gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, bottleneck_dim),
                nn.Sigmoid()
            )

        # Ŷ_base: X-only stable prediction
        self.physics_floor = PhysicsFloor(input_dim=input_dim, output_dim=target_dim)
        self.direct_head = TargetMLP(input_dim=bottleneck_dim, output_dim=target_dim, dropout=dropout)
        # A: learnable blend α = σ(blend_logit); init=0 → α=0.5
        self.blend_logit = nn.Parameter(torch.zeros(1))

        # C: correction confidence gate σ(f_conf(X)) ∈ (0,1) — suppresses ΔŶ on OOD inputs
        self.corr_gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.confidence_blend_gate = nn.Sequential(
            nn.Linear(input_dim + acp_dim, 32),
            nn.SiLU(),
            nn.Linear(32, target_dim),
            nn.Sigmoid(),
        )

        # B: ACP-informed correction f_corr([gh, ACP_hat]) → ΔŶ
        self.correction_head = nn.Sequential(
            nn.Linear(bottleneck_dim + acp_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, target_dim),
        )
        # Near-zero init: early training dominated by Ŷ_base
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def _base_and_correction(self, x, h, true_acp=None):
        acp_hat = self.acp_head(h)
        gh = self.gate(x) * h if self.use_gate else h

        alpha = torch.sigmoid(self.blend_logit)
        y_base = alpha * self.direct_head(h) + (1 - alpha) * self.physics_floor(x)

        # Oracle mode: use true ACP in correction head (infeasible at inference)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        raw_correction = self.correction_head(torch.cat([gh, acp_for_corr], dim=-1))
        if self.use_confidence_blend:
            # Target-wise confidence for the ACP-mediated path. Low confidence
            # falls back to the X-only base prediction.
            conf = self.confidence_blend_gate(torch.cat([x, acp_for_corr], dim=-1))
        else:
            conf = self.corr_gate(x)
        y_correction = raw_correction * conf
        return y_base, y_correction, acp_hat

    def forward(self, x, true_acp=None):
        h = self.encoder(x)
        z_contrastive = self.contrastive_proj(h)
        y_base, y_correction, acp_hat = self._base_and_correction(x, h, true_acp=true_acp)
        return y_base + y_correction, acp_hat, y_correction, z_contrastive

    def predict(self, x, true_acp=None, mc_dropout=False):
        if mc_dropout:
            self.train()
        else:
            self.eval()
        h = self.encoder(x)
        y_base, y_correction, _ = self._base_and_correction(x, h, true_acp=true_acp)
        return y_base + y_correction

class PRISMTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.target_cols = config.TARGET_COLS
        self.acp_cols = config.ACP_COLS

    def teacher_repr_dim(self):
        profile = getattr(self.args, "model_profile", "shared_encoder")
        if profile == "shared_encoder":
            return int(self.args.bottleneck_dim)
        return TEACHER_PENULTIMATE_DIM

    def load_teacher(self, oracle_path, input_dim):
        profile = getattr(self.args, "model_profile", "shared_encoder")
        print(f"[INFO] Loading Teacher (Oracle MLP, profile={profile}) from {oracle_path}")
        if profile == "shared_encoder":
            teacher = build_role_model(
                "oracle_mlp",
                input_dim=input_dim + len(self.acp_cols),
                output_dim=len(self.target_cols),
                profile=profile,
                latent_dim=int(self.args.bottleneck_dim),
            )
        else:
            teacher = TargetMLP(input_dim=input_dim + len(self.acp_cols), output_dim=len(self.target_cols))
        teacher.load_state_dict(torch.load(oracle_path, map_location=self.device, weights_only=True))
        teacher.to(self.device)
        teacher.eval()
        return teacher

    @staticmethod
    def get_teacher_repr(teacher, x_acp):
        if hasattr(teacher, "encode"):
            return teacher.encode(x_acp)
        return teacher.model[:-1](x_acp)


    def pretrain_acp_path(self, student, train_loader, val_loader):
        epochs = int(getattr(self.args, "pretrain_acp_epochs", 0))
        if epochs <= 0:
            return

        lr = float(getattr(self.args, "pretrain_acp_lr", config.MLP_LR))
        optimizer = torch.optim.Adam(
            list(student.encoder.parameters()) + list(student.acp_head.parameters()),
            lr=lr,
        )
        best_state = None
        best_val = float("inf")
        no_improve = 0
        patience = int(getattr(self.args, "pretrain_acp_patience", min(50, max(5, epochs // 4))))

        print(f"[PRETRAIN] ACP path X -> encoder -> ACP_hat for {epochs} epochs")
        for epoch in range(1, epochs + 1):
            student.train()
            losses = []
            for xb, acpb, _ in train_loader:
                xb = xb.to(self.device)
                acpb = acpb.to(self.device)
                optimizer.zero_grad()
                acp_hat = student.acp_head(student.encoder(xb))
                loss = F.mse_loss(acp_hat, acpb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            student.eval()
            vals = []
            with torch.no_grad():
                for xb, acpb, _ in val_loader:
                    xb = xb.to(self.device)
                    acpb = acpb.to(self.device)
                    vals.append(F.mse_loss(student.acp_head(student.encoder(xb)), acpb).item())
            val = float(np.mean(vals))
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"[PRETRAIN] epoch={epoch:03d} train_acp={np.mean(losses):.5f} val_acp={val:.5f} patience={no_improve}")
            if no_improve >= patience:
                print(f"[PRETRAIN] early stop at epoch {epoch}; best val_acp={best_val:.5f}")
                break

        if best_state is not None:
            student.load_state_dict(best_state)

    def train(self, ext_indices=None):
        """
        ext_indices: optional (train_idx, val_idx, test_idx) for OOD experiments.
                     If None, uses standard split_dataset with self.args.seed.
        """
        set_seed(self.args.seed)
        subdir_name = self.args.model_subdir if self.args.model_subdir else f"seed_{self.args.seed}_PRISM"
        out_dir = Path(self.args.output_dir) / subdir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Data Loading
        X_tensor, _, y_cpu, df = get_data_loaders()
        acp_cpu = torch.tensor(df[self.acp_cols].values.astype(np.float32))
        dataset = TensorDataset(X_tensor, acp_cpu, y_cpu)

        if ext_indices is not None:
            train_idx, val_idx, test_idx = ext_indices
            from torch.utils.data import Subset
            train_ds = Subset(dataset, train_idx)
            val_ds   = Subset(dataset, val_idx)
            test_ds  = Subset(dataset, test_idx)
        else:
            train_ds, val_ds, test_ds, _ = split_dataset(dataset, seed=self.args.seed)

        # Subsample training data if data-fraction < 1.0
        if self.args.data_fraction < 1.0:
            n_sub = int(len(train_ds) * self.args.data_fraction)
            perm = torch.randperm(len(train_ds))[:n_sub]
            train_ds = torch.utils.data.Subset(train_ds, perm)
            print(f"[INFO] Using {self.args.data_fraction*100:.1f}% of training data: {n_sub} samples")

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        # 2. Model Setup
        use_gate = getattr(self.args, "use_gate", True)
        student = PRISMv4Model(
            input_dim=X_tensor.shape[1],
            bottleneck_dim=self.args.bottleneck_dim,
            acp_dim=len(self.acp_cols),
            target_dim=len(self.target_cols),
            use_gate=use_gate,
            use_confidence_blend=getattr(self.args, "confidence_blend", False),
            teacher_repr_dim=self.teacher_repr_dim(),
        ).to(self.device)
        gate_tag = "with gate" if use_gate else "no gate"
        blend_tag = ", confidence blend" if getattr(self.args, "confidence_blend", False) else ""
        print(f"[INFO] Using PRISMv4Model (additive correction, {gate_tag}{blend_tag})")

        teacher = self.load_teacher(self.args.oracle_path, X_tensor.shape[1])

        self.pretrain_acp_path(student, train_loader, val_loader)

        optimizer = torch.optim.Adam(student.parameters(), lr=config.MLP_LR)

        best_val_mse = float("inf")
        best_state = None
        no_improve = 0

        curve_lambda = getattr(self.args, "curvature_lambda", 0.1)
        curve_start = getattr(self.args, "curve_start_epoch", 500)
        contrastive_lambda = getattr(self.args, "contrastive_lambda", 0.2)
        use_oracle_acp = getattr(self.args, "use_oracle_acp", False)
        train_oracle_acp = getattr(self.args, "train_oracle_acp", False)  # TF-PRISM: true ACP at train only

        wiebe_lambda  = float(getattr(self.args, "wiebe_lambda",  0.0))
        energy_lambda = float(getattr(self.args, "energy_lambda", 0.0))

        # Estimate energy constraint coefficients once from the full training set
        energy_alpha = energy_beta = energy_intercept = 0.0
        if energy_lambda > 0:
            X_tr_all = torch.cat([xb for xb, _, _ in train_loader], dim=0).cpu()
            Y_tr_all = torch.cat([yb for _, _, yb in train_loader], dim=0).cpu()
            energy_alpha, energy_beta, energy_intercept = estimate_energy_coefficients(
                X_tr_all, Y_tr_all)
            print(f"[PHYSICS] Energy coeff: α={energy_alpha:.4f} β={energy_beta:.4f} γ={energy_intercept:.4f}")

        print("[START] PRISMv4 Distillation & Physics Constrained Training...")
        for epoch in range(1, self.args.epochs + 1):
            student.train()
            train_losses = []

            for xb, acpb, yb in train_loader:
                xb, acpb, yb = xb.to(self.device), acpb.to(self.device), yb.to(self.device)

                # Forward
                oracle_acp_arg = acpb if (use_oracle_acp or train_oracle_acp) else None
                y_hat, acp_hat, y_correction, z_contrastive = student(xb, true_acp=oracle_acp_arg)
                h_student = student.encoder(xb)

                # Primary & Aux Loss
                loss_y = F.mse_loss(y_hat, yb)
                loss_acp = F.mse_loss(acp_hat, acpb)

                # A. Representation KD: student projection matches teacher penultimate
                with torch.no_grad():
                    x_acp_teacher = torch.cat([xb, acpb], dim=1)
                    h_teacher = self.get_teacher_repr(teacher, x_acp_teacher)
                loss_distill = F.mse_loss(student.distill_proj(h_student), h_teacher)

                # B. Monotonicity
                if self.args.mono_lambda > 0:
                    if getattr(self.args, "perturb_mono", False):
                        # Perturbation-based: ceteris-paribus +ε sweep (matches VR evaluation)
                        loss_mono = perturbation_monotonicity_loss(
                            lambda x: student.predict(x),
                            xb, y_hat, EXT_MONO_PAIRS_Y,
                        )
                    else:
                        loss_mono = (monotonicity_loss(xb, acp_hat, ACP_MONO_PAIRS)
                                     + monotonicity_loss(xb, y_hat, EXT_MONO_PAIRS_Y))
                else:
                    loss_mono = torch.tensor(0.0, device=self.device)

                # D. Curvature Loss (Second derivative penalty for smoothness)
                if curve_lambda > 0 and epoch >= curve_start:
                    from prism.utils import curvature_loss
                    loss_curve = curvature_loss(student, xb, target_idx=0, feat_idx=7)
                else:
                    loss_curve = torch.tensor(0.0, device=self.device)

                # E. Physics-Informed Contrastive (Supervised Physics Contrastive)
                if contrastive_lambda > 0:
                    from prism.utils import supervised_physics_contrastive_loss
                    loss_contrastive = supervised_physics_contrastive_loss(
                        z_contrastive, acpb, temperature=0.07, sigma=0.5
                    )
                else:
                    loss_contrastive = torch.tensor(0.0, device=self.device)
                mixup_lambda = getattr(self.args, "mixup_lambda", 0.0)
                if mixup_lambda > 0:
                    mixed = local_neighborhood_mixup(
                        xb, acpb, yb,
                        k=getattr(self.args, "mixup_k", 5),
                        alpha=getattr(self.args, "mixup_alpha", 0.4),
                    )
                    if mixed is not None:
                        x_mix, acp_mix, y_mix = mixed
                        oracle_acp_mix = acp_mix if getattr(self.args, "oracle_prism", False) else None
                        y_mix_hat, acp_mix_hat, _, _ = student(x_mix, true_acp=oracle_acp_mix)
                        loss_mixup = (F.mse_loss(y_mix_hat, y_mix)
                                      + self.args.acp_lambda * F.mse_loss(acp_mix_hat, acp_mix))
                    else:
                        loss_mixup = torch.tensor(0.0, device=self.device)
                else:
                    loss_mixup = torch.tensor(0.0, device=self.device)

                # F. Wiebe physics constraints (ordering + shape)
                if wiebe_lambda > 0:
                    loss_wiebe = (wiebe_ordering_loss(acp_hat)
                                  + wiebe_shape_loss(acp_hat))
                else:
                    loss_wiebe = torch.tensor(0.0, device=self.device)

                # G. Energy consistency (IMEP ~ α·η_i + β·FM + γ)
                if energy_lambda > 0:
                    loss_energy = energy_consistency_loss(
                        y_hat, xb,
                        energy_alpha, energy_beta, energy_intercept,
                    )
                else:
                    loss_energy = torch.tensor(0.0, device=self.device)

                loss = (loss_y
                        + self.args.acp_lambda     * loss_acp
                        + self.args.distill_lambda * loss_distill
                        + self.args.mono_lambda    * loss_mono
                        + curve_lambda             * loss_curve
                        + contrastive_lambda       * loss_contrastive
                        + mixup_lambda             * loss_mixup
                        + wiebe_lambda             * loss_wiebe
                        + energy_lambda            * loss_energy)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append({
                    "total":       loss.item(),
                    "y":           loss_y.item(),
                    "acp":         loss_acp.item(),
                    "distill":     loss_distill.item(),
                    "mono":        loss_mono.item() if isinstance(loss_mono, torch.Tensor) else loss_mono,
                    "contrastive": loss_contrastive.item(),
                    "mixup":       loss_mixup.item(),
                    "wiebe":       loss_wiebe.item(),
                    "energy":      loss_energy.item(),
                })

            # Validation
            student.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, acpb_val, yb in val_loader:
                    xb = xb.to(self.device)
                    if use_oracle_acp:
                        val_acp = acpb_val.to(self.device)
                        preds = student.predict(xb, true_acp=val_acp).cpu()
                    else:
                        preds = student.predict(xb).cpu()
                    all_preds.append(preds)
                    all_labels.append(yb)

            val_mse_each, _, _, _, _ = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))
            val_mse = val_mse_each.mean().item()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = copy.deepcopy(student.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                avg = {k: np.mean([b[k] for b in train_losses]) for k in train_losses[0]}
                # 가중 적용 후 실제 기여 크기
                weighted = {
                    "y":           avg["y"],
                    "acp":         self.args.acp_lambda     * avg["acp"],
                    "distill":     self.args.distill_lambda * avg["distill"],
                    "mono":        self.args.mono_lambda    * avg["mono"],
                    "contrastive": contrastive_lambda       * avg["contrastive"],
                    "mixup":       mixup_lambda             * avg["mixup"],
                    "wiebe":       wiebe_lambda             * avg["wiebe"],
                    "energy":      energy_lambda            * avg["energy"],
                }
                breakdown = " | ".join(f"{k}={v:.5f}" for k, v in weighted.items())
                print(f"Epoch {epoch:03d} | total={avg['total']:.5f} | {breakdown} | val_mse={val_mse:.5f} | patience={no_improve}")

            if no_improve >= config.PATIENCE:
                print("Early stopping.")
                break

        # Evaluation
        student.load_state_dict(best_state)
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, acpb_test, yb in test_loader:
                xb = xb.to(self.device)
                if use_oracle_acp:
                    test_acp = acpb_test.to(self.device)
                    preds = student.predict(xb, true_acp=test_acp).cpu()
                else:
                    preds = student.predict(xb).cpu()
                all_preds.append(preds)
                all_labels.append(yb)

        mse_each, mape_each, rmse_each, mae_each, r2_each = compute_metrics(
            torch.cat(all_labels), torch.cat(all_preds))

        results = {
            "Mean_MSE":  mse_each.mean().item(),
            "Mean_RMSE": rmse_each.mean().item(),
            "Mean_MAE":  mae_each.mean().item(),
            "Mean_R2":   r2_each.mean().item(),
            "MSE":  {n: m.item() for n, m in zip(self.target_cols, mse_each)},
            "RMSE": {n: m.item() for n, m in zip(self.target_cols, rmse_each)},
            "MAE":  {n: m.item() for n, m in zip(self.target_cols, mae_each)},
            "R2":   {n: m.item() for n, m in zip(self.target_cols, r2_each)},
            "MAPE": {n: m.item() for n, m in zip(self.target_cols, mape_each)},
            "model_profile": getattr(self.args, "model_profile", "shared_encoder"),
            "bottleneck_dim": int(self.args.bottleneck_dim),
        }

        print(f"\n[FINAL TEST RESULTS - PRISM]")
        print(f"Mean RMSE: {results['Mean_RMSE']:.6f}  MSE: {results['Mean_MSE']:.6f}  MAE: {results['Mean_MAE']:.6f}  R2: {results['Mean_R2']:.4f}")

        torch.save(student.state_dict(), out_dir / "PRISM_model.pt")
        save_json(results, out_dir / "summary.json")
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-path", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/advanced_joint")
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--acp-lambda", type=float, default=0.5)
    parser.add_argument("--distill-lambda", type=float, default=0.05, help="Repr-level KD weight")
    parser.add_argument("--mono_lambda", type=float, default=0.01,
                        help="ACP + Y 단조성 손실 가중치 (ACP 9쌍 + Y 5쌍 합산)")
    parser.add_argument("--perturb-mono", action="store_true", default=False,
                        help="Use perturbation-based monotonicity loss (ceteris-paribus ε-sweep, "
                             "directly matches Valid Rate evaluation)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--model-subdir", type=str, default=None)
    parser.add_argument("--use-gate", action="store_true", default=True)
    parser.add_argument("--no-gate", action="store_false", dest="use_gate")
    parser.add_argument("--confidence-blend", action="store_true", default=False,
                        help="Use target-wise confidence gate over the ACP correction path")
    parser.add_argument("--data-fraction", type=float, default=1.0, help="Use fraction of training data (0.0 to 1.0)")
    parser.add_argument("--curvature-lambda", type=float, default=0.1,
                        help="Lambda for curvature loss (second-derivative smoothness penalty)")
    parser.add_argument("--curve-start-epoch", type=int, default=500,
                        help="Epoch from which curvature loss is applied")
    parser.add_argument("--contrastive-lambda", type=float, default=0.2,
                        help="Lambda for supervised physics-contrastive loss on ACP embeddings")
    parser.add_argument("--mixup-lambda", type=float, default=0.0,
                        help="Lambda for local neighborhood mix-up on X/ACP/Y")
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                        help="Beta distribution alpha for local neighborhood mix-up")
    parser.add_argument("--mixup-k", type=int, default=5,
                        help="Number of nearest in-batch neighbors for local mix-up")
    parser.add_argument("--pretrain-acp-epochs", type=int, default=0,
                        help="Warm-start encoder+ACPHead on X -> ACP before joint PRISM training")
    parser.add_argument("--pretrain-acp-lr", type=float, default=config.MLP_LR,
                        help="Learning rate for ACP warm-start")
    parser.add_argument("--pretrain-acp-patience", type=int, default=20,
                        help="Early-stopping patience for ACP warm-start")
    parser.add_argument("--model-profile", default="shared_encoder",
                        help="Registered model family used by oracle teacher and comparable MLP baselines")
    parser.add_argument("--model-module", nargs="*", default=[],
                        help="Optional modules that register custom model families")
    parser.add_argument("--use-oracle-acp", action="store_true", default=False,
                        help="Feed true ACP into correction head at both train and test time (infeasible upper bound)")
    parser.add_argument("--train-oracle-acp", action="store_true", default=False,
                        help="TF-PRISM: feed true ACP into correction head during training only; use predicted ACP at inference")
    # Physics constraints
    parser.add_argument("--wiebe-lambda", type=float, default=0.0,
                        help="Weight for Wiebe physics losses (ordering + shape); 0 = disabled")
    parser.add_argument("--energy-lambda", type=float, default=0.0,
                        help="Weight for energy consistency loss (IMEP ~ alpha*eta_i + beta*FM); 0 = disabled")
    args = parser.parse_args()

    import_model_modules(args.model_module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = PRISMTrainer(args, device)
    trainer.train()
