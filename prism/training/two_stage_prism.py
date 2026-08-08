"""
train_two_stage_prism.py
========================
Two-Stage PRISM: trains a frozen base with an ACP-conditioned residual correction.

Stage 1 (frozen): Load pre-trained V01_Plain_MLP → Y_base = f_base(X)
Stage 2 (trainable):
  ACPHead:      X(8) → 64 → 64 → ACP_hat(4)
  CorrEncoder:  X(8) → 64 → 64 → h_corr(32)
  CorrHead:     [h_corr(32), ACP(4)] → 64 → 64 → ΔY(4)

Training: use ACP_true in CorrHead (teacher forcing on frozen base residuals)
  Loss = MSE(Y_base + ΔY, Y) + 0.5 * MSE(ACP_hat, ACP_true)

Inference: ACP_hat = ACPHead(X)
           Y_pred  = Y_base + CorrHead([CorrEncoder(X), ACP_hat])

Usage:
  python train_two_stage_prism.py --base-ckpt ../results/ablation/seed_1/V01_Plain_MLP/PRISM_model.pt \\
      --seed 1 --output-dir ../results/ablation/seed_1 --model-subdir TwoStage_PRISM
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from prism.training.distilled_joint import EXT_MONO_PAIRS_Y, ACP_MONO_PAIRS
from prism.utils import perturbation_monotonicity_loss
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from prism import config
from prism.data_loader import get_data_loaders
from prism.split_utils import split_dataset
from prism.utils import set_seed, compute_metrics, save_json


# ---------------------------------------------------------------------------
# Stage-2 architecture
# ---------------------------------------------------------------------------

class ACPHead(nn.Module):
    def __init__(self, input_dim, acp_dim, hidden=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, acp_dim),
        )
    def forward(self, x): return self.head(x)


class CorrEncoder(nn.Module):
    def __init__(self, input_dim, out_dim=32, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.enc(x)


class CorrHead(nn.Module):
    def __init__(self, in_dim, target_dim, hidden=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
    def forward(self, h, acp): return self.head(torch.cat([h, acp], dim=-1))


class TwoStageModel(nn.Module):
    def __init__(self, input_dim, acp_dim, target_dim, enc_dim=32):
        super().__init__()
        self.acp_head    = ACPHead(input_dim, acp_dim)
        self.corr_encoder = CorrEncoder(input_dim, enc_dim)
        self.corr_head   = CorrHead(enc_dim + acp_dim, target_dim)

    def forward(self, x, true_acp=None):
        acp_hat = self.acp_head(x)
        h = self.corr_encoder(x)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        delta_y = self.corr_head(h, acp_for_corr)
        return acp_hat, delta_y

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            acp_hat, delta_y = self.forward(x, true_acp=None)
        return delta_y


class TwoStageACPOnlyModel(nn.Module):
    """ACP-only residual correction: frozen base + [X, ACP_hat] -> Delta Y.

    Key difference from TwoStageModel:
      - NO CorrEncoder(X) — eliminates redundancy between h_corr and ACP_hat
      - CorrHead takes [X, ACP_hat] directly: X provides context, ACP provides physics
      - ACP_hat is the ONLY structured ACP pathway into the correction

    Expected benefit: forces CorrHead to use ACP signal (not bypass via h_corr).
    """
    def __init__(self, input_dim, acp_dim, target_dim, hidden=64):
        super().__init__()
        self.acp_head = ACPHead(input_dim, acp_dim)
        self.corr_head = nn.Sequential(
            nn.Linear(input_dim + acp_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        nn.init.zeros_(self.corr_head[-1].weight)
        nn.init.zeros_(self.corr_head[-1].bias)

    def forward(self, x, true_acp=None):
        acp_hat = self.acp_head(x)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        delta_y = self.corr_head(torch.cat([x, acp_for_corr], dim=-1))
        return acp_hat, delta_y

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            acp_hat, delta_y = self.forward(x, true_acp=None)
        return delta_y


class MonotoneCorrHead(nn.Module):
    """Direction A: structurally monotone correction from ACP_hat → ΔY.

    ACP dims: [MFB10(0), MFB50(1), MFB90(2), p_max(3)]
    Y dims:   [η_i(0), vol_eff(1), T_exh(2), IMEP(3)]

    Structural guarantees via abs() weights:
      p_max  ↑ → Δη_i  ↑   (more pressure = higher efficiency)
      MFB50  ↑ → Δη_i  ↓   (later combustion = lower efficiency; negated input)
      p_max  ↑ → ΔIMEP ↑   (more pressure = more work output)
      MFB50  ↑ → ΔT_exh↑   (later combustion = hotter exhaust)

    Free path handles all remaining ACP → ΔY relationships.
    """
    def __init__(self, acp_dim: int, target_dim: int, hidden: int = 64):
        super().__init__()
        mono_h = hidden // 4

        # Free path: unconstrained ACP → ΔY
        self.free = nn.Sequential(
            nn.Linear(acp_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        # Monotone η_i path: [p_max, -MFB50] → Δη_i_mono  (both increase → Δη_i ↑)
        self.mono_etai_h   = nn.Linear(2,  mono_h, bias=True)
        self.mono_etai_out = nn.Linear(mono_h, 1,  bias=True)
        # Monotone IMEP path: [p_max] → ΔIMEP_mono
        self.mono_imep_h   = nn.Linear(1,  mono_h, bias=True)
        self.mono_imep_out = nn.Linear(mono_h, 1,  bias=True)
        # Monotone T_exh path: [MFB50] → ΔT_exh_mono  (later combustion → hotter)
        self.mono_texh_h   = nn.Linear(1,  mono_h, bias=True)
        self.mono_texh_out = nn.Linear(mono_h, 1,  bias=True)

        for m in [self.free[-1], self.mono_etai_out, self.mono_imep_out, self.mono_texh_out]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, acp: torch.Tensor) -> torch.Tensor:
        delta_y = self.free(acp)

        # η_i (index 0): p_max↑→↑, MFB50↑→↓  (negate MFB50 before abs path)
        inp_etai = torch.stack([acp[:, 3], -acp[:, 1]], dim=1)
        h = F.relu(F.linear(inp_etai, self.mono_etai_h.weight.abs(), self.mono_etai_h.bias))
        delta_y = torch.cat([
            delta_y[:, :1] + F.linear(h, self.mono_etai_out.weight.abs(), self.mono_etai_out.bias),
            delta_y[:, 1:],
        ], dim=1)

        # IMEP (index 3): p_max↑→↑
        inp_imep = acp[:, 3:4]
        h = F.relu(F.linear(inp_imep, self.mono_imep_h.weight.abs(), self.mono_imep_h.bias))
        delta_y = torch.cat([
            delta_y[:, :3],
            delta_y[:, 3:4] + F.linear(h, self.mono_imep_out.weight.abs(), self.mono_imep_out.bias),
        ], dim=1)

        # T_exh (index 2): MFB50↑→↑
        inp_texh = acp[:, 1:2]
        h = F.relu(F.linear(inp_texh, self.mono_texh_h.weight.abs(), self.mono_texh_h.bias))
        delta_y = torch.cat([
            delta_y[:, :2],
            delta_y[:, 2:3] + F.linear(h, self.mono_texh_out.weight.abs(), self.mono_texh_out.bias),
            delta_y[:, 3:],
        ], dim=1)

        return delta_y


class TwoStagePhysicsModel(nn.Module):
    """Direction A+B: physical causal chain X → ACP_hat → ΔY.

    Direction B: CorrHead receives ONLY ACP_hat (X removed from correction path).
    Direction A: CorrHead has structural monotone guarantees via abs() weights.

    Causal chain:
      X → ACPHead → ACP_hat → MonotoneCorrHead → ΔY
    No X bypass into CorrHead — X's effect on Y must be mediated through ACP.
    """
    def __init__(self, input_dim: int, acp_dim: int, target_dim: int, hidden: int = 64):
        super().__init__()
        self.acp_head  = ACPHead(input_dim, acp_dim, hidden)
        self.corr_head = MonotoneCorrHead(acp_dim, target_dim, hidden)

    def forward(self, x: torch.Tensor, true_acp: torch.Tensor = None):
        acp_hat = self.acp_head(x)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        delta_y = self.corr_head(acp_for_corr)
        return acp_hat, delta_y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, delta_y = self.forward(x)
        return delta_y


class TwoStagePhysicsEncoderModel(nn.Module):
    """Stage 3 of 3-stage PRISM: uses a pre-trained, frozen physics encoder.

    Difference from TwoStageACPOnlyModel:
      - TwoStageACPOnly: CorrHead([X(8), ACP_hat(4)])  — raw X as context
      - TwoStagePhysicsEncoder: CorrHead([h(32), ACP_hat(4)])  — physics h as context

    h is produced by an encoder pre-trained in Stage 2 with KD + Contrastive + ACP.
    The frozen h encodes X into a physics-aligned latent space rather than raw features.
    ACP_hat is produced by an ACPHead that also comes from Stage 2 (via h, not directly X).

    Both encoder and acp_head are FROZEN during Stage 3; only corr_head is trainable.
    """

    def __init__(
        self,
        encoder: nn.Module,
        acp_head: nn.Module,
        target_dim: int,
        enc_dim: int = 32,
        acp_dim: int = 4,
        hidden: int = 64,
    ):
        super().__init__()
        # Frozen: set requires_grad=False so optimizer never touches them
        self.encoder  = encoder
        self.acp_head = acp_head
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.acp_head.parameters():
            p.requires_grad = False

        # Trainable correction head
        self.corr_head = nn.Sequential(
            nn.Linear(enc_dim + acp_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),            nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        nn.init.zeros_(self.corr_head[-1].weight)
        nn.init.zeros_(self.corr_head[-1].bias)

    def _encode(self, x: torch.Tensor):
        with torch.no_grad():
            h       = self.encoder(x)
            acp_hat = self.acp_head(h)
        return h, acp_hat

    def forward(self, x: torch.Tensor, true_acp: torch.Tensor = None):
        h, acp_hat = self._encode(x)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        delta_y = self.corr_head(torch.cat([h, acp_for_corr], dim=-1))
        return acp_hat, delta_y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h, acp_hat = self._encode(x)
            return self.corr_head(torch.cat([h, acp_hat], dim=-1))


class AlphaNet(nn.Module):
    """Input-conditioned blending gate: X → α ∈ (0,1)^target_dim.

    Learns per-sample, per-target blending weight between Y_base and
    Y_base + delta_Y. Initialized to output ~1 (start with full correction).
    """
    def __init__(self, input_dim: int, target_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        # Bias init: sigmoid(4) ≈ 0.98 → start near full correction
        nn.init.constant_(self.net[-1].bias, 4.0)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class TwoStageAdaptiveModel(nn.Module):
    """TwoStageACPOnly + adaptive blending gate.

    Y_final = Y_base + α(X) ⊙ delta_Y
    where alpha ∈ (0,1)^target_dim is learned per-sample, per-target.

    α is conditioned on X only (not ACP_hat) so training is inference-consistent.
    delta_Y = CorrHead([X, ACP_for_corr])  (teacher-forced during training).

    When α → 1: same as TwoStageACPOnly.
    When α → 0: falls back to Y_base (V01 is trusted for this sample).
    """
    def __init__(self, input_dim: int, acp_dim: int, target_dim: int,
                 hidden: int = 64, alpha_hidden: int = 32):
        super().__init__()
        self.acp_head  = ACPHead(input_dim, acp_dim)
        self.corr_head = nn.Sequential(
            nn.Linear(input_dim + acp_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),              nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )
        nn.init.zeros_(self.corr_head[-1].weight)
        nn.init.zeros_(self.corr_head[-1].bias)
        self.alpha_net = AlphaNet(input_dim, target_dim, alpha_hidden)

    def forward(self, x: torch.Tensor, true_acp: torch.Tensor = None):
        acp_hat = self.acp_head(x)
        acp_for_corr = true_acp if true_acp is not None else acp_hat
        delta_y = self.corr_head(torch.cat([x, acp_for_corr], dim=-1))
        alpha   = self.alpha_net(x)
        return acp_hat, alpha * delta_y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, delta_y_scaled = self.forward(x)
        return delta_y_scaled

    def get_alpha(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.alpha_net(x)


class TwoStageNoACPModel(nn.Module):
    """Ablation: two-stage residual correction WITHOUT ACP supervision.
    CorrHead takes only h_corr (no ACP input). No ACPHead, no ACP loss.
    Isolates whether ACP supervision (not just two-stage structure) drives improvement.
    """
    def __init__(self, input_dim, target_dim, enc_dim=32):
        super().__init__()
        self.corr_encoder = CorrEncoder(input_dim, enc_dim)
        self.corr_head = nn.Sequential(
            nn.Linear(enc_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
            nn.Linear(64, target_dim),
        )
        nn.init.zeros_(self.corr_head[-1].weight)
        nn.init.zeros_(self.corr_head[-1].bias)

    def forward(self, x):
        h = self.corr_encoder(x)
        return self.corr_head(h)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ---------------------------------------------------------------------------
# Load frozen base model — auto-detect v3 vs v4
# ---------------------------------------------------------------------------

class _PRISMv3Wrapper(nn.Module):
    """Minimal wrapper to load a PRISMv3 (PRISMModel, no-gate variant) checkpoint
    and expose a clean forward(x) → y interface for use as a frozen Stage-1 base.

    PRISMv3 keys: encoder.*, acp_head.*, distill_proj.*, contrastive_proj.*, target_head.*
    Predict path (use_gate=False): h = encoder(X);  Y = target_head(cat[X, h])
    """
    def __init__(self, input_dim, bottleneck_dim, acp_dim, target_dim):
        super().__init__()
        from prism.models import Encoder, ACPHead, ProjectionHead, TargetMLP
        self.encoder          = Encoder(input_dim=input_dim, latent_dim=bottleneck_dim)
        self.acp_head         = ACPHead(latent_dim=bottleneck_dim, acp_dim=acp_dim)
        self.distill_proj     = nn.Linear(bottleneck_dim, 64)
        self.contrastive_proj = ProjectionHead(input_dim=bottleneck_dim, proj_dim=bottleneck_dim)
        self.target_head      = TargetMLP(input_dim=input_dim + bottleneck_dim, output_dim=target_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.target_head(torch.cat([x, h], dim=-1))


def load_base_model(ckpt_path: Path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    # PRISMv4: has learnable alpha blend
    if "blend_logit" in sd:
        from prism.training.distilled_joint import PRISMv4Model
        m = PRISMv4Model(
            input_dim=len(config.INPUT_COLS), bottleneck_dim=32,
            acp_dim=len(config.ACP_COLS), target_dim=len(config.TARGET_COLS),
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        print("[Stage1] Loaded frozen base: PRISMv4")
        return m

    # PRISMv3 (V01_Plain_MLP / V08_PRISM_v3): has target_head + encoder, no gate
    if "target_head.model.0.weight" in sd:
        m = _PRISMv3Wrapper(
            input_dim=len(config.INPUT_COLS), bottleneck_dim=32,
            acp_dim=len(config.ACP_COLS),    target_dim=len(config.TARGET_COLS),
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        print("[Stage1] Loaded frozen base: PRISMv3 (no-gate encoder + target_head)")
        return m

    # Simple MLP: TargetMLP(8→Y) with keys model.0.weight ...
    from prism.encoder import TargetMLP
    m = TargetMLP(input_dim=len(config.INPUT_COLS), output_dim=len(config.TARGET_COLS)).to(device)
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    print("[Stage1] Loaded frozen base: Simple MLP (TargetMLP)")
    return m


def base_predict(base_model, x):
    with torch.no_grad():
        out = base_model(x)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    X_tensor, _, y_cpu, df = get_data_loaders()
    acp_tensor = torch.tensor(df[config.ACP_COLS].values.astype("float32"))
    dataset = TensorDataset(X_tensor, acp_tensor, y_cpu)
    train_ds, val_ds, test_ds, _ = split_dataset(dataset, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    # Stage 1: frozen base
    base = load_base_model(Path(args.base_ckpt), device)

    # Stage 2: trainable correction
    no_acp   = getattr(args, "no_acp",   False)
    acp_only = getattr(args, "acp_only", False)
    physics  = getattr(args, "physics",  False)
    if no_acp:
        stage2 = TwoStageNoACPModel(
            input_dim=len(config.INPUT_COLS),
            target_dim=len(config.TARGET_COLS),
        ).to(device)
    elif physics:
        stage2 = TwoStagePhysicsModel(
            input_dim=len(config.INPUT_COLS),
            acp_dim=len(config.ACP_COLS),
            target_dim=len(config.TARGET_COLS),
        ).to(device)
    elif acp_only:
        stage2 = TwoStageACPOnlyModel(
            input_dim=len(config.INPUT_COLS),
            acp_dim=len(config.ACP_COLS),
            target_dim=len(config.TARGET_COLS),
        ).to(device)
    else:
        stage2 = TwoStageModel(
            input_dim=len(config.INPUT_COLS),
            acp_dim=len(config.ACP_COLS),
            target_dim=len(config.TARGET_COLS),
        ).to(device)

    optimizer = torch.optim.Adam(stage2.parameters(), lr=config.MLP_LR)

    best_val_mse = float("inf")
    best_state = None
    no_improve = 0

    mode_tag = "NoACP" if no_acp else ("Physics(A+B)" if physics else ("ACPOnly" if acp_only else "ACP-supervised"))
    print(f"[START] Two-Stage PRISM ({mode_tag})  seed={args.seed}")
    print(f"        Base: {args.base_ckpt}")

    for epoch in range(1, args.epochs + 1):
        stage2.train()
        losses = []

        for xb, acpb, yb in train_loader:
            xb, acpb, yb = xb.to(device), acpb.to(device), yb.to(device)
            optimizer.zero_grad()

            y_base = base_predict(base, xb)
            if no_acp:
                delta_y = stage2(xb)
                y_pred  = y_base + delta_y
                loss = F.mse_loss(y_pred, yb)
            else:
                # acp_only and standard both use TwoStageModel-compatible interface
                acp_hat, delta_y = stage2(xb, true_acp=acpb)  # teacher-force true ACP
                y_pred   = y_base + delta_y
                loss_y   = F.mse_loss(y_pred, yb)
                loss_acp = F.mse_loss(acp_hat, acpb) * args.acp_lambda
                loss = loss_y + loss_acp
            # Perturbation-based monotonicity on Y_pred (ceteris paribus)
            if args.mono_lambda > 0:
                def _y_forward(x):
                    yb_ = base_predict(base, x)
                    if no_acp:
                        return yb_ + stage2(x)
                    ah_, dy_ = stage2(x)   # inference-mode: ACP_hat (no teacher-forcing)
                    return yb_ + dy_
                loss = loss + args.mono_lambda * perturbation_monotonicity_loss(
                    _y_forward, xb, y_pred, EXT_MONO_PAIRS_Y)
                # ACP_hat monotonicity: enforce physics constraints on ACP predictions
                if not no_acp:
                    loss = loss + args.mono_lambda * perturbation_monotonicity_loss(
                        stage2.acp_head, xb, acp_hat, ACP_MONO_PAIRS)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation (feasible inference)
        stage2.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, _, yb in val_loader:
                xb = xb.to(device)
                y_base = base_predict(base, xb)
                delta_y = stage2.predict(xb)     # NoACP: just h_corr; ACP: uses ACP_hat
                all_preds.append((y_base + delta_y).cpu())
                all_labels.append(yb)

        val_mse_each, _, _, _, _ = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))
        val_mse = val_mse_each.mean().item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(stage2.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.6f} | Val MSE: {val_mse:.6f} | Patience: {no_improve}")

        if no_improve >= config.PATIENCE:
            print("Early stopping.")
            break

    # Final evaluation
    stage2.load_state_dict(best_state)
    stage2.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, _, yb in test_loader:
            xb = xb.to(device)
            y_base = base_predict(base, xb)
            delta_y = stage2.predict(xb)
            all_preds.append((y_base + delta_y).cpu())
            all_labels.append(yb)

    mse, mape, rmse, mae, r2 = compute_metrics(torch.cat(all_labels), torch.cat(all_preds))

    results = {
        "Mean_MSE":  mse.mean().item(),
        "Mean_RMSE": rmse.mean().item(),
        "Mean_MAE":  mae.mean().item(),
        "Mean_R2":   r2.mean().item(),
        "MSE":  {n: m.item() for n, m in zip(config.TARGET_COLS, mse)},
        "RMSE": {n: m.item() for n, m in zip(config.TARGET_COLS, rmse)},
        "MAE":  {n: m.item() for n, m in zip(config.TARGET_COLS, mae)},
        "R2":   {n: m.item() for n, m in zip(config.TARGET_COLS, r2)},
    }

    print(f"\n[FINAL] Mean RMSE: {results['Mean_RMSE']:.6f}  MSE: {results['Mean_MSE']:.6f}")

    out_dir = Path(args.output_dir) / args.model_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / "summary.json")
    torch.save(stage2.state_dict(), out_dir / "stage2.pt")
    print(f"[DONE] Saved to {out_dir}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt",   required=True, help="Path to frozen Stage-1 checkpoint (V01 or Simple_MLP)")
    p.add_argument("--seed",        type=int, default=1)
    p.add_argument("--output-dir",  default="../results/ablation/seed_1")
    p.add_argument("--model-subdir", default="TwoStage_PRISM")
    p.add_argument("--acp-lambda",  type=float, default=0.5)
    p.add_argument("--mono-lambda", dest="mono_lambda", type=float, default=0.001)
    p.add_argument("--epochs",      type=int, default=1000)
    p.add_argument("--no-acp",      action="store_true",
                   help="Ablation: two-stage correction without ACP head or ACP loss")
    p.add_argument("--acp-only",    action="store_true",
                   help="ACP-only: correction uses [X, ACP_hat] directly, no CorrEncoder")
    p.add_argument("--physics",     action="store_true",
                   help="Direction A+B: monotone CorrHead(ACP_hat only, no X) — structural physics guarantee")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
