"""Shared model loading helpers for PRISM experiments."""

from __future__ import annotations

from pathlib import Path

import torch

from prism import config
from prism.encoder import TargetMLP
from prism.training.distilled_joint import PRISMv4Model
from prism.training.two_stage_prism import TwoStageACPOnlyModel, load_base_model


def load_simple_mlp(path: Path, device: torch.device) -> TargetMLP:
    model = TargetMLP(input_dim=len(config.INPUT_COLS), output_dim=len(config.TARGET_COLS)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def load_oracle_mlp(path: Path, device: torch.device) -> TargetMLP:
    model = TargetMLP(
        input_dim=len(config.INPUT_COLS) + len(config.ACP_COLS),
        output_dim=len(config.TARGET_COLS),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def load_prism_v4(path: Path, device: torch.device, use_gate: bool = True) -> PRISMv4Model:
    model = PRISMv4Model(
        input_dim=len(config.INPUT_COLS),
        bottleneck_dim=32,
        acp_dim=len(config.ACP_COLS),
        target_dim=len(config.TARGET_COLS),
        use_gate=use_gate,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def load_two_stage_acp_only(stage2_path: Path, base_path: Path, device: torch.device):
    base = load_base_model(base_path, device)
    stage2 = TwoStageACPOnlyModel(
        input_dim=len(config.INPUT_COLS),
        acp_dim=len(config.ACP_COLS),
        target_dim=len(config.TARGET_COLS),
    ).to(device)
    stage2.load_state_dict(torch.load(stage2_path, map_location=device, weights_only=True))
    return base.eval(), stage2.eval()
