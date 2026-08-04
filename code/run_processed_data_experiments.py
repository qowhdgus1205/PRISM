#!/usr/bin/env python
"""Quick PRISM-style experiments for the extra processed datasets.

The datasets under ``../data/processed`` were converted to an LEC-like schema:

    feature_*      -> deployable inputs X
    intermediate_* -> privileged/intermediate variables ACP
    target_*       -> prediction targets Y

This script runs a compact comparison:

    simple_mlp  : X -> Y
    two_stage   : X -> ACP_hat, [X, ACP_hat] -> Y
    oracle_mlp  : [X, true ACP] -> Y (infeasible upper reference)
    prism       : PRISMv4-style teacher distillation + ACP path + gated correction

The simpler ``prism_lite`` ablation is available through ``--models`` but is
not part of the default long run.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from prism.analysis.extra_baselines import DEFAULT_DL_BACKBONES, DEFAULT_ML_MODELS, extra_baseline_predictions
from prism.encoder import build_role_model
from prism.models.encoders import Encoder
from prism.models.heads import ACPHead
from prism.models.regressors import TargetMLP
from prism.losses import local_neighborhood_mixup
from prism.reproducibility import set_seed
from prism.training.distilled_joint import PRISMv4Model

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover - progress bars are optional at runtime
    trange = None


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    split_col: str | None = None
    train_value: str | None = None
    test_value: str | None = None
    group_col: str | None = None
    group_context_cols: tuple[str, ...] = ()
    role_preset: str = "prefix"


DATASETS = {
    "drivaernet": DatasetSpec(
        name="drivaernet",
        path=DATA_ROOT / "drivaernet" / "drivaernet_prism.csv",
        group_col="ID_body_family",
    ),
    "circuitnet": DatasetSpec(
        name="circuitnet",
        path=DATA_ROOT / "circuitnet" / "circuitnet_prism.csv",
        group_col="ID_design",
        group_context_cols=("ID_rtl_family",),
    ),
    "phm16_cmp": DatasetSpec(
        name="phm16_cmp",
        path=DATA_ROOT / "phm16_cmp" / "phm16_cmp_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_col="ID_wafer",
        group_context_cols=("ID_stage",),
    ),
    "kelmarsh_wind": DatasetSpec(
        name="kelmarsh_wind",
        path=DATA_ROOT / "kelmarsh_wind" / "kelmarsh_wind_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_turbine",),
    ),
    "nist_netzero_hvac": DatasetSpec(
        name="nist_netzero_hvac",
        path=DATA_ROOT / "nist_netzero_hvac" / "nist_netzero_hvac_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_year",),
    ),
    "bsm2_wastewater": DatasetSpec(
        name="bsm2_wastewater",
        path=DATA_ROOT / "bsm2_wastewater" / "bsm2_wastewater_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_col="ID_scenario",
    ),
    "utility_pv_scada": DatasetSpec(
        name="utility_pv_scada",
        path=DATA_ROOT / "utility_pv_scada" / "utility_pv_scada_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_inverter",),
    ),
    "multi_stage_manufacturing": DatasetSpec(
        name="multi_stage_manufacturing",
        path=DATA_ROOT / "multi_stage_manufacturing" / "multi_stage_manufacturing_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "pharmaceutical_manufacturing": DatasetSpec(
        name="pharmaceutical_manufacturing",
        path=DATA_ROOT / "pharmaceutical_manufacturing" / "pharmaceutical_manufacturing_prism.csv",
        group_col="ID_batch",
        group_context_cols=("ID_product_code",),
    ),
    "water_treatment_plant": DatasetSpec(
        name="water_treatment_plant",
        path=DATA_ROOT / "water_treatment_plant" / "water_treatment_plant_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "mining_flotation": DatasetSpec(
        name="mining_flotation",
        path=DATA_ROOT / "mining_flotation" / "mining_flotation_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "debutanizer_column": DatasetSpec(
        name="debutanizer_column",
        path=DATA_ROOT / "debutanizer_column" / "debutanizer_column_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "wave_energy_converters": DatasetSpec(
        name="wave_energy_converters",
        path=DATA_ROOT / "wave_energy_converters" / "wave_energy_converters_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_scenario",),
    ),
    "nasa_milling_wear": DatasetSpec(
        name="nasa_milling_wear",
        path=DATA_ROOT / "nasa_milling_wear" / "nasa_milling_wear_prism.csv",
        group_col="ID_case",
    ),
    "aventa_wind_turbine": DatasetSpec(
        name="aventa_wind_turbine",
        path=DATA_ROOT / "aventa_wind_turbine" / "aventa_wind_turbine_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "injection_molding_quality": DatasetSpec(
        name="injection_molding_quality",
        path=DATA_ROOT / "injection_molding_quality" / "injection_molding_quality_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "lattice_physics": DatasetSpec(
        name="lattice_physics",
        path=DATA_ROOT / "lattice_physics" / "lattice_physics_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "fdm_printing": DatasetSpec(
        name="fdm_printing",
        path=DATA_ROOT / "fdm_printing" / "fdm_printing_prism.csv",
    ),
    "uhpc_manufacturing": DatasetSpec(
        name="uhpc_manufacturing",
        path=DATA_ROOT / "uhpc_manufacturing" / "uhpc_manufacturing_prism.csv",
    ),
    "tool_path_machining": DatasetSpec(
        name="tool_path_machining",
        path=DATA_ROOT / "tool_path_machining" / "tool_path_machining_prism.csv",
        group_col="ID_part",
    ),
    "cnc_milling_tool_life": DatasetSpec(
        name="cnc_milling_tool_life",
        path=DATA_ROOT / "cnc_milling_tool_life" / "cnc_milling_tool_life_prism.csv",
        group_col="ID_tool",
    ),
    "steel_fatigue": DatasetSpec(
        name="steel_fatigue",
        path=DATA_ROOT / "steel_fatigue" / "steel_fatigue_prism.csv",
    ),
    "resistance_spot_welding": DatasetSpec(
        name="resistance_spot_welding",
        path=DATA_ROOT / "resistance_spot_welding" / "resistance_spot_welding_prism.csv",
    ),
    "cnc_turning_quality": DatasetSpec(
        name="cnc_turning_quality",
        path=DATA_ROOT / "cnc_turning_quality" / "cnc_turning_quality_prism.csv",
        group_col="ID_tool",
    ),
    "lpbf_ti64": DatasetSpec(
        name="lpbf_ti64",
        path=DATA_ROOT / "lpbf_ti64" / "lpbf_ti64_prism.csv",
        group_col="ID_parameter_set",
    ),
    "lpbf_alsi10mg": DatasetSpec(
        name="lpbf_alsi10mg",
        path=DATA_ROOT / "lpbf_alsi10mg" / "lpbf_alsi10mg_prism.csv",
        group_col="ID_parameter_set",
    ),
    "higgs": DatasetSpec(
        name="higgs",
        path=DATA_ROOT / "higgs" / "higgs_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "susy": DatasetSpec(
        name="susy",
        path=DATA_ROOT / "susy" / "susy_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "concrete_compressive_strength": DatasetSpec(
        name="concrete_compressive_strength",
        path=DATA_ROOT / "concrete_compressive_strength" / "concrete_compressive_strength_prism.csv",
    ),
    "seoul_bike_sharing": DatasetSpec(
        name="seoul_bike_sharing",
        path=DATA_ROOT / "seoul_bike_sharing" / "seoul_bike_sharing_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "electrical_grid_stability": DatasetSpec(
        name="electrical_grid_stability",
        path=DATA_ROOT / "electrical_grid_stability" / "electrical_grid_stability_prism.csv",
    ),
    "ai4i_predictive_maintenance": DatasetSpec(
        name="ai4i_predictive_maintenance",
        path=DATA_ROOT / "ai4i_predictive_maintenance" / "ai4i_predictive_maintenance_prism.csv",
    ),
    "steel_industry_energy": DatasetSpec(
        name="steel_industry_energy",
        path=DATA_ROOT / "steel_industry_energy" / "steel_industry_energy_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "steel_plates_faults": DatasetSpec(
        name="steel_plates_faults",
        path=DATA_ROOT / "steel_plates_faults" / "steel_plates_faults_prism.csv",
    ),
    "sensorless_drive_diagnosis": DatasetSpec(
        name="sensorless_drive_diagnosis",
        path=DATA_ROOT / "sensorless_drive_diagnosis" / "sensorless_drive_diagnosis_prism.csv",
    ),
    "robot_execution_failures": DatasetSpec(
        name="robot_execution_failures",
        path=DATA_ROOT / "robot_execution_failures" / "robot_execution_failures_prism.csv",
        group_context_cols=("ID_source",),
    ),
    "wall_following_robot": DatasetSpec(
        name="wall_following_robot",
        path=DATA_ROOT / "wall_following_robot" / "wall_following_robot_prism.csv",
    ),
    "occupancy_detection": DatasetSpec(
        name="occupancy_detection",
        path=DATA_ROOT / "occupancy_detection" / "occupancy_detection_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_source",),
    ),
    "room_occupancy_estimation": DatasetSpec(
        name="room_occupancy_estimation",
        path=DATA_ROOT / "room_occupancy_estimation" / "room_occupancy_estimation_prism.csv",
    ),
    "sgemm_gpu_kernel_performance": DatasetSpec(
        name="sgemm_gpu_kernel_performance",
        path=DATA_ROOT / "sgemm_gpu_kernel_performance" / "sgemm_gpu_kernel_performance_prism.csv",
    ),
    "average_localization_error": DatasetSpec(
        name="average_localization_error",
        path=DATA_ROOT / "average_localization_error" / "average_localization_error_prism.csv",
    ),
    "gas_sensor_flow_modulation": DatasetSpec(
        name="gas_sensor_flow_modulation",
        path=DATA_ROOT / "gas_sensor_flow_modulation" / "gas_sensor_flow_modulation_prism.csv",
    ),
    "gas_sensor_drift_concentration": DatasetSpec(
        name="gas_sensor_drift_concentration",
        path=DATA_ROOT / "gas_sensor_drift_concentration" / "gas_sensor_drift_concentration_prism.csv",
    ),
    "servo": DatasetSpec(
        name="servo",
        path=DATA_ROOT / "servo" / "servo_prism.csv",
    ),
    "cmaps": DatasetSpec(
        name="cmaps",
        path=DATA_ROOT / "cmaps" / "cmaps_all.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_col="ID_unit",
        group_context_cols=("dataset", "split"),
        role_preset="cmaps_rich",
    ),
    "tennessee_eastman": DatasetSpec(
        name="tennessee_eastman",
        path=DATA_ROOT / "tennessee_eastman" / "tennessee_eastman_lec_like.csv",
        role_preset="tennessee_time",
    ),
    "gas_turbine_co_nox": DatasetSpec(
        name="gas_turbine_co_nox",
        path=DATA_ROOT / "gas_turbine_co_nox" / "gas_turbine_co_nox_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
        group_context_cols=("ID_year", "split"),
    ),
    "naval_propulsion": DatasetSpec(
        name="naval_propulsion",
        path=DATA_ROOT / "naval_propulsion" / "naval_propulsion_prism.csv",
        group_col="ID_speed",
    ),
    "air_quality": DatasetSpec(
        name="air_quality",
        path=DATA_ROOT / "air_quality" / "air_quality_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "appliances_energy": DatasetSpec(
        name="appliances_energy",
        path=DATA_ROOT / "appliances_energy" / "appliances_energy_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "concrete_slump_test": DatasetSpec(
        name="concrete_slump_test",
        path=DATA_ROOT / "concrete_slump_test" / "concrete_slump_test_prism.csv",
    ),
    "student_performance": DatasetSpec(
        name="student_performance",
        path=DATA_ROOT / "student_performance" / "student_performance_prism.csv",
        group_col="ID_row",
        group_context_cols=("ID_subject",),
    ),
    "parkinsons_telemonitoring": DatasetSpec(
        name="parkinsons_telemonitoring",
        path=DATA_ROOT / "parkinsons_telemonitoring" / "parkinsons_telemonitoring_prism.csv",
        group_col="ID_subject",
    ),
    "energy_efficiency": DatasetSpec(
        name="energy_efficiency",
        path=DATA_ROOT / "energy_efficiency" / "energy_efficiency_prism.csv",
    ),
    "hydraulic_systems": DatasetSpec(
        name="hydraulic_systems",
        path=DATA_ROOT / "hydraulic_systems" / "hydraulic_systems_prism.csv",
    ),
    "household_power_consumption": DatasetSpec(
        name="household_power_consumption",
        path=DATA_ROOT / "household_power_consumption" / "household_power_consumption_prism.csv",
        split_col="split",
        train_value="train",
        test_value="test",
    ),
    "bike_sharing": DatasetSpec(
        name="bike_sharing",
        path=DATA_ROOT / "bike_sharing" / "bike_sharing_prism.csv",
        group_col="ID_date",
    ),
    "wine_quality": DatasetSpec(
        name="wine_quality",
        path=DATA_ROOT / "wine_quality" / "wine_quality_prism.csv",
        group_col="ID_row",
        group_context_cols=("ID_wine_type",),
    ),
    "superconductivity": DatasetSpec(
        name="superconductivity",
        path=DATA_ROOT / "superconductivity" / "superconductivity_prism.csv",
    ),
    "auto_mpg": DatasetSpec(
        name="auto_mpg",
        path=DATA_ROOT / "auto_mpg" / "auto_mpg_prism.csv",
    ),
    "airfoil_self_noise": DatasetSpec(
        name="airfoil_self_noise",
        path=DATA_ROOT / "airfoil_self_noise" / "airfoil_self_noise_prism.csv",
    ),
    "yacht_hydrodynamics": DatasetSpec(
        name="yacht_hydrodynamics",
        path=DATA_ROOT / "yacht_hydrodynamics" / "yacht_hydrodynamics_prism.csv",
    ),
    "combined_cycle_power_plant": DatasetSpec(
        name="combined_cycle_power_plant",
        path=DATA_ROOT / "combined_cycle_power_plant" / "combined_cycle_power_plant_prism.csv",
    ),
    "real_estate_valuation": DatasetSpec(
        name="real_estate_valuation",
        path=DATA_ROOT / "real_estate_valuation" / "real_estate_valuation_prism.csv",
    ),
}

TREE_MODELS = list(DEFAULT_ML_MODELS)
DL_MODELS = [f"{backbone}_mlp" for backbone in DEFAULT_DL_BACKBONES]
PRISM_TREE_MODELS = [f"prism_z_{name}" for name in TREE_MODELS] + [f"prism_z_acp_{name}" for name in TREE_MODELS]
MODEL_CHOICES = [
    "simple_mlp",
    "two_stage_mlp",
    "oracle_mlp",
    "prism",
    "full_prism",
    "prism_lite",
    *TREE_MODELS,
    *DL_MODELS,
    *PRISM_TREE_MODELS,
]
MODEL_ALIASES = {"full_prism": "prism"}


class PRISMLite(nn.Module):
    """Small generic PRISM variant for arbitrary feature/intermediate/target dims."""

    def __init__(self, input_dim: int, acp_dim: int, target_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
        self.acp_head = ACPHead(latent_dim=latent_dim, acp_dim=acp_dim)
        self.direct_head = TargetMLP(input_dim=latent_dim, output_dim=target_dim, dropout=dropout)
        self.correction_head = nn.Sequential(
            nn.Linear(latent_dim + acp_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, target_dim),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, x):
        z = self.encoder(x)
        acp_hat = self.acp_head(z)
        y = self.direct_head(z) + self.correction_head(torch.cat([z, acp_hat], dim=1))
        return y, acp_hat


def supervised_acp_contrastive(embeddings, acp_targets, temperature=0.07, sigma=0.5):
    z = F.normalize(embeddings, dim=1)
    sim = torch.matmul(z, z.t()) / temperature
    dist = torch.cdist(acp_targets, acp_targets)
    mask = torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)
    mask = mask * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
    return -((mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)).mean()


def teacher_repr(teacher, x_acp):
    if hasattr(teacher, "encode"):
        return teacher.encode(x_acp)
    return teacher.model[:-1](x_acp)


def numeric_role_columns(df: pd.DataFrame, spec: DatasetSpec):
    if spec.role_preset == "cmaps_rich":
        sensor_feature_ids = {2, 3, 4, 7, 11, 12, 15, 20, 21}
        feature_cols = [
            c for c in df.columns
            if c.startswith("feature_")
            or (c.startswith("intermediate_sensor_") and int(c.rsplit("_", 1)[1]) in sensor_feature_ids)
        ]
        acp_cols = [
            c for c in df.columns
            if c.startswith("intermediate_sensor_") and c not in feature_cols
        ]
        target_cols = [c for c in df.columns if c.startswith("target_")]
    elif spec.role_preset == "tennessee_time":
        feature_cols = ["ID_time"] + [
            c for c in df.columns if c.startswith("feature_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        acp_cols = [c for c in df.columns if c.startswith("intermediate_") and pd.api.types.is_numeric_dtype(df[c])]
        target_cols = [c for c in df.columns if c.startswith("target_") and pd.api.types.is_numeric_dtype(df[c])]
    elif spec.role_preset == "battery_cell":
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        acp_cols = [c for c in df.columns if c.startswith("intermediate_first10_")]
        target_cols = [c for c in df.columns if c.startswith("target_")]
    elif spec.role_preset == "battery_cell_compact":
        feature_cols = [
            c for c in df.columns
            if c.startswith("feature_max_voltage_limit_in_V")
            or c.startswith("feature_min_voltage_limit_in_V")
            or c.startswith("feature_charge_protocol_")
            or c.startswith("feature_discharge_protocol_")
        ]
        acp_cols = [
            "intermediate_first10_current_in_A_mean",
            "intermediate_first10_current_in_A_std",
            "intermediate_first10_current_in_A_max",
            "intermediate_first10_voltage_in_V_mean",
            "intermediate_first10_voltage_in_V_std",
            "intermediate_first10_voltage_in_V_min",
            "intermediate_first10_charge_capacity_in_Ah_mean",
            "intermediate_first10_charge_capacity_in_Ah_std",
            "intermediate_first10_charge_capacity_in_Ah_max",
            "intermediate_first10_discharge_capacity_in_Ah_mean",
            "intermediate_first10_discharge_capacity_in_Ah_std",
            "intermediate_first10_discharge_capacity_in_Ah_max",
            "intermediate_first10_temperature_in_C_mean",
        ]
        acp_cols = [c for c in acp_cols if c in df.columns]
        target_cols = [c for c in df.columns if c.startswith("target_")]
    elif spec.role_preset == "battery_cycle":
        feature_cols = [
            c for c in df.columns
            if c.startswith("feature_")
        ]
        acp_cols = [c for c in df.columns if c.startswith("intermediate_")]
        target_cols = [c for c in df.columns if c.startswith("target_")]
    else:
        feature_cols = [c for c in df.columns if c.startswith("feature_") and pd.api.types.is_numeric_dtype(df[c])]
        acp_cols = [c for c in df.columns if c.startswith("intermediate_") and pd.api.types.is_numeric_dtype(df[c])]
        target_cols = [c for c in df.columns if c.startswith("target_") and pd.api.types.is_numeric_dtype(df[c])]

    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    acp_cols = [c for c in acp_cols if pd.api.types.is_numeric_dtype(df[c])]
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]

    usable = feature_cols + acp_cols + target_cols
    nonempty = [c for c in usable if not df[c].isna().all()]
    nonconstant = [c for c in nonempty if c in target_cols or df[c].nunique(dropna=True) > 1]
    dropped = sorted(set(usable) - set(nonconstant))
    feature_cols = [c for c in feature_cols if c in nonconstant]
    acp_cols = [c for c in acp_cols if c in nonconstant]
    target_cols = [c for c in target_cols if c in nonconstant]
    return feature_cols, acp_cols, target_cols, dropped


def encode_categorical_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    categorical = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if not categorical:
        return df
    encoded = pd.get_dummies(df, columns=categorical, dummy_na=False, dtype=np.float32)
    return encoded


def split_indices(df: pd.DataFrame, spec: DatasetSpec, seed: int, test_size: float, val_size: float):
    idx = np.arange(len(df))
    groups_all = group_values(df, spec)
    if spec.split_col:
        trainval_idx = idx[df[spec.split_col].astype(str).to_numpy() == spec.train_value]
        test_idx = idx[df[spec.split_col].astype(str).to_numpy() == spec.test_value]
    elif spec.group_col:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(splitter.split(idx, groups=groups_all))
    else:
        trainval_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)

    if spec.group_col and spec.group_col in df.columns:
        rel = np.arange(len(trainval_idx))
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1000)
        tr_rel, val_rel = next(splitter.split(rel, groups=groups_all.iloc[trainval_idx]))
        train_idx, val_idx = trainval_idx[tr_rel], trainval_idx[val_rel]
    else:
        train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size, random_state=seed + 1000, shuffle=True)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def group_values(df: pd.DataFrame, spec: DatasetSpec):
    if not spec.group_col or spec.group_col not in df.columns:
        return None
    cols = [c for c in (*spec.group_context_cols, spec.group_col) if c in df.columns]
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("::".join, axis=1)


def split_diagnostics(df: pd.DataFrame, spec: DatasetSpec, train_idx, val_idx, test_idx, target_cols):
    diagnostics = {
        "split_group_col": spec.group_col or "",
        "train_val_group_overlap": 0,
        "train_test_group_overlap": 0,
        "val_test_group_overlap": 0,
        "target_constant_within_group": False,
    }
    groups_all = group_values(df, spec)
    if groups_all is not None:
        train_groups = set(groups_all.iloc[train_idx])
        val_groups = set(groups_all.iloc[val_idx])
        test_groups = set(groups_all.iloc[test_idx])
        diagnostics.update({
            "train_val_group_overlap": len(train_groups & val_groups),
            "train_test_group_overlap": len(train_groups & test_groups),
            "val_test_group_overlap": len(val_groups & test_groups),
        })
        if len(target_cols) == 1:
            diagnostics["target_constant_within_group"] = bool(
                df.assign(__group_key__=groups_all)
                .groupby("__group_key__")[target_cols[0]]
                .nunique(dropna=True)
                .le(1)
                .all()
            )
    return diagnostics


def make_arrays(df, cols, indices, scaler=None, fit=False):
    arr = df.iloc[indices][cols].astype(np.float64).to_numpy()
    if fit:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr).astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values remained after scaling columns: {cols}")
        return arr, scaler
    arr = scaler.transform(arr).astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values remained after scaling columns: {cols}")
    return arr


def loader(*arrays, batch_size: int, shuffle: bool, drop_last: bool = False):
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_supervised(
    model,
    train_loader,
    val_loader,
    epochs,
    patience,
    lr,
    device,
    prism=False,
    acp_weight=0.25,
    progress=True,
    desc="train",
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    stale = 0

    if progress and trange is not None:
        epoch_iter = trange(1, epochs + 1, desc=desc, leave=False, dynamic_ncols=True)
    else:
        epoch_iter = range(1, epochs + 1)

    for epoch in epoch_iter:
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad()
            if prism:
                xb, acpb, yb = [x.to(device) for x in batch]
                yp, acpp = model(xb)
                loss = F.mse_loss(yp, yb) + acp_weight * F.mse_loss(acpp, acpb)
            else:
                xb, yb = [x.to(device) for x in batch]
                loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                if prism:
                    xb, acpb, yb = [x.to(device) for x in batch]
                    yp, acpp = model(xb)
                    vals.append((F.mse_loss(yp, yb) + acp_weight * F.mse_loss(acpp, acpb)).item())
                else:
                    xb, yb = [x.to(device) for x in batch]
                    vals.append(F.mse_loss(model(xb), yb).item())
        val = float(np.mean(vals))
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

        if progress and trange is not None:
            epoch_iter.set_postfix(
                train=f"{float(np.mean(train_losses)):.4f}",
                val=f"{val:.4f}",
                best=f"{best_val:.4f}",
                patience=f"{stale}/{patience}",
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def pretrain_full_prism_acp(model, train_loader, val_loader, epochs, patience, lr, device, progress=True, desc="pretrain_acp"):
    if epochs <= 0:
        return model

    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.acp_head.parameters()), lr=lr)
    best_state = None
    best_val = float("inf")
    stale = 0
    if progress and trange is not None:
        epoch_iter = trange(1, epochs + 1, desc=desc, leave=False, dynamic_ncols=True)
    else:
        epoch_iter = range(1, epochs + 1)

    for epoch in epoch_iter:
        model.train()
        train_losses = []
        for xb, acpb, _ in train_loader:
            xb, acpb = xb.to(device), acpb.to(device)
            opt.zero_grad()
            acp_hat = model.acp_head(model.encoder(xb))
            loss = F.mse_loss(acp_hat, acpb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, acpb, _ in val_loader:
                xb, acpb = xb.to(device), acpb.to(device)
                vals.append(F.mse_loss(model.acp_head(model.encoder(xb)), acpb).item())
        val = float(np.mean(vals))
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if progress and trange is not None:
            epoch_iter.set_postfix(
                train=f"{float(np.mean(train_losses)):.4f}",
                val=f"{val:.4f}",
                best=f"{best_val:.4f}",
                patience=f"{stale}/{patience}",
            )
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_full_prism(
    student,
    teacher,
    train_loader,
    val_loader,
    epochs,
    patience,
    lr,
    device,
    acp_lambda,
    distill_lambda,
    contrastive_lambda,
    mixup_lambda,
    mixup_alpha,
    mixup_k,
    progress=True,
    desc="full_prism",
):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    stale = 0
    if progress and trange is not None:
        epoch_iter = trange(1, epochs + 1, desc=desc, leave=False, dynamic_ncols=True)
    else:
        epoch_iter = range(1, epochs + 1)

    teacher.eval()
    for epoch in epoch_iter:
        student.train()
        train_totals, train_y, train_acp, train_distill, train_ctr, train_mix = [], [], [], [], [], []
        for xb, acpb, yb in train_loader:
            xb, acpb, yb = xb.to(device), acpb.to(device), yb.to(device)
            opt.zero_grad()

            y_hat, acp_hat, _, z_contrastive = student(xb)
            h_student = student.encoder(xb)
            with torch.no_grad():
                h_teacher = teacher_repr(teacher, torch.cat([xb, acpb], dim=1))

            loss_y = F.mse_loss(y_hat, yb)
            loss_acp = F.mse_loss(acp_hat, acpb)
            loss_distill = F.mse_loss(student.distill_proj(h_student), h_teacher)
            if contrastive_lambda > 0 and len(xb) > 1:
                loss_ctr = supervised_acp_contrastive(z_contrastive, acpb)
            else:
                loss_ctr = torch.tensor(0.0, device=device)
            if mixup_lambda > 0:
                mixed = local_neighborhood_mixup(xb, acpb, yb, k=mixup_k, alpha=mixup_alpha)
                if mixed is not None:
                    x_mix, acp_mix, y_mix = mixed
                    y_mix_hat, acp_mix_hat, _, _ = student(x_mix)
                    loss_mix = F.mse_loss(y_mix_hat, y_mix) + acp_lambda * F.mse_loss(acp_mix_hat, acp_mix)
                else:
                    loss_mix = torch.tensor(0.0, device=device)
            else:
                loss_mix = torch.tensor(0.0, device=device)
            loss = (
                loss_y
                + acp_lambda * loss_acp
                + distill_lambda * loss_distill
                + contrastive_lambda * loss_ctr
                + mixup_lambda * loss_mix
            )
            loss.backward()
            opt.step()

            train_totals.append(loss.item())
            train_y.append(loss_y.item())
            train_acp.append(loss_acp.item())
            train_distill.append(loss_distill.item())
            train_ctr.append(loss_ctr.item())
            train_mix.append(loss_mix.item())

        student.eval()
        vals = []
        with torch.no_grad():
            for xb, _, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vals.append(F.mse_loss(student.predict(xb), yb).item())
        val = float(np.mean(vals))
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(student.state_dict())
            stale = 0
        else:
            stale += 1

        if progress and trange is not None:
            epoch_iter.set_postfix(
                total=f"{float(np.mean(train_totals)):.4f}",
                y=f"{float(np.mean(train_y)):.4f}",
                acp=f"{float(np.mean(train_acp)):.4f}",
                dist=f"{float(np.mean(train_distill)):.4f}",
                ctr=f"{float(np.mean(train_ctr)):.4f}",
                mix=f"{float(np.mean(train_mix)):.4f}",
                val=f"{val:.4f}",
                best=f"{best_val:.4f}",
                patience=f"{stale}/{patience}",
            )
        if stale >= patience:
            break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


def predict(model, x, device, prism=False, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in loader(x, batch_size=batch_size, shuffle=False):
            xb = xb.to(device)
            yp = model(xb)[0] if prism else model(xb)
            out.append(yp.cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_full_prism(model, x, device, batch_size=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in loader(x, batch_size=batch_size, shuffle=False):
            out.append(model.predict(xb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def prism_representation(model, x, device, include_acp: bool, batch_size=4096):
    model.eval()
    chunks = []
    with torch.no_grad():
        for (xb,) in loader(x, batch_size=batch_size, shuffle=False):
            xb = xb.to(device)
            z = model.encoder(xb)
            if include_acp:
                acp_hat = model.acp_head(z)
                z = torch.cat([z, acp_hat], dim=1)
            chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def metrics(y_true_std, y_pred_std, y_scaler, target_cols):
    y_true = y_scaler.inverse_transform(y_true_std)
    y_pred = y_scaler.inverse_transform(y_pred_std)
    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "mean_rmse": float(np.mean(rmse)),
        "mean_mae": float(np.mean(mae)),
        "mean_r2": float(np.mean(r2)),
        "target_rmse": {k: float(v) for k, v in zip(target_cols, rmse)},
        "target_r2": {k: float(v) for k, v in zip(target_cols, r2)},
    }


def run_one_dataset(spec: DatasetSpec, args, seed: int, device):
    raw = encode_categorical_features(pd.read_csv(spec.path))
    x_cols, acp_cols, y_cols, dropped = numeric_role_columns(raw, spec)
    needed = x_cols + acp_cols + y_cols
    numeric = raw[needed].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    valid_rows = (
        np.isfinite(values).all(axis=1)
        & (np.abs(values) <= np.finfo(np.float32).max).all(axis=1)
    )
    invalid_rows = int((~valid_rows).sum())
    if invalid_rows:
        print(
            f"[CLEAN] {spec.name}: dropping {invalid_rows} rows with "
            "NaN, infinity, or values outside float32 range",
            flush=True,
        )
    df = raw.loc[valid_rows].copy()
    df.loc[:, needed] = numeric.loc[valid_rows]
    df = df.reset_index(drop=True)
    if args.max_rows_per_dataset and len(df) > args.max_rows_per_dataset:
        if spec.split_col and spec.split_col in df.columns:
            parts = []
            for _, part in df.groupby(spec.split_col, sort=False):
                n_part = max(1, int(round(args.max_rows_per_dataset * len(part) / len(df))))
                parts.append(part.sample(n=min(n_part, len(part)), random_state=seed))
            df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        elif spec.group_col and spec.group_col in df.columns:
            groups = df[spec.group_col].drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
            chosen = []
            total = 0
            for group in groups:
                part = df[df[spec.group_col] == group]
                chosen.append(part)
                total += len(part)
                if total >= args.max_rows_per_dataset:
                    break
            df = pd.concat(chosen, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(n=args.max_rows_per_dataset, random_state=seed).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"{spec.name}: not enough complete rows after dropping NaNs ({len(df)})")
    if not x_cols or not acp_cols or not y_cols:
        raise ValueError(f"{spec.name}: missing role columns x={len(x_cols)} acp={len(acp_cols)} y={len(y_cols)}")

    train_idx, val_idx, test_idx = split_indices(df, spec, seed, args.test_size, args.val_size)
    diag = split_diagnostics(df, spec, train_idx, val_idx, test_idx, y_cols)
    if args.fail_on_group_overlap and spec.group_col:
        overlaps = (
            diag["train_val_group_overlap"]
            + diag["train_test_group_overlap"]
            + diag["val_test_group_overlap"]
        )
        if overlaps:
            raise RuntimeError(f"{spec.name}: group leakage detected: {diag}")
    print(
        f"      rows={len(df)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"X={len(x_cols)} ACP={len(acp_cols)} Y={len(y_cols)} diag={diag}",
        flush=True,
    )

    X_tr, x_scaler = make_arrays(df, x_cols, train_idx, fit=True)
    X_val = make_arrays(df, x_cols, val_idx, scaler=x_scaler)
    X_te = make_arrays(df, x_cols, test_idx, scaler=x_scaler)
    A_tr, a_scaler = make_arrays(df, acp_cols, train_idx, fit=True)
    A_val = make_arrays(df, acp_cols, val_idx, scaler=a_scaler)
    A_te = make_arrays(df, acp_cols, test_idx, scaler=a_scaler)
    Y_tr, y_scaler = make_arrays(df, y_cols, train_idx, fit=True)
    Y_val = make_arrays(df, y_cols, val_idx, scaler=y_scaler)
    Y_te = make_arrays(df, y_cols, test_idx, scaler=y_scaler)

    batch = min(args.batch_size, max(8, len(train_idx)))
    rows = []

    requested = {MODEL_ALIASES.get(model, model) for model in args.models}

    if "simple_mlp" in requested:
        simple = build_role_model(
            "simple_mlp",
            input_dim=X_tr.shape[1],
            output_dim=Y_tr.shape[1],
            profile=args.model_profile,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)
        train_supervised(
            simple,
            loader(X_tr, Y_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1),
            loader(X_val, Y_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/simple",
        )
        rows.append(("simple_mlp", metrics(Y_te, predict(simple, X_te, device), y_scaler, y_cols)))

    requested_tree_models = [model for model in TREE_MODELS if model in requested]
    if requested_tree_models:
        tree_preds = extra_baseline_predictions(
            X_tr=X_tr,
            y_tr=Y_tr,
            X_val=X_val,
            y_val=Y_val,
            X_te=X_te,
            y_mean=y_scaler.mean_.reshape(1, -1),
            y_std=y_scaler.scale_.reshape(1, -1),
            output_dim=Y_tr.shape[1],
            latent_dim=args.latent_dim,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=batch,
            seed=seed,
            ml_models=requested_tree_models,
            dl_backbones=(),
            include_x_acp=False,
        )
        for model_name, pred_raw in tree_preds:
            pred_std = y_scaler.transform(pred_raw)
            rows.append((model_name, metrics(Y_te, pred_std, y_scaler, y_cols)))

    requested_dl_backbones = [
        model.removesuffix("_mlp")
        for model in DL_MODELS
        if model in requested
    ]
    if requested_dl_backbones:
        dl_preds = extra_baseline_predictions(
            X_tr=X_tr,
            y_tr=Y_tr,
            X_val=X_val,
            y_val=Y_val,
            X_te=X_te,
            y_mean=y_scaler.mean_.reshape(1, -1),
            y_std=y_scaler.scale_.reshape(1, -1),
            output_dim=Y_tr.shape[1],
            latent_dim=args.latent_dim,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=batch,
            seed=seed,
            ml_models=(),
            dl_backbones=requested_dl_backbones,
            include_x_acp=False,
        )
        for model_name, pred_raw in dl_preds:
            pred_std = y_scaler.transform(pred_raw)
            rows.append((model_name, metrics(Y_te, pred_std, y_scaler, y_cols)))

    if "two_stage_mlp" in requested:
        acp_model = build_role_model(
            "acp_predictor",
            input_dim=X_tr.shape[1],
            output_dim=A_tr.shape[1],
            profile=args.model_profile,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)
        train_supervised(
            acp_model,
            loader(X_tr, A_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1),
            loader(X_val, A_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/acp",
        )
        Ahat_tr = predict(acp_model, X_tr, device)
        Ahat_val = predict(acp_model, X_val, device)
        Ahat_te = predict(acp_model, X_te, device)
        two_stage = build_role_model(
            "two_stage_mlp",
            input_dim=X_tr.shape[1] + A_tr.shape[1],
            output_dim=Y_tr.shape[1],
            profile=args.model_profile,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)
        train_supervised(
            two_stage,
            loader(np.hstack([X_tr, Ahat_tr]), Y_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1),
            loader(np.hstack([X_val, Ahat_val]), Y_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/two_stage",
        )
        rows.append(("two_stage_mlp", metrics(Y_te, predict(two_stage, np.hstack([X_te, Ahat_te]), device), y_scaler, y_cols)))

    oracle = None
    requested_prism_tree_models = [model for model in PRISM_TREE_MODELS if model in requested]
    needs_full_prism = "prism" in requested or requested_prism_tree_models

    if "oracle_mlp" in requested or needs_full_prism:
        oracle = build_role_model(
            "oracle_mlp",
            input_dim=X_tr.shape[1] + A_tr.shape[1],
            output_dim=Y_tr.shape[1],
            profile=args.model_profile,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        ).to(device)
        train_supervised(
            oracle,
            loader(np.hstack([X_tr, A_tr]), Y_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1),
            loader(np.hstack([X_val, A_val]), Y_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/oracle",
        )
        if "oracle_mlp" in requested:
            rows.append(("oracle_mlp", metrics(Y_te, predict(oracle, np.hstack([X_te, A_te]), device), y_scaler, y_cols)))

    if "prism_lite" in requested:
        prism = PRISMLite(X_tr.shape[1], A_tr.shape[1], Y_tr.shape[1], latent_dim=args.latent_dim, dropout=args.dropout).to(device)
        train_supervised(
            prism,
            loader(X_tr, A_tr, Y_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1),
            loader(X_val, A_val, Y_val, batch_size=batch, shuffle=False),
            args.epochs,
            args.patience,
            args.lr,
            device,
            prism=True,
            acp_weight=args.acp_weight,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/prism_lite",
        )
        rows.append(("prism_lite", metrics(Y_te, predict(prism, X_te, device, prism=True), y_scaler, y_cols)))

    if needs_full_prism:
        full_prism = PRISMv4Model(
            input_dim=X_tr.shape[1],
            bottleneck_dim=args.latent_dim,
            acp_dim=A_tr.shape[1],
            target_dim=Y_tr.shape[1],
            dropout=args.dropout,
            use_gate=True,
            use_confidence_blend=args.confidence_blend,
            teacher_repr_dim=args.latent_dim if args.model_profile == "shared_encoder" else 64,
        ).to(device)
        full_train_loader = loader(X_tr, A_tr, Y_tr, batch_size=batch, shuffle=True, drop_last=len(train_idx) > 1)
        full_val_loader = loader(X_val, A_val, Y_val, batch_size=batch, shuffle=False)
        pretrain_full_prism_acp(
            full_prism,
            full_train_loader,
            full_val_loader,
            args.pretrain_acp_epochs,
            args.pretrain_acp_patience,
            args.pretrain_acp_lr if args.pretrain_acp_lr is not None else args.lr,
            device,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/prism_pre_acp",
        )
        train_full_prism(
            full_prism,
            oracle,
            full_train_loader,
            full_val_loader,
            args.epochs,
            args.patience,
            args.lr,
            device,
            acp_lambda=args.full_acp_lambda,
            distill_lambda=args.distill_lambda,
            contrastive_lambda=args.contrastive_lambda,
            mixup_lambda=args.mixup_lambda,
            mixup_alpha=args.mixup_alpha,
            mixup_k=args.mixup_k,
            progress=not args.no_progress,
            desc=f"{spec.name}/s{seed}/prism",
        )
        if "prism" in requested:
            rows.append(("prism", metrics(Y_te, predict_full_prism(full_prism, X_te, device), y_scaler, y_cols)))

        for prefix, include_acp in [("prism_z_acp", True), ("prism_z", False)]:
            requested_for_prefix = [
                model.removeprefix(f"{prefix}_")
                for model in requested_prism_tree_models
                if model.startswith(f"{prefix}_")
                and (prefix != "prism_z" or not model.startswith("prism_z_acp_"))
            ]
            if not requested_for_prefix:
                continue
            Z_tr = prism_representation(full_prism, X_tr, device, include_acp=include_acp)
            Z_val = prism_representation(full_prism, X_val, device, include_acp=include_acp)
            Z_te = prism_representation(full_prism, X_te, device, include_acp=include_acp)
            tree_preds = extra_baseline_predictions(
                X_tr=Z_tr,
                y_tr=Y_tr,
                X_val=Z_val,
                y_val=Y_val,
                X_te=Z_te,
                y_mean=y_scaler.mean_.reshape(1, -1),
                y_std=y_scaler.scale_.reshape(1, -1),
                output_dim=Y_tr.shape[1],
                latent_dim=args.latent_dim,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                batch_size=batch,
                seed=seed,
                ml_models=requested_for_prefix,
                dl_backbones=(),
                include_x_acp=False,
            )
            for model_name, pred_raw in tree_preds:
                pred_std = y_scaler.transform(pred_raw)
                rows.append((f"{prefix}_{model_name}", metrics(Y_te, pred_std, y_scaler, y_cols)))

    result_rows = []
    for model, m in rows:
        result_rows.append({
            "dataset": spec.name,
            "seed": seed,
            "model": model,
            "n_rows": len(df),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "n_features": len(x_cols),
            "n_intermediates": len(acp_cols),
            "n_targets": len(y_cols),
            "mean_rmse": m["mean_rmse"],
            "mean_mae": m["mean_mae"],
            "mean_r2": m["mean_r2"],
            "target_rmse_json": json.dumps(m["target_rmse"], sort_keys=True),
            "target_r2_json": json.dumps(m["target_r2"], sort_keys=True),
            "dropped_all_nan_cols": ",".join(dropped),
            **diag,
        })
    return result_rows


def aggregate(results: pd.DataFrame):
    return (
        results.groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            mean_mae=("mean_mae", "mean"),
            mean_r2=("mean_r2", "mean"),
            std_r2=("mean_r2", "std"),
            n_seeds=("seed", "nunique"),
            n_train=("n_train", "first"),
            n_test=("n_test", "first"),
            n_features=("n_features", "first"),
            n_intermediates=("n_intermediates", "first"),
            n_targets=("n_targets", "first"),
        )
        .sort_values(["dataset", "mean_rmse", "model"])
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["simple_mlp", "two_stage_mlp", "oracle_mlp", "prism"],
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", "--bottleneck-dim", dest="latent_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--acp-weight", type=float, default=0.25)
    parser.add_argument("--full-acp-lambda", type=float, default=0.5)
    parser.add_argument("--distill-lambda", type=float, default=0.05)
    parser.add_argument("--contrastive-lambda", type=float, default=0.0)
    parser.add_argument("--mixup-lambda", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--mixup-k", type=int, default=5)
    parser.add_argument("--pretrain-acp-epochs", type=int, default=200)
    parser.add_argument("--pretrain-acp-patience", type=int, default=40)
    parser.add_argument("--pretrain-acp-lr", type=float, default=None)
    parser.add_argument("--confidence-blend", action="store_true")
    parser.add_argument("--model-profile", default="shared_encoder")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="../results/processed_data_experiments")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm epoch progress bars.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing result checkpoint and rerun every requested combination.",
    )
    parser.add_argument("--fail-on-group-overlap", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "processed_extra_results.csv"
    aggregate_path = out_dir / "processed_extra_aggregate.csv"
    if result_path.exists() and not args.no_resume:
        existing = pd.read_csv(result_path)
        all_rows = existing.to_dict("records")
        print(f"[RESUME] loaded {len(existing)} completed result rows from {result_path}")
    else:
        all_rows = []

    def save_checkpoint(rows):
        results = pd.DataFrame(rows)
        results = results.drop_duplicates(subset=["dataset", "seed", "model"], keep="last")
        temp_result = result_path.with_suffix(".csv.tmp")
        results.to_csv(temp_result, index=False)
        temp_result.replace(result_path)
        agg = aggregate(results)
        temp_aggregate = aggregate_path.with_suffix(".csv.tmp")
        agg.to_csv(temp_aggregate, index=False)
        temp_aggregate.replace(aggregate_path)
        return results, agg

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for seed in args.seeds:
            requested = {MODEL_ALIASES.get(model, model) for model in args.models}
            completed = {
                row["model"] for row in all_rows
                if row["dataset"] == dataset and int(row["seed"]) == seed
            }
            remaining = sorted(requested - completed)
            if not remaining:
                print(f"[SKIP] dataset={dataset} seed={seed}: all requested models complete")
                continue
            print(f"[RUN] dataset={dataset} seed={seed} device={device}", flush=True)
            set_seed(seed)
            run_args = copy.copy(args)
            run_args.models = remaining
            all_rows.extend(run_one_dataset(spec, run_args, seed, device))
            results, _ = save_checkpoint(all_rows)
            all_rows = results.to_dict("records")
            print(f"[CHECKPOINT] saved {len(results)} rows to {result_path}", flush=True)

    results, agg = save_checkpoint(all_rows)

    print("\n[AGGREGATE]")
    cols = ["dataset", "model", "mean_rmse", "std_rmse", "mean_r2", "n_seeds", "n_train", "n_test", "n_features", "n_intermediates", "n_targets"]
    print(agg[cols].to_string(index=False))
    print(f"\n[DONE] wrote {result_path}")
    print(f"[DONE] wrote {aggregate_path}")


if __name__ == "__main__":
    main()
