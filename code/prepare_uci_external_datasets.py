#!/usr/bin/env python
"""Convert downloaded UCI external datasets to PRISM processed schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"


DOWNLOADS = {
    "airfoil_self_noise": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
    "energy_efficiency": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    "hydraulic_systems": "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
    "household_power_consumption": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "concrete_slump_test": "https://archive.ics.uci.edu/static/public/182/concrete+slump+test.zip",
    "wine_quality": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "superconductivity": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "auto_mpg": "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip",
    "combined_cycle_power_plant": "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    "air_quality": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
    "parkinsons_telemonitoring": "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
    "real_estate_valuation": "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip",
    "student_performance": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "concrete_compressive_strength": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    "seoul_bike_sharing": "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip",
    "electrical_grid_stability": "https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip",
    "ai4i_predictive_maintenance": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "steel_industry_energy": "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
    "steel_plates_faults": "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip",
    "sensorless_drive_diagnosis": "https://archive.ics.uci.edu/static/public/325/dataset+for+sensorless+drive+diagnosis.zip",
    "robot_execution_failures": "https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip",
    "wall_following_robot": "https://archive.ics.uci.edu/static/public/194/wall+following+robot+navigation+data.zip",
    "occupancy_detection": "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
    "room_occupancy_estimation": "https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip",
    "sgemm_gpu_kernel_performance": "https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip",
    "average_localization_error": "https://archive.ics.uci.edu/static/public/844/average%2Blocalization%2Berror%2B%28ale%29%2Bin%2Bsensor%2Bnode%2Blocalization%2Bprocess%2Bin%2Bwsns.zip",
    "gas_sensor_flow_modulation": "https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip",
    "gas_sensor_drift_concentration": "https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip",
    "servo": "https://archive.ics.uci.edu/static/public/87/servo.zip",
}


def ensure_downloaded(name: str) -> Path:
    raw_dir = EXTERNAL / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        return raw_dir
    # Support manually downloaded UCI files as well as archives fetched by this
    # script. This also avoids a network request in an existing experiment tree.
    if any(path.is_file() and not path.name.startswith(".") for path in raw_dir.rglob("*")):
        return raw_dir

    zip_path = EXTERNAL / name / f"uci_{name}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"[DOWNLOAD] {name}: {DOWNLOADS[name]}")
        urlretrieve(DOWNLOADS[name], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return raw_dir


def _find_first(raw_dir: Path, suffixes: tuple[str, ...]) -> Path:
    for path in sorted(raw_dir.rglob("*")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if path.is_file() and path.name.lower().endswith(suffixes):
            return path
    raise FileNotFoundError(f"No file with suffix {suffixes} under {raw_dir}")


def _one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(series.astype(str), prefix=prefix, dtype=int)


def _time_columns(date_series: pd.Series | None = None, time_series: pd.Series | None = None) -> dict[str, pd.Series]:
    if date_series is not None and time_series is not None:
        ts = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str), errors="coerce")
    elif date_series is not None:
        ts = pd.to_datetime(date_series.astype(str), errors="coerce")
    elif time_series is not None:
        ts = pd.to_datetime(time_series.astype(str), errors="coerce")
    else:
        raise ValueError("date_series or time_series is required")
    hour = ts.dt.hour.fillna(0).astype(float)
    minute = ts.dt.minute.fillna(0).astype(float)
    dayofweek = ts.dt.dayofweek.fillna(0).astype(float)
    return {
        "feature_hour_sin": np.sin(2 * np.pi * (hour + minute / 60.0) / 24.0),
        "feature_hour_cos": np.cos(2 * np.pi * (hour + minute / 60.0) / 24.0),
        "feature_dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7.0),
        "feature_dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7.0),
    }


def prepare_concrete_compressive_strength() -> Path:
    raw_dir = ensure_downloaded("concrete_compressive_strength")
    raw_path = _find_first(raw_dir, (".xls", ".xlsx"))
    df = pd.read_excel(raw_path)
    df.columns = [str(c).strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "feature_cement": df.iloc[:, 0],
            "feature_blast_furnace_slag": df.iloc[:, 1],
            "feature_fly_ash": df.iloc[:, 2],
            "feature_water": df.iloc[:, 3],
            "feature_superplasticizer": df.iloc[:, 4],
            "feature_coarse_aggregate": df.iloc[:, 5],
            "feature_fine_aggregate": df.iloc[:, 6],
            "intermediate_curing_age_days": df.iloc[:, 7],
            "target_compressive_strength": df.iloc[:, 8],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "concrete_compressive_strength"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concrete_compressive_strength_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_seoul_bike_sharing() -> Path:
    raw_dir = ensure_downloaded("seoul_bike_sharing")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path, encoding="latin1")
    date = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    out = pd.DataFrame(
        {
            "feature_hour": df["Hour"],
            "feature_dayofweek": date.dt.dayofweek,
            "feature_month": date.dt.month,
            "feature_season": df["Seasons"],
            "feature_holiday": df["Holiday"],
            "feature_functioning_day": df["Functioning Day"],
            "intermediate_temperature": df["Temperature(°C)"],
            "intermediate_humidity": df["Humidity(%)"],
            "intermediate_wind_speed": df["Wind speed (m/s)"],
            "intermediate_visibility": df["Visibility (10m)"],
            "intermediate_dew_point_temperature": df["Dew point temperature(°C)"],
            "intermediate_solar_radiation": df["Solar Radiation (MJ/m2)"],
            "intermediate_rainfall": df["Rainfall(mm)"],
            "intermediate_snowfall": df["Snowfall (cm)"],
            "target_rented_bike_count": df["Rented Bike Count"],
            "ID_date": df["Date"],
            "split": "train",
        }
    )
    valid_date = date.notna()
    out = out[valid_date].copy()
    cutoff = int(len(out) * 0.7)
    out.loc[out.index[cutoff:], "split"] = "test"
    out_dir = PROCESSED / "seoul_bike_sharing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seoul_bike_sharing_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_electrical_grid_stability() -> Path:
    raw_dir = ensure_downloaded("electrical_grid_stability")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(
        {
            "feature_reaction_time_producer": df["tau1"],
            "feature_nominal_power_producer": df["p1"],
            "feature_price_elasticity_producer": df["g1"],
            "intermediate_reaction_time_consumer_1": df["tau2"],
            "intermediate_reaction_time_consumer_2": df["tau3"],
            "intermediate_reaction_time_consumer_3": df["tau4"],
            "intermediate_nominal_power_consumer_1": df["p2"],
            "intermediate_nominal_power_consumer_2": df["p3"],
            "intermediate_nominal_power_consumer_3": df["p4"],
            "intermediate_price_elasticity_consumer_1": df["g2"],
            "intermediate_price_elasticity_consumer_2": df["g3"],
            "intermediate_price_elasticity_consumer_3": df["g4"],
            "target_stability": df["stab"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "electrical_grid_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "electrical_grid_stability_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_ai4i_predictive_maintenance() -> Path:
    raw_dir = ensure_downloaded("ai4i_predictive_maintenance")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(
        {
            "feature_product_type": df["Type"],
            "feature_air_temperature": df["Air temperature [K]"],
            "feature_process_temperature": df["Process temperature [K]"],
            "intermediate_rotational_speed": df["Rotational speed [rpm]"],
            "intermediate_torque": df["Torque [Nm]"],
            "intermediate_tool_wear": df["Tool wear [min]"],
            "target_machine_failure": df["Machine failure"],
            "target_tool_wear_failure": df["TWF"],
            "target_heat_dissipation_failure": df["HDF"],
            "target_power_failure": df["PWF"],
            "target_overstrain_failure": df["OSF"],
            "target_random_failure": df["RNF"],
            "ID_uid": df["UDI"],
        }
    )
    out_dir = PROCESSED / "ai4i_predictive_maintenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ai4i_predictive_maintenance_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_steel_industry_energy() -> Path:
    raw_dir = ensure_downloaded("steel_industry_energy")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    timestamp = pd.to_datetime(df["date"], errors="coerce")
    out = pd.DataFrame(
        {
            "feature_nsm": df["NSM"],
            "feature_hour": timestamp.dt.hour.fillna(0),
            "feature_month": timestamp.dt.month.fillna(0),
            "feature_week_status": df["WeekStatus"],
            "feature_day_of_week": df["Day_of_week"],
            "intermediate_lagging_reactive_power": df["Lagging_Current_Reactive.Power_kVarh"],
            "intermediate_leading_reactive_power": df["Leading_Current_Reactive_Power_kVarh"],
            "intermediate_lagging_power_factor": df["Lagging_Current_Power_Factor"],
            "intermediate_leading_power_factor": df["Leading_Current_Power_Factor"],
            "target_usage_kwh": df["Usage_kWh"],
            "ID_timestamp": df["date"],
            "split": "train",
        }
    )
    cutoff = int(len(out) * 0.7)
    out.loc[out.index[cutoff:], "split"] = "test"
    out_dir = PROCESSED / "steel_industry_energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steel_industry_energy_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_steel_plates_faults() -> Path:
    raw_dir = ensure_downloaded("steel_plates_faults")
    raw_path = _find_first(raw_dir, (".nna", ".data", ".txt"))
    cols = [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum",
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Length_of_Conveyer",
        "TypeOfSteel_A300",
        "TypeOfSteel_A400",
        "Steel_Plate_Thickness",
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
        "Orientation_Index",
        "Luminosity_Index",
        "SigmoidOfAreas",
        "target_Pastry",
        "target_Z_Scratch",
        "target_K_Scatch",
        "target_Stains",
        "target_Dirtiness",
        "target_Bumps",
        "target_Other_Faults",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=cols)
    out = pd.DataFrame(
        {
            "feature_x_minimum": df["X_Minimum"],
            "feature_x_maximum": df["X_Maximum"],
            "feature_y_minimum": df["Y_Minimum"],
            "feature_y_maximum": df["Y_Maximum"],
            "feature_length_of_conveyer": df["Length_of_Conveyer"],
            "feature_type_of_steel_a300": df["TypeOfSteel_A300"],
            "feature_type_of_steel_a400": df["TypeOfSteel_A400"],
            "feature_steel_plate_thickness": df["Steel_Plate_Thickness"],
            "intermediate_pixels_areas": df["Pixels_Areas"],
            "intermediate_x_perimeter": df["X_Perimeter"],
            "intermediate_y_perimeter": df["Y_Perimeter"],
            "intermediate_sum_of_luminosity": df["Sum_of_Luminosity"],
            "intermediate_minimum_of_luminosity": df["Minimum_of_Luminosity"],
            "intermediate_maximum_of_luminosity": df["Maximum_of_Luminosity"],
            "intermediate_edges_index": df["Edges_Index"],
            "intermediate_empty_index": df["Empty_Index"],
            "intermediate_square_index": df["Square_Index"],
            "intermediate_outside_x_index": df["Outside_X_Index"],
            "intermediate_edges_x_index": df["Edges_X_Index"],
            "intermediate_edges_y_index": df["Edges_Y_Index"],
            "intermediate_outside_global_index": df["Outside_Global_Index"],
            "intermediate_log_of_areas": df["LogOfAreas"],
            "intermediate_log_x_index": df["Log_X_Index"],
            "intermediate_log_y_index": df["Log_Y_Index"],
            "intermediate_orientation_index": df["Orientation_Index"],
            "intermediate_luminosity_index": df["Luminosity_Index"],
            "intermediate_sigmoid_of_areas": df["SigmoidOfAreas"],
            "target_pastry": df["target_Pastry"],
            "target_z_scratch": df["target_Z_Scratch"],
            "target_k_scatch": df["target_K_Scatch"],
            "target_stains": df["target_Stains"],
            "target_dirtiness": df["target_Dirtiness"],
            "target_bumps": df["target_Bumps"],
            "target_other_faults": df["target_Other_Faults"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "steel_plates_faults"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steel_plates_faults_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_sensorless_drive_diagnosis() -> Path:
    raw_dir = ensure_downloaded("sensorless_drive_diagnosis")
    raw_path = _find_first(raw_dir, (".txt",))
    df = pd.read_csv(raw_path, sep=r"\s+", header=None)
    feature_cols = [f"feature_sensorless_{i:02d}" for i in range(16)]
    acp_cols = [f"intermediate_sensorless_{i:02d}" for i in range(16, 48)]
    out = pd.DataFrame(df.iloc[:, :48].to_numpy(), columns=feature_cols + acp_cols)
    out = pd.concat([out, _one_hot(df.iloc[:, 48], "target_drive_class")], axis=1)
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "sensorless_drive_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensorless_drive_diagnosis_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_robot_execution_failures() -> Path:
    raw_dir = ensure_downloaded("robot_execution_failures")
    records = []
    for path in sorted(raw_dir.glob("lp*.data")):
        lines = [line.rstrip() for line in path.read_text(encoding="latin1").splitlines()]
        label = None
        values: list[list[float]] = []
        for line in lines + [""]:
            stripped = line.strip()
            if not stripped:
                if label is not None and values:
                    arr = np.asarray(values, dtype=float)
                    rec = {"label": label, "ID_source": path.stem}
                    for i, value in enumerate(arr[0]):
                        rec[f"feature_initial_ft_{i}"] = value
                    for i, value in enumerate(arr.mean(axis=0)):
                        rec[f"intermediate_mean_ft_{i}"] = value
                    for i, value in enumerate(arr.std(axis=0)):
                        rec[f"intermediate_std_ft_{i}"] = value
                    records.append(rec)
                label = None
                values = []
                continue
            parts = stripped.split()
            if len(parts) == 1 and not parts[0].lstrip("-").replace(".", "", 1).isdigit():
                label = parts[0]
            else:
                values.append([float(x) for x in parts])
    df = pd.DataFrame(records)
    out = pd.concat([df.drop(columns=["label"]), _one_hot(df["label"], "target_failure")], axis=1)
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "robot_execution_failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "robot_execution_failures_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_wall_following_robot() -> Path:
    raw_dir = ensure_downloaded("wall_following_robot")
    raw_path = raw_dir / "sensor_readings_24.data"
    df = pd.read_csv(raw_path, header=None)
    out = pd.DataFrame()
    for i in range(8):
        out[f"feature_ultrasonic_{i:02d}"] = df.iloc[:, i]
    for i in range(8, 24):
        out[f"intermediate_ultrasonic_{i:02d}"] = df.iloc[:, i]
    out = pd.concat([out, _one_hot(df.iloc[:, 24], "target_navigation")], axis=1)
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "wall_following_robot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wall_following_robot_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_occupancy_detection() -> Path:
    raw_dir = ensure_downloaded("occupancy_detection")
    frames = []
    for name, split in [("datatraining.txt", "train"), ("datatest.txt", "test"), ("datatest2.txt", "test")]:
        df = pd.read_csv(raw_dir / name)
        time_cols = _time_columns(date_series=df["date"])
        out = pd.DataFrame(time_cols)
        out["intermediate_temperature"] = df["Temperature"]
        out["intermediate_humidity"] = df["Humidity"]
        out["intermediate_light"] = df["Light"]
        out["intermediate_co2"] = df["CO2"]
        out["intermediate_humidity_ratio"] = df["HumidityRatio"]
        out["target_occupancy"] = df["Occupancy"]
        out["ID_source"] = name
        out["split"] = split
        frames.append(out)
    merged = pd.concat(frames, ignore_index=True)
    out_dir = PROCESSED / "occupancy_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "occupancy_detection_prism.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def prepare_room_occupancy_estimation() -> Path:
    raw_dir = ensure_downloaded("room_occupancy_estimation")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(_time_columns(date_series=df["Date"], time_series=df["Time"]))
    for col in [c for c in df.columns if c.startswith("S")]:
        out[f"intermediate_{col.lower()}"] = df[col]
    out["target_room_occupancy_count"] = df["Room_Occupancy_Count"]
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "room_occupancy_estimation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "room_occupancy_estimation_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_sgemm_gpu_kernel_performance() -> Path:
    raw_dir = ensure_downloaded("sgemm_gpu_kernel_performance")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    feature_cols = ["MWG", "NWG", "KWG", "MDIMC", "NDIMC", "KWI"]
    acp_cols = ["MDIMA", "NDIMB", "VWM", "VWN", "STRM", "STRN", "SA", "SB"]
    out = pd.DataFrame()
    for col in feature_cols:
        out[f"feature_{col.lower()}"] = df[col]
    for col in acp_cols:
        out[f"intermediate_{col.lower()}"] = df[col]
    run_cols = ["Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"]
    out["target_runtime_ms"] = df[run_cols].mean(axis=1)
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "sgemm_gpu_kernel_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sgemm_gpu_kernel_performance_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_average_localization_error() -> Path:
    raw_dir = ensure_downloaded("average_localization_error")
    raw_path = _find_first(raw_dir, (".csv",))
    df = pd.read_csv(raw_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    out = pd.DataFrame(
        {
            "feature_anchor_ratio": df.iloc[:, 0],
            "feature_transmission_range": df.iloc[:, 1],
            "feature_node_density": df.iloc[:, 2],
            "intermediate_iterations": df.iloc[:, 3],
            "target_average_localization_error": df.iloc[:, 4],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "average_localization_error"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "average_localization_error_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_gas_sensor_flow_modulation() -> Path:
    raw_dir = ensure_downloaded("gas_sensor_flow_modulation")
    raw_path = raw_dir / "features.csv"
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(
        {
            "feature_exp": df["exp"],
            "feature_batch": df["batch"],
            "feature_lab": df["lab"],
            "feature_col": df["col"],
            "target_acetone_concentration": df["ace_conc"],
            "target_ethanol_concentration": df["eth_conc"],
            "ID_row": range(len(df)),
        }
    )
    sensor_cols = [c for c in df.columns if c.startswith("S")]
    for col in sensor_cols:
        out[f"intermediate_{col.lower()}"] = df[col]
    out_dir = PROCESSED / "gas_sensor_flow_modulation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gas_sensor_flow_modulation_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_gas_sensor_drift_concentration() -> Path:
    raw_dir = ensure_downloaded("gas_sensor_drift_concentration")
    rows = []
    for path in sorted(raw_dir.glob("batch*.dat"), key=lambda p: int(p.stem.replace("batch", ""))):
        batch = int(path.stem.replace("batch", ""))
        for line in path.read_text(encoding="latin1").splitlines():
            if not line.strip():
                continue
            head, *pairs = line.split()
            gas_class, concentration = head.split(";")
            rec = {
                "feature_batch": batch,
                "target_concentration": float(concentration),
                "target_gas_class": gas_class,
            }
            for pair in pairs:
                idx, value = pair.split(":")
                idx_int = int(idx)
                prefix = "feature" if idx_int <= 32 else "intermediate"
                rec[f"{prefix}_gas_drift_{idx_int:03d}"] = float(value)
            rows.append(rec)
    df = pd.DataFrame(rows).fillna(0.0)
    target_class = _one_hot(df.pop("target_gas_class"), "target_gas_class")
    out = pd.concat([df, target_class], axis=1)
    out["ID_row"] = range(len(out))
    out_dir = PROCESSED / "gas_sensor_drift_concentration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gas_sensor_drift_concentration_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_servo() -> Path:
    raw_dir = ensure_downloaded("servo")
    raw_path = raw_dir / "servo.data"
    df = pd.read_csv(raw_path, header=None, names=["motor", "screw", "pgain", "vgain", "class"])
    out = pd.concat(
        [
            _one_hot(df["motor"], "feature_motor"),
            _one_hot(df["screw"], "feature_screw"),
            pd.DataFrame(
                {
                    "intermediate_pgain": df["pgain"],
                    "intermediate_vgain": df["vgain"],
                    "target_rise_time": df["class"],
                    "ID_row": range(len(df)),
                }
            ),
        ],
        axis=1,
    )
    out_dir = PROCESSED / "servo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "servo_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_gas_turbine() -> Path:
    raw_dir = EXTERNAL / "gas_turbine_co_nox" / "raw"
    frames = []
    for csv_path in sorted(raw_dir.glob("gt_*.csv")):
        year = int(csv_path.stem.split("_")[1])
        df = pd.read_csv(csv_path)
        out = pd.DataFrame(
            {
                "feature_ambient_temperature": df["AT"],
                "feature_ambient_pressure": df["AP"],
                "feature_ambient_humidity": df["AH"],
                "feature_air_filter_diff_pressure": df["AFDP"],
                "intermediate_gt_exhaust_pressure": df["GTEP"],
                "intermediate_turbine_inlet_temperature": df["TIT"],
                "intermediate_turbine_after_temperature": df["TAT"],
                "intermediate_turbine_energy_yield": df["TEY"],
                "intermediate_compressor_discharge_pressure": df["CDP"],
                "target_co": df["CO"],
                "target_nox": df["NOX"],
                "ID_year": year,
                "ID_row": range(len(df)),
                "split": "train" if year <= 2013 else "test",
            }
        )
        frames.append(out)

    merged = pd.concat(frames, ignore_index=True)
    out_dir = PROCESSED / "gas_turbine_co_nox"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gas_turbine_co_nox_prism.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def prepare_naval_propulsion() -> Path:
    raw_path = EXTERNAL / "naval_propulsion" / "raw" / "UCI CBM Dataset" / "data.txt"
    cols = [
        "lp",
        "v",
        "GTT",
        "GTn",
        "GGn",
        "Ts",
        "Tp",
        "T48",
        "T1",
        "T2",
        "P48",
        "P1",
        "P2",
        "Pexh",
        "TIC",
        "mf",
        "kMc",
        "kMt",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=cols)
    out = pd.DataFrame(
        {
            "feature_lever_position": df["lp"],
            "feature_ship_speed": df["v"],
            "intermediate_gt_shaft_torque": df["GTT"],
            "intermediate_gt_rate_revolutions": df["GTn"],
            "intermediate_gas_generator_rate_revolutions": df["GGn"],
            "intermediate_starboard_propeller_torque": df["Ts"],
            "intermediate_port_propeller_torque": df["Tp"],
            "intermediate_hp_turbine_exit_temperature": df["T48"],
            "intermediate_compressor_inlet_air_temperature": df["T1"],
            "intermediate_compressor_outlet_air_temperature": df["T2"],
            "intermediate_hp_turbine_exit_pressure": df["P48"],
            "intermediate_compressor_inlet_air_pressure": df["P1"],
            "intermediate_compressor_outlet_air_pressure": df["P2"],
            "intermediate_gt_exhaust_gas_pressure": df["Pexh"],
            "intermediate_turbine_injection_control": df["TIC"],
            "intermediate_fuel_flow": df["mf"],
            "target_compressor_decay": df["kMc"],
            "target_turbine_decay": df["kMt"],
            "ID_speed": df["v"],
        }
    )
    out_dir = PROCESSED / "naval_propulsion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "naval_propulsion_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_air_quality() -> Path:
    raw_dir = ensure_downloaded("air_quality")
    raw_path = _find_first(raw_dir, ("airqualityuci.csv",))
    df = pd.read_csv(raw_path, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all")
    df = df.replace(-200, pd.NA)
    timestamp = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str).str.replace(".", ":", regex=False),
        dayfirst=True,
        errors="coerce",
    )
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    dayofweek = timestamp.dt.dayofweek
    out = pd.DataFrame(
        {
            "feature_temperature": df["T"],
            "feature_relative_humidity": df["RH"],
            "feature_absolute_humidity": df["AH"],
            "feature_hour_sin": (2 * np.pi * hour / 24.0).map(np.sin),
            "feature_hour_cos": (2 * np.pi * hour / 24.0).map(np.cos),
            "feature_dayofweek_sin": (2 * np.pi * dayofweek / 7.0).map(np.sin),
            "feature_dayofweek_cos": (2 * np.pi * dayofweek / 7.0).map(np.cos),
            "intermediate_sensor_co": df["PT08.S1(CO)"],
            "intermediate_sensor_nmhc": df["PT08.S2(NMHC)"],
            "intermediate_sensor_nox": df["PT08.S3(NOx)"],
            "intermediate_sensor_no2": df["PT08.S4(NO2)"],
            "intermediate_sensor_o3": df["PT08.S5(O3)"],
            "target_co": df["CO(GT)"],
            "target_benzene": df["C6H6(GT)"],
            "target_nox": df["NOx(GT)"],
            "target_no2": df["NO2(GT)"],
            "ID_timestamp": timestamp,
            "split": "train",
        }
    )
    valid_time = out["ID_timestamp"].notna()
    out = out[valid_time].copy()
    cutoff = int(len(out) * 0.7)
    out.loc[out.index[cutoff:], "split"] = "test"
    out = out.drop(columns=["ID_timestamp"])
    out_dir = PROCESSED / "air_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "air_quality_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_appliances_energy() -> Path:
    raw_path = EXTERNAL / "appliances_energy" / "raw" / "energydata_complete.csv"
    df = pd.read_csv(raw_path)
    timestamp = pd.to_datetime(df["date"], errors="coerce")
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    dayofweek = timestamp.dt.dayofweek
    out = pd.DataFrame(
        {
            "feature_outdoor_temperature": df["T_out"],
            "feature_pressure": df["Press_mm_hg"],
            "feature_outdoor_relative_humidity": df["RH_out"],
            "feature_windspeed": df["Windspeed"],
            "feature_visibility": df["Visibility"],
            "feature_dewpoint": df["Tdewpoint"],
            "feature_hour_sin": (2 * np.pi * hour / 24.0).map(np.sin),
            "feature_hour_cos": (2 * np.pi * hour / 24.0).map(np.cos),
            "feature_dayofweek_sin": (2 * np.pi * dayofweek / 7.0).map(np.sin),
            "feature_dayofweek_cos": (2 * np.pi * dayofweek / 7.0).map(np.cos),
            "intermediate_lights": df["lights"],
            "target_appliances": df["Appliances"],
            "ID_timestamp": timestamp,
            "split": "train",
        }
    )
    for i in range(1, 10):
        out[f"intermediate_room_temperature_{i}"] = df[f"T{i}"]
        out[f"intermediate_room_humidity_{i}"] = df[f"RH_{i}"]
    valid_time = out["ID_timestamp"].notna()
    out = out[valid_time].copy()
    cutoff = int(len(out) * 0.7)
    out.loc[out.index[cutoff:], "split"] = "test"
    out = out.drop(columns=["ID_timestamp"])
    out_dir = PROCESSED / "appliances_energy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "appliances_energy_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_concrete_slump_test() -> Path:
    raw_dir = ensure_downloaded("concrete_slump_test")
    raw_path = _find_first(raw_dir, ("slump_test.data",))
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(
        {
            "feature_cement": df["Cement"],
            "feature_slag": df["Slag"],
            "feature_fly_ash": df["Fly ash"],
            "feature_water": df["Water"],
            "feature_superplasticizer": df["SP"],
            "feature_coarse_aggregate": df["Coarse Aggr."],
            "feature_fine_aggregate": df["Fine Aggr."],
            "intermediate_slump": df["SLUMP(cm)"],
            "intermediate_flow": df["FLOW(cm)"],
            "target_compressive_strength_28d": df["Compressive Strength (28-day)(Mpa)"],
            "ID_sample": df["No"],
        }
    )
    out_dir = PROCESSED / "concrete_slump_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concrete_slump_test_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_student_performance() -> Path:
    raw_dir = ensure_downloaded("student_performance")
    frames = []
    for subject, filename in [("mathematics", "student-mat.csv"), ("portuguese", "student-por.csv")]:
        raw_path = next(raw_dir.rglob(filename))
        df = pd.read_csv(raw_path, sep=";")
        out = df.drop(columns=["G1", "G2", "G3"]).copy()
        out.columns = [f"feature_{col.lower()}" for col in out.columns]
        out["feature_subject"] = subject
        out["intermediate_grade_period_1"] = df["G1"]
        out["intermediate_grade_period_2"] = df["G2"]
        out["target_final_grade"] = df["G3"]
        out["ID_subject"] = subject
        out["ID_row"] = range(len(df))
        frames.append(out)

    merged = pd.concat(frames, ignore_index=True)
    out_dir = PROCESSED / "student_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "student_performance_prism.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def prepare_parkinsons_telemonitoring() -> Path:
    raw_dir = ensure_downloaded("parkinsons_telemonitoring")
    raw_path = _find_first(raw_dir, ("parkinsons_updrs.data",))
    df = pd.read_csv(raw_path)
    voice_cols = [
        "Jitter(%)",
        "Jitter(Abs)",
        "Jitter:RAP",
        "Jitter:PPQ5",
        "Jitter:DDP",
        "Shimmer",
        "Shimmer(dB)",
        "Shimmer:APQ3",
        "Shimmer:APQ5",
        "Shimmer:APQ11",
        "Shimmer:DDA",
        "NHR",
        "HNR",
        "RPDE",
        "DFA",
        "PPE",
    ]
    out = pd.DataFrame(
        {
            "feature_age": df["age"],
            "feature_sex": df["sex"],
            "feature_test_time": df["test_time"],
            "intermediate_motor_updrs": df["motor_UPDRS"],
            "target_total_updrs": df["total_UPDRS"],
            "ID_subject": df["subject#"],
        }
    )
    for col in voice_cols:
        safe = col.lower().replace("(", "").replace(")", "").replace("%", "pct").replace(":", "_")
        out[f"feature_voice_{safe}"] = df[col]

    out_dir = PROCESSED / "parkinsons_telemonitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parkinsons_telemonitoring_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_energy_efficiency() -> Path:
    raw_dir = ensure_downloaded("energy_efficiency")
    raw_path = _find_first(raw_dir, (".xlsx", ".xls"))
    df = pd.read_excel(raw_path)
    out = pd.DataFrame(
        {
            "feature_relative_compactness": df["X1"],
            "feature_surface_area": df["X2"],
            "feature_wall_area": df["X3"],
            "feature_roof_area": df["X4"],
            "feature_overall_height": df["X5"],
            "feature_orientation": df["X6"],
            "feature_glazing_area": df["X7"],
            "feature_glazing_area_distribution": df["X8"],
            "intermediate_heating_load": df["Y1"],
            "target_cooling_load": df["Y2"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "energy_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "energy_efficiency_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _sensor_summary(df: pd.DataFrame, prefix: str, role: str, cols: slice | None = None) -> pd.DataFrame:
    arr = df.iloc[:, cols].to_numpy(dtype=np.float32) if cols is not None else df.to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            f"{role}_{prefix}_mean": arr.mean(axis=1),
            f"{role}_{prefix}_std": arr.std(axis=1),
            f"{role}_{prefix}_min": arr.min(axis=1),
            f"{role}_{prefix}_max": arr.max(axis=1),
        }
    )


def prepare_hydraulic_systems() -> Path:
    extracted_dir = ensure_downloaded("hydraulic_systems")
    raw_dir = next(extracted_dir.rglob("profile.txt")).parent
    profile = pd.read_csv(raw_dir / "profile.txt", sep=r"\s+", header=None)
    profile.columns = [
        "target_cooler_condition",
        "target_valve_condition",
        "target_internal_pump_leakage",
        "target_hydraulic_accumulator",
        "ID_stable_flag",
    ]

    feature_sensors = ["PS1", "PS2", "FS1", "TS1", "VS1", "EPS1"]
    acp_sensors = ["PS3", "PS4", "PS5", "PS6", "FS2", "TS2", "TS3", "TS4", "CE", "CP", "SE"]
    parts = [profile]
    for sensor in feature_sensors:
        df = pd.read_csv(raw_dir / f"{sensor}.txt", sep=r"\s+", header=None)
        half = max(1, df.shape[1] // 2)
        parts.append(_sensor_summary(df, sensor.lower(), "feature", slice(0, half)))
    for sensor in acp_sensors:
        df = pd.read_csv(raw_dir / f"{sensor}.txt", sep=r"\s+", header=None)
        half = max(1, df.shape[1] // 2)
        parts.append(_sensor_summary(df, sensor.lower(), "intermediate", slice(half, None)))

    out = pd.concat(parts, axis=1)
    out["ID_cycle"] = range(len(out))
    out_dir = PROCESSED / "hydraulic_systems"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hydraulic_systems_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_household_power_consumption() -> Path:
    raw_dir = ensure_downloaded("household_power_consumption")
    raw_path = _find_first(raw_dir, ("household_power_consumption.txt",))
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    hourly_sums: list[pd.DataFrame] = []
    hourly_counts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(raw_path, sep=";", na_values=["?"], chunksize=200_000):
        timestamp = pd.to_datetime(
            chunk["Date"].astype(str) + " " + chunk["Time"].astype(str),
            dayfirst=True,
            errors="coerce",
        )
        values = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        values.index = timestamp.dt.floor("h")
        values = values[values.index.notna()].dropna()
        hourly_sums.append(values.groupby(level=0).sum())
        hourly_counts.append(values.groupby(level=0).count())

    total = pd.concat(hourly_sums).groupby(level=0).sum()
    count = pd.concat(hourly_counts).groupby(level=0).sum()
    hourly = total.div(count).dropna().rename_axis("timestamp").reset_index()
    hour = hourly["timestamp"].dt.hour
    dayofweek = hourly["timestamp"].dt.dayofweek
    month = hourly["timestamp"].dt.month
    out = pd.DataFrame(
        {
            "feature_global_active_power": hourly["Global_active_power"],
            "feature_voltage": hourly["Voltage"],
            "feature_global_intensity": hourly["Global_intensity"],
            "feature_hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "feature_hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "feature_dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7.0),
            "feature_dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7.0),
            "feature_month_sin": np.sin(2 * np.pi * month / 12.0),
            "feature_month_cos": np.cos(2 * np.pi * month / 12.0),
            "intermediate_sub_metering_1": hourly["Sub_metering_1"],
            "intermediate_sub_metering_2": hourly["Sub_metering_2"],
            "intermediate_sub_metering_3": hourly["Sub_metering_3"],
            "target_global_reactive_power": hourly["Global_reactive_power"],
            "split": "train",
        }
    )
    cutoff = int(len(out) * 0.7)
    out.loc[out.index[cutoff:], "split"] = "test"
    out_dir = PROCESSED / "household_power_consumption"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "household_power_consumption_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_bike_sharing() -> Path:
    raw_path = EXTERNAL / "bike_sharing" / "raw" / "hour.csv"
    df = pd.read_csv(raw_path)
    out = pd.DataFrame(
        {
            "feature_season": df["season"],
            "feature_year": df["yr"],
            "feature_month": df["mnth"],
            "feature_hour": df["hr"],
            "feature_holiday": df["holiday"],
            "feature_weekday": df["weekday"],
            "feature_workingday": df["workingday"],
            "feature_weather_situation": df["weathersit"],
            "feature_temperature": df["temp"],
            "feature_feels_like_temperature": df["atemp"],
            "feature_humidity": df["hum"],
            "feature_windspeed": df["windspeed"],
            "intermediate_casual_users": df["casual"],
            "target_registered_users": df["registered"],
            "ID_date": df["dteday"],
            "ID_hour": df["hr"],
        }
    )
    out_dir = PROCESSED / "bike_sharing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bike_sharing_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_wine_quality() -> Path:
    raw_dir = ensure_downloaded("wine_quality")
    frames = []
    for wine_type, filename in [("red", "winequality-red.csv"), ("white", "winequality-white.csv")]:
        df = pd.read_csv(next(raw_dir.rglob(filename)), sep=";")
        out = pd.DataFrame(
            {
                "feature_fixed_acidity": df["fixed acidity"],
                "feature_volatile_acidity": df["volatile acidity"],
                "feature_citric_acid": df["citric acid"],
                "feature_residual_sugar": df["residual sugar"],
                "feature_chlorides": df["chlorides"],
                "feature_free_sulfur_dioxide": df["free sulfur dioxide"],
                "feature_total_sulfur_dioxide": df["total sulfur dioxide"],
                "feature_sulphates": df["sulphates"],
                "feature_wine_type": wine_type,
                "intermediate_density": df["density"],
                "intermediate_ph": df["pH"],
                "intermediate_alcohol": df["alcohol"],
                "target_quality": df["quality"],
                "ID_wine_type": wine_type,
                "ID_row": range(len(df)),
            }
        )
        frames.append(out)
    merged = pd.concat(frames, ignore_index=True)
    out_dir = PROCESSED / "wine_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wine_quality_prism.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def prepare_superconductivity() -> Path:
    raw_dir = ensure_downloaded("superconductivity")
    descriptors = pd.read_csv(next(raw_dir.rglob("train.csv")))
    composition = pd.read_csv(next(raw_dir.rglob("unique_m.csv"))).drop(columns=["critical_temp", "material"])
    out = pd.DataFrame(
        {
            **{f"feature_element_{col}": composition[col] for col in composition.columns},
            **{
                f"intermediate_descriptor_{col}": descriptors[col]
                for col in descriptors.columns
                if col != "critical_temp"
            },
            "target_critical_temperature": descriptors["critical_temp"],
            "ID_row": range(len(descriptors)),
        }
    )
    out_dir = PROCESSED / "superconductivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "superconductivity_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_auto_mpg() -> Path:
    raw_dir = ensure_downloaded("auto_mpg")
    raw_path = _find_first(raw_dir, ("auto-mpg.data",))
    cols = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, na_values="?", names=cols)
    out = pd.DataFrame(
        {
            "feature_cylinders": df["cylinders"],
            "feature_displacement": df["displacement"],
            "feature_weight": df["weight"],
            "feature_model_year": df["model_year"],
            "feature_origin": df["origin"],
            "intermediate_horsepower": df["horsepower"],
            "intermediate_acceleration": df["acceleration"],
            "target_mpg": df["mpg"],
            "ID_car_name": df["car_name"],
        }
    )
    out_dir = PROCESSED / "auto_mpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "auto_mpg_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_airfoil_self_noise() -> Path:
    raw_dir = ensure_downloaded("airfoil_self_noise")
    raw_path = _find_first(raw_dir, ("airfoil_self_noise.dat",))
    cols = [
        "frequency",
        "angle_of_attack",
        "chord_length",
        "free_stream_velocity",
        "suction_side_displacement_thickness",
        "scaled_sound_pressure_level",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=cols)
    out = pd.DataFrame(
        {
            "feature_frequency": df["frequency"],
            "feature_angle_of_attack": df["angle_of_attack"],
            "feature_chord_length": df["chord_length"],
            "feature_free_stream_velocity": df["free_stream_velocity"],
            "intermediate_suction_side_displacement_thickness": df["suction_side_displacement_thickness"],
            "target_scaled_sound_pressure_level": df["scaled_sound_pressure_level"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "airfoil_self_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "airfoil_self_noise_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_yacht_hydrodynamics() -> Path:
    raw_path = EXTERNAL / "yacht_hydrodynamics" / "raw" / "yacht_hydrodynamics.data"
    cols = [
        "longitudinal_position",
        "prismatic_coefficient",
        "length_displacement_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
        "residuary_resistance",
    ]
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=cols)
    out = pd.DataFrame(
        {
            "feature_longitudinal_position": df["longitudinal_position"],
            "feature_prismatic_coefficient": df["prismatic_coefficient"],
            "feature_length_displacement_ratio": df["length_displacement_ratio"],
            "feature_beam_draught_ratio": df["beam_draught_ratio"],
            "feature_froude_number": df["froude_number"],
            "intermediate_length_beam_ratio": df["length_beam_ratio"],
            "target_residuary_resistance": df["residuary_resistance"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "yacht_hydrodynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yacht_hydrodynamics_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_combined_cycle_power_plant() -> Path:
    raw_dir = ensure_downloaded("combined_cycle_power_plant")
    raw_path = _find_first(raw_dir, ("folds5x2_pp.xlsx",))
    df = pd.read_excel(raw_path)
    out = pd.DataFrame(
        {
            "feature_ambient_temperature": df["AT"],
            "feature_exhaust_vacuum": df["V"],
            "feature_relative_humidity": df["RH"],
            "intermediate_ambient_pressure": df["AP"],
            "target_net_hourly_electrical_energy": df["PE"],
            "ID_row": range(len(df)),
        }
    )
    out_dir = PROCESSED / "combined_cycle_power_plant"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_cycle_power_plant_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


def prepare_real_estate_valuation() -> Path:
    raw_dir = ensure_downloaded("real_estate_valuation")
    raw_path = _find_first(raw_dir, (".xlsx", ".xls"))
    df = pd.read_excel(raw_path)
    out = pd.DataFrame(
        {
            "feature_transaction_date": df["X1 transaction date"],
            "feature_house_age": df["X2 house age"],
            "intermediate_distance_to_mrt": df["X3 distance to the nearest MRT station"],
            "intermediate_convenience_stores": df["X4 number of convenience stores"],
            "intermediate_latitude": df["X5 latitude"],
            "intermediate_longitude": df["X6 longitude"],
            "target_house_price_unit_area": df["Y house price of unit area"],
            "ID_row": df["No"],
        }
    )
    out_dir = PROCESSED / "real_estate_valuation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_estate_valuation_prism.csv"
    out.to_csv(out_path, index=False)
    return out_path


PAPER_GROUPS = {
    "main": [
        "airfoil_self_noise",
        "energy_efficiency",
        "hydraulic_systems",
        "household_power_consumption",
        "concrete_slump_test",
    ],
    "case1": [
        "wine_quality",
        "superconductivity",
        "auto_mpg",
        "combined_cycle_power_plant",
    ],
    "case2": [
        "air_quality",
        "parkinsons_telemonitoring",
        "real_estate_valuation",
        "student_performance",
    ],
}

PREPARERS = {
    "airfoil_self_noise": prepare_airfoil_self_noise,
    "energy_efficiency": prepare_energy_efficiency,
    "hydraulic_systems": prepare_hydraulic_systems,
    "household_power_consumption": prepare_household_power_consumption,
    "concrete_slump_test": prepare_concrete_slump_test,
    "wine_quality": prepare_wine_quality,
    "superconductivity": prepare_superconductivity,
    "auto_mpg": prepare_auto_mpg,
    "combined_cycle_power_plant": prepare_combined_cycle_power_plant,
    "air_quality": prepare_air_quality,
    "parkinsons_telemonitoring": prepare_parkinsons_telemonitoring,
    "real_estate_valuation": prepare_real_estate_valuation,
    "student_performance": prepare_student_performance,
}
PAPER_DATASETS = [name for group in PAPER_GROUPS.values() for name in group]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare PRISM paper datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(PREPARERS),
        default=PAPER_DATASETS,
        help="Datasets to prepare (default: all Main/Case1/Case2 paper datasets).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.datasets:
        path = PREPARERS[name]()
        df = pd.read_csv(path)
        print(f"[DONE] {path} rows={len(df)} cols={len(df.columns)}")


if __name__ == "__main__":
    main()
