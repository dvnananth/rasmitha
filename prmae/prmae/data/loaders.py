from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from prmae.utils.common import knn_impute, compute_ti_proxy, compute_yaw_misalignment_proxy
from prmae.data.power_curve import PowerCurve


@dataclass
class T1Columns:
    timestamp: str = "Date/Time"
    power_kw: str = "LV ActivePower (kW)"
    wind_speed: str = "Wind Speed (m/s)"
    theoretical_power: str = "Theoretical_Power_Curve (KWh)"  # note: kWh label in sample
    wind_direction: str = "Wind Direction (°)"


@dataclass
class TexasColumns:
    timestamp: str = "Time stamp"
    power_kw: str = "System power generated | (kW)"
    wind_speed: str = "Wind speed | (m/s)"
    wind_direction: str = "Wind direction | (deg)"
    pressure_atm: str = "Pressure | (atm)"
    air_temp_c: str = "Air temperature | ('C)"


def load_t1_csv(path: str, cols: T1Columns = T1Columns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse time with multiple possible formats
    df["timestamp"] = pd.to_datetime(df[cols.timestamp], errors="coerce", dayfirst=True, infer_datetime_format=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rename to canonical
    rename_map = {
        cols.power_kw: "power_kw",
        cols.wind_speed: "wind_speed",
        cols.theoretical_power: "power_theoretical",
        cols.wind_direction: "wind_direction_deg",
    }
    df = df.rename(columns=rename_map)

    # Impute numeric columns
    num = df[["power_kw", "wind_speed", "power_theoretical", "wind_direction_deg"]].copy()
    num = knn_impute(num)

    # Features
    df = df.drop(columns=[cols.timestamp]).copy()
    df[["power_kw", "wind_speed", "power_theoretical", "wind_direction_deg"]] = num

    # Physics proxies
    df["ti_proxy"] = compute_ti_proxy(df["wind_speed"], window=6)
    df["yaw_var"] = compute_yaw_misalignment_proxy(df["wind_direction_deg"], window=6)
    return df


def load_texas_csv(path: str, cols: TexasColumns = TexasColumns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df[cols.timestamp], errors="coerce", infer_datetime_format=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    rename_map = {
        cols.power_kw: "power_kw",
        cols.wind_speed: "wind_speed",
        cols.wind_direction: "wind_direction_deg",
        cols.pressure_atm: "pressure_atm",
        cols.air_temp_c: "air_temp_c",
    }
    df = df.rename(columns=rename_map)

    # Impute
    num_cols = ["power_kw", "wind_speed", "wind_direction_deg", "pressure_atm", "air_temp_c"]
    num = knn_impute(df[num_cols])
    df[num_cols] = num

    # Physics proxies
    df["ti_proxy"] = compute_ti_proxy(df["wind_speed"], window=6)
    df["yaw_var"] = compute_yaw_misalignment_proxy(df["wind_direction_deg"], window=6)
    return df


def compute_theoretical_power_from_curve(df: pd.DataFrame, power_curve: PowerCurve) -> pd.Series:
    p = power_curve.map_wind_to_power(df["wind_speed"].values.astype(float))
    return pd.Series(p, index=df.index, name="power_theoretical")
