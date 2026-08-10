from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ScalingArtifacts:
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_columns: list
    target_column: str


def knn_impute(df: pd.DataFrame, n_neighbors: int = 3) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df.values)
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)


def fit_minmax_scalers(
    features: pd.DataFrame,
    target: pd.Series,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> ScalingArtifacts:
    fscaler = MinMaxScaler(feature_range=feature_range)
    tscaler = MinMaxScaler(feature_range=feature_range)
    fscaler.fit(features.values)
    tscaler.fit(target.values.reshape(-1, 1))
    return ScalingArtifacts(
        feature_scaler=fscaler,
        target_scaler=tscaler,
        feature_columns=list(features.columns),
        target_column=target.name,
    )


def apply_scalers(
    artifacts: ScalingArtifacts,
    features: pd.DataFrame,
    target: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    f = artifacts.feature_scaler.transform(features[artifacts.feature_columns].values)
    t = None
    if target is not None:
        t = artifacts.target_scaler.transform(target.values.reshape(-1, 1)).reshape(-1)
    return f.astype(np.float32), (t.astype(np.float32) if t is not None else None)


def inverse_scale_target(artifacts: ScalingArtifacts, y_norm: np.ndarray) -> np.ndarray:
    return artifacts.target_scaler.inverse_transform(y_norm.reshape(-1, 1)).reshape(-1)


def make_sliding_windows(
    array_2d: np.ndarray,
    window_size: int,
    step: int = 1,
) -> np.ndarray:
    """Create sliding windows over the first dimension of a 2D array.

    Parameters
    ----------
    array_2d: shape (T, D)
    window_size: number of steps per window
    step: stride between window starts

    Returns
    -------
    windows: shape (N, window_size, D)
    """
    T, D = array_2d.shape
    if T < window_size:
        raise ValueError("Time series shorter than window_size")
    starts = np.arange(0, T - window_size + 1, step)
    n = len(starts)
    windows = np.empty((n, window_size, D), dtype=array_2d.dtype)
    for i, s in enumerate(starts):
        windows[i] = array_2d[s : s + window_size]
    return windows


def compute_ti_proxy(wind_speed: pd.Series, window: int = 6) -> pd.Series:
    mean_ws = wind_speed.rolling(window=window, min_periods=1).mean()
    std_ws = wind_speed.rolling(window=window, min_periods=1).std().fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ti = (std_ws / (mean_ws.replace(0, np.nan))).fillna(0.0)
    return ti.clip(lower=0.0)


def compute_yaw_misalignment_proxy(direction_deg: pd.Series, window: int = 6) -> pd.Series:
    # Simple proxy: short-term direction variability
    diff = direction_deg.diff().abs().fillna(0.0)
    yaw_var = diff.rolling(window=window, min_periods=1).mean()
    return yaw_var
