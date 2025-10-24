from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


@dataclass
class PowerCurve:
    wind_speeds: np.ndarray  # m/s
    powers_kw: np.ndarray  # kW

    def interpolate(self) -> interp1d:
        # Ensure monotonic increasing for interpolation
        order = np.argsort(self.wind_speeds)
        ws = self.wind_speeds[order]
        p = self.powers_kw[order]
        return interp1d(ws, p, kind="linear", bounds_error=False, fill_value=(p[0], p[-1]))

    @staticmethod
    def from_ge_csv(path: str) -> "PowerCurve":
        # Expect two columns: Power (kW), Wind Speed (m/s)
        df = pd.read_csv(path)
        # Try robust column detection
        cols = [c.lower() for c in df.columns]
        power_col = [c for c in df.columns if "power" in c.lower()][0]
        ws_col = [c for c in df.columns if "wind" in c.lower() and "speed" in c.lower()][0]
        powers = df[power_col].values.astype(float)
        ws = df[ws_col].values.astype(float)
        return PowerCurve(wind_speeds=ws, powers_kw=powers)

    def map_wind_to_power(self, wind_speed: np.ndarray) -> np.ndarray:
        f = self.interpolate()
        return f(wind_speed)
