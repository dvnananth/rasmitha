from __future__ import annotations

import numpy as np

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def physics_validity_score(cp_est: np.ndarray, betz_limit: float = 0.593) -> float:
    within = (cp_est >= 0.0) & (cp_est <= betz_limit)
    return float(np.mean(within.astype(np.float32)))


def residual_energy_attribution(r_micro: np.ndarray, r_meso: np.ndarray, r_macro: np.ndarray) -> dict:
    # report variance explained by each component in residual
    total = r_micro + r_meso + r_macro
    var_total = np.var(total) + 1e-12
    return {
        "micro_var_frac": float(np.var(r_micro) / var_total),
        "meso_var_frac": float(np.var(r_meso) / var_total),
        "macro_var_frac": float(np.var(r_macro) / var_total),
    }
