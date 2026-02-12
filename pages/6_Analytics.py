from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from reactorsim.core import parse_kinetics, simulate_batch_isothermal, simulate_cstr_adiabatic, simulate_pfr_isothermal

try:
    st.set_page_config(page_title="ReactorSim - Analytics", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Analytics")
st.caption("Compare Normal Physics vs AIML vs Hybrid (Physics + AIML Residuals)")

# Utilities

def _infer_xcol(df: pd.DataFrame) -> str:
    for c in ["t", "z", "time", "x"]:
        if c in df.columns:
            return c
    if "volume" in df.columns:
        return "volume"
    return df.columns[0]


def _infer_species(df: pd.DataFrame, kin: Optional[Dict]) -> List[str]:
    if kin and kin.get("species"):
        return [s for s in kin["species"] if s in df.columns]
    # Support both wide and long formats
    if {"species", "value"}.issubset(df.columns):
        # long format; species will be handled by pivot
        species_names = sorted([s for s in df["species"].unique() if isinstance(s, (str, int, float))])
        return [str(s) for s in species_names]
    exclude = {"t", "time", "z", "x", "volume", "T"}
    return [c for c in df.columns if c not in exclude]


def _ensure_wide(df: pd.DataFrame, kin: Optional[Dict]) -> pd.DataFrame:
    if {"species", "value"}.issubset(df.columns):
        xcol = _infer_xcol(df)
        try:
            wide = df.pivot_table(index=xcol, columns="species", values="value", aggfunc="mean").reset_index()
            return wide
        except Exception:
            return df
    return df


def _physics_predict_grid(kin: Dict, setup: Dict, x_grid: np.ndarray) -> pd.DataFrame:
    kinetics = parse_kinetics(kin)
    rtype = setup.get("type")
    params = dict(setup.get("params", {}))

    if rtype == "Batch (isothermal)":
        params = {**params, "tend": float(x_grid.max()), "dt": float(np.diff(x_grid).min(initial=0.05) or 0.05)}
        dfp = simulate_batch_isothermal(kinetics, params)
        xcol = "t"
    elif rtype == "CSTR (adiabatic)":
        params = {**params, "tend": float(x_grid.max()), "dt": float(np.diff(x_grid).min(initial=0.05) or 0.05)}
        dfp = simulate_cstr_adiabatic(kinetics, params)
        xcol = "t"
    elif rtype == "PFR (isothermal)":
        params = {**params, "tau_end": float(x_grid.max()), "dt": float(np.diff(x_grid).min(initial=0.05) or 0.05)}
        dfp = simulate_pfr_isothermal(kinetics, params)
        xcol = "z"
    else:
        raise ValueError(f"Unknown reactor type: {rtype}")

    # Interpolate to x_grid
    out = pd.DataFrame({xcol: x_grid})
    for col in [c for c in dfp.columns if c not in {xcol}]:
        out[col] = np.interp(x_grid, dfp[xcol].to_numpy(), dfp[col].to_numpy())
    return out


def _build_features(df: pd.DataFrame, setup: Optional[Dict]) -> pd.DataFrame:
    xcol = _infer_xcol(df)
    X = pd.DataFrame({xcol: df[xcol].to_numpy()})
    # Add temperature if present
    for c in ["T", "Tin"]:
        if c in df.columns:
            X[c] = df[c].to_numpy()
    # Add static reactor params as features
    if setup:
        for key, val in setup.get("params", {}).items():
            if isinstance(val, (int, float)):
                X[key] = float(val)
    return X


@dataclass
class EvalMetrics:
    rmse: float
    r2: float


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    return EvalMetrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


# Data sources
cfd_df: Optional[pd.DataFrame] = st.session_state.get("cfd_df")
sim_df: Optional[pd.DataFrame] = st.session_state.get("sim_df")
kin = st.session_state.get("kinetics")
setup = st.session_state.get("reactor_setup")

source = st.radio("Ground truth dataset", [opt for opt in ["CFD/Uploaded", "Simulation"] if (cfd_df if opt=="CFD/Uploaded" else sim_df) is not None])
if source == "CFD/Uploaded":
df_gt = cfd_df
else:
df_gt = sim_df

if df_gt is None:
    st.warning("No dataset available. Upload in Data tab or run a simulation in Solver tab.")
    st.stop()

if not kin or not setup:
    st.warning("Define kinetics and reactor setup to enable Physics/Hybrid predictions.")

df_gt = _ensure_wide(df_gt, kin)
xcol = _infer_xcol(df_gt)
species_cols = _infer_species(df_gt, kin)

st.write(f"Detected independent variable: {xcol}")
st.write(f"Target variables: {species_cols}")

if not species_cols:
    st.warning("No target variables found. Ensure your dataset includes species columns (not just t/z and T).")
    st.stop()

# Train/test split
split = st.slider("Train fraction", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
random_state = st.number_input("Random seed", min_value=0, step=1, value=42)

# Option: train per-species RandomForest
n_estimators = st.number_input("RandomForest trees", min_value=10, max_value=1000, value=200, step=10)

# Build feature matrix
X_full = _build_features(df_gt, setup)

# Physics predictions on same grid
df_phys = None
if kin and setup:
    try:
        df_phys = _physics_predict_grid(kin, setup, df_gt[xcol].to_numpy())
    except Exception as e:
        st.warning(f"Physics prediction failed: {e}")

# Train/evaluate
metrics_rows: List[Dict] = []
pred_frames: Dict[str, pd.DataFrame] = {}

for target in species_cols:
    y = df_gt[target].to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_full, y, np.arange(len(y)), train_size=split, random_state=int(random_state)
    )

    # AIML model
    rf = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))
    rf.fit(X_train, y_train)
    y_pred_ml = rf.predict(X_full)

    # Physics baseline
    if df_phys is not None and target in df_phys.columns:
        y_pred_phys = df_phys[target].to_numpy()
    else:
        y_pred_phys = np.full_like(y, fill_value=np.nan, dtype=float)

    # Hybrid residuals: y ≈ y_phys + f(X)
    if np.all(np.isfinite(y_pred_phys)):
        resid = y_train - y_pred_phys[idx_train]
        rf_resid = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state))
        rf_resid.fit(X_train, resid)
        y_pred_hybrid = y_pred_phys + rf_resid.predict(X_full)
    else:
        y_pred_hybrid = y_pred_ml.copy()

    # Evaluate on test indices
    m_phys = _evaluate(y_test, y_pred_phys[idx_test]) if np.all(np.isfinite(y_pred_phys)) else EvalMetrics(np.nan, np.nan)
    m_ml = _evaluate(y_test, y_pred_ml[idx_test])
    m_hyb = _evaluate(y_test, y_pred_hybrid[idx_test])

    metrics_rows.append({
        "variable": target,
        "physics_rmse": m_phys.rmse,
        "physics_r2": m_phys.r2,
        "ml_rmse": m_ml.rmse,
        "ml_r2": m_ml.r2,
        "hybrid_rmse": m_hyb.rmse,
        "hybrid_r2": m_hyb.r2,
    })

    pred_frames[target] = pd.DataFrame({
        xcol: df_gt[xcol].to_numpy(),
        "y_true": y,
        "y_phys": y_pred_phys,
        "y_ml": y_pred_ml,
        "y_hybrid": y_pred_hybrid,
    })

# Show metrics
st.subheader("Performance metrics (lower RMSE, higher R² are better)")
metrics_df = pd.DataFrame(metrics_rows)
st.dataframe(metrics_df, use_container_width=True)

csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
st.download_button("Download metrics CSV", data=csv_bytes, file_name="analytics_metrics.csv", mime="text/csv")

# Plots
st.subheader("Predictions vs Ground Truth")
sel = st.selectbox("Variable", species_cols)
if sel in pred_frames:
    dfp = pred_frames[sel]
    cols_to_plot = [c for c in ["y_true", "y_phys", "y_ml", "y_hybrid"] if c in dfp.columns]
    st.line_chart(dfp.set_index(xcol)[cols_to_plot])

# Save results in session
st.session_state["analytics_metrics"] = metrics_df
