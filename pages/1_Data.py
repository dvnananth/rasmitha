import io
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

try:
    st.set_page_config(page_title="ReactorSim - Data", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Data")
st.caption("Load a CFD/experimental dataset, or synthesize one from the current model.")

# Helpers

def _infer_species_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"t", "time", "z", "T"}
    return [c for c in df.columns if c not in exclude]


# Uploader
uploaded = st.file_uploader("Upload CSV", type=["csv"])  # expecting columns like t/z, species names, optional T
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.session_state["cfd_df"] = df
        st.success(f"Loaded dataset with shape {df.shape}")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Existing dataset
cfd_df = st.session_state.get("cfd_df")
if cfd_df is not None:
    st.subheader("Current dataset")
    st.dataframe(cfd_df.head(50), use_container_width=True)
    st.write(f"Columns: {list(cfd_df.columns)}")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear dataset", use_container_width=True):
        st.session_state.pop("cfd_df", None)
        st.experimental_rerun()

with col2:
    if cfd_df is not None:
        csv_bytes = cfd_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="dataset.csv", mime="text/csv", use_container_width=True)

with col3:
    st.write("")

st.divider()

# Synthesize dataset from current model
st.subheader("Synthesize dataset from model")
st.caption("Runs the current physics model and adds small noise to emulate measurements.")
kin = st.session_state.get("kinetics")
setup = st.session_state.get("reactor_setup")
noise = st.slider("Noise level (std as fraction of value)", min_value=0.0, max_value=0.1, value=0.02, step=0.005)

if st.button("Generate synthetic dataset", use_container_width=True):
    if not kin or not setup:
        st.warning("Please define kinetics and reactor setup first.")
    else:
        try:
            from reactorsim.core import parse_kinetics, simulate_batch_isothermal, simulate_cstr_adiabatic, simulate_pfr_isothermal

            kinetics = parse_kinetics(kin)
            rtype = setup.get("type")
            params = dict(setup.get("params", {}))
            if rtype == "Batch (isothermal)":
                df_sim = simulate_batch_isothermal(kinetics, params)
            elif rtype == "CSTR (adiabatic)":
                df_sim = simulate_cstr_adiabatic(kinetics, params)
            elif rtype == "PFR (isothermal)":
                df_sim = simulate_pfr_isothermal(kinetics, params)
            else:
                st.error(f"Unknown reactor type: {rtype}")
                df_sim = None

            if df_sim is not None:
                species_cols = _infer_species_columns(df_sim)
                noisy = df_sim.copy()
                for c in species_cols:
                    val = noisy[c].to_numpy()
                    sigma = np.maximum(np.abs(val) * noise, 1e-12)
                    noisy[c] = val + np.random.normal(0.0, sigma)
                st.session_state["cfd_df"] = noisy
                st.success("Synthetic dataset generated and loaded as current dataset.")
                st.dataframe(noisy.head(50), use_container_width=True)
        except Exception as e:
            st.exception(e)
