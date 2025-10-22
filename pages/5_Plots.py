from typing import List

import numpy as np
import pandas as pd
import streamlit as st

try:
    st.set_page_config(page_title="ReactorSim - Plots", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Plots")
st.caption("Visualize time/space series for simulations and datasets.")

sim_df = st.session_state.get("sim_df")
cfd_df = st.session_state.get("cfd_df")

col_top = st.columns(3)
with col_top[0]:
    which = st.selectbox("Dataset", [opt for opt in ["Simulation", "CFD/Uploaded"] if (sim_df if opt=="Simulation" else cfd_df) is not None])

if which == "Simulation":
    df = sim_df
else:
    df = cfd_df

if df is None:
    st.info("No data to plot yet.")
else:
    # Determine independent and targets
    indep_candidates = [c for c in ["t", "z", "time", "x"] if c in df.columns]
    xcol = indep_candidates[0] if indep_candidates else df.columns[0]
    target_cols = [c for c in df.columns if c not in {xcol, "T"}]

    st.write(f"Using independent variable: {xcol}")
    sel_targets = st.multiselect("Variables to plot", target_cols, default=target_cols)

    if sel_targets:
        st.line_chart(df.set_index(xcol)[sel_targets])

    if sim_df is not None and cfd_df is not None and which == "Simulation":
        st.subheader("Overlay with CFD/Uploaded")
        overlay_targets = [c for c in sel_targets if c in cfd_df.columns]
        if overlay_targets:
            st.line_chart(
                pd.DataFrame({
                    **{f"sim_{c}": sim_df.set_index(xcol)[c] for c in overlay_targets if c in sim_df.columns},
                    **{f"cfd_{c}": cfd_df.set_index(xcol)[c] for c in overlay_targets if c in cfd_df.columns},
                })
            )
