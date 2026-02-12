from typing import Dict

import pandas as pd
import streamlit as st

from reactorsim.core import parse_kinetics, simulate_batch_isothermal, simulate_cstr_adiabatic, simulate_pfr_isothermal

try:
    st.set_page_config(page_title="ReactorSim - Solver", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Solver")
st.caption("Run physics simulations and export results.")

kin = st.session_state.get("kinetics")
setup = st.session_state.get("reactor_setup")

if not kin:
    st.warning("Kinetics not defined. Go to Kinetics tab.")
if not setup:
    st.warning("Reactor setup not defined. Go to Reactors tab.")

run = st.button("Run simulation", use_container_width=True)

if run and kin and setup:
    try:
        kinetics = parse_kinetics(kin)
        rtype = setup.get("type")
        params = dict(setup.get("params", {}))
        if rtype == "Batch (isothermal)":
            df = simulate_batch_isothermal(kinetics, params)
        elif rtype == "CSTR (adiabatic)":
            df = simulate_cstr_adiabatic(kinetics, params)
        elif rtype == "PFR (isothermal)":
            df = simulate_pfr_isothermal(kinetics, params)
        else:
            st.error(f"Unknown reactor type: {rtype}")
            df = None

        if df is not None:
            st.session_state["sim_df"] = df
            st.success(f"Simulation complete: {df.shape[0]} points, {df.shape[1]} columns.")
            st.dataframe(df.head(100), use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="simulation.csv", mime="text/csv")
    except Exception as e:
        st.exception(e)

st.divider()

st.subheader("Current simulation result")
st.dataframe(st.session_state.get("sim_df"), use_container_width=True)
