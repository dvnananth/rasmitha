import platform
import importlib
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="ReactorSim", page_icon="🧪", layout="wide")

st.title("🧪 ReactorSim")
st.caption("Modeling and simulation for Batch, CSTR, and PFR with Arrhenius kinetics")

# --- Quick Links / Navigation ---
st.markdown("### Quick navigation")
cols = st.columns(5)
with cols[0]:
    if st.button("Data", use_container_width=True):
        try:
            st.switch_page("pages/1_Data.py")
        except Exception:
            st.info("Use the sidebar to open Data tab.")
with cols[1]:
    if st.button("Kinetics", use_container_width=True):
        try:
            st.switch_page("pages/2_Kinetics.py")
        except Exception:
            st.info("Use the sidebar to open Kinetics tab.")
with cols[2]:
    if st.button("Reactors", use_container_width=True):
        try:
            st.switch_page("pages/3_Reactors.py")
        except Exception:
            st.info("Use the sidebar to open Reactors tab.")
with cols[3]:
    if st.button("Solver", use_container_width=True):
        try:
            st.switch_page("pages/4_Solver.py")
        except Exception:
            st.info("Use the sidebar to open Solver tab.")
with cols[4]:
    if st.button("Plots", use_container_width=True):
        try:
            st.switch_page("pages/5_Plots.py")
        except Exception:
            st.info("Use the sidebar to open Plots tab.")

st.divider()

# --- Session Status ---
st.markdown("### Session status")
left, mid, right = st.columns(3)
with left:
    df = st.session_state.get("cfd_df")
    if df is not None:
        st.metric("CFD dataset", f"{len(df):,} rows")
    else:
        st.metric("CFD dataset", "Not loaded")
with mid:
    kin = st.session_state.get("kinetics")
    if kin is not None:
        st.metric("Kinetics", f"{len(kin['species'])} species / {len(kin['stoich'])} rxns")
    else:
        st.metric("Kinetics", "Not defined")
with right:
    setup = st.session_state.get("reactor_setup")
    if setup is not None:
        st.metric("Reactor setup", setup.get("type", "Defined"))
    else:
        st.metric("Reactor setup", "Not selected")

# --- Quickstart ---
st.markdown("### Quickstart")
st.markdown(
    "- **Step 1**: Load CFD dataset in Data tab (optional).\n"
    "- **Step 2**: Define species and reactions in Kinetics tab.\n"
    "- **Step 3**: Choose reactor type and parameters in Reactors tab.\n"
    "- **Step 4**: Run the solver and export CSV.\n"
    "- **Step 5**: Visualize results in Plots tab."
)

with st.expander("Command-line examples (optional)"):
    st.code(
        """
python -m reactorsim.cli pfr --species A B --stoich=-1,1 --A 1 --Ea 0 --T 300 --cin 1 0 --tau_end 3 --dt 0.05 --csv pfr.csv
python -m reactorsim.cli batch-adiabatic --species A --stoich=-1 --A 1 --Ea 0 --T0 300 --c0 1 --rho_cp 4000 --dH -50000 --tend 2 --dt 0.02 --csv batch_adiabatic.csv
        """,
        language="bash",
    )

# --- One-click examples ---
st.markdown("### One-click examples")
ex_cols = st.columns(2)
with ex_cols[0]:
    if st.button("Load example: A → B (Batch, isothermal)", use_container_width=True):
        st.session_state["kinetics"] = {
            "species": ["A", "B"],
            "stoich": ["-1,1"],
            "A": [1.0],
            "Ea": [0.0],
        }
        st.session_state["reactor_setup"] = {
            "type": "Batch (isothermal)",
            "params": {"T": 300.0, "c0": [1.0, 0.0], "tend": 5.0, "dt": 0.1},
        }
        st.success("Example loaded. Opening Solver…")
        try:
            st.switch_page("pages/4_Solver.py")
        except Exception:
            st.info("Use the sidebar to open Solver tab.")
with ex_cols[1]:
    if st.button("Load example: A (CSTR, adiabatic)", use_container_width=True):
        st.session_state["kinetics"] = {
            "species": ["A"],
            "stoich": ["-1"],
            "A": [1.0],
            "Ea": [0.0],
        }
        st.session_state["reactor_setup"] = {
            "type": "CSTR (adiabatic)",
            "params": {
                "Tin": 300.0,
                "c0": [0.0],
                "cin": [1.0],
                "tau": 1.0,
                "rho_cp": 4000.0,
                "dH": [-50000.0],
                "tend": 5.0,
                "dt": 0.05,
            },
        }
        st.success("Example loaded. Opening Solver…")
        try:
            st.switch_page("pages/4_Solver.py")
        except Exception:
            st.info("Use the sidebar to open Solver tab.")

st.divider()

# --- Environment / About ---
st.markdown("### Environment")
pyver = platform.python_version()
try:
    rx = importlib.import_module("reactorsim")
    rx_ver = getattr(rx, "__version__", "unknown")
except Exception:
    rx_ver = "unknown"

env_cols = st.columns(3)
with env_cols[0]:
    st.write(f"Python: {pyver}")
with env_cols[1]:
    st.write(f"ReactorSim: {rx_ver}")
with env_cols[2]:
    st.write(f"Launched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.info("Tip: Use the sidebar to switch tabs at any time.")
