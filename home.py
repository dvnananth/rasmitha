import platform
import importlib
import importlib.util
import os
from pathlib import Path
from datetime import datetime
import streamlit as st

try:
    st.set_page_config(
        page_title="ReactorSim",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",  # force sidebar open
    )
except Exception:
    # set_page_config may have been called already if rendering in-place
    pass

# --- Sidebar navigation (manual fallback) ---
st.sidebar.header("Navigate")
_pages = {
    "Home": None,
    "Data": "pages/1_Data.py",
    "Kinetics": "pages/2_Kinetics.py",
    "Reactors": "pages/3_Reactors.py",
    "Solver": "pages/4_Solver.py",
    "Plots": "pages/5_Plots.py",
    "Analytics": "pages/6_Analytics.py",
}
def _render_fallback(page_name: str) -> None:
    target = _pages.get(page_name)
    if not target:
        return
    base_dir = Path(__file__).parent
    abspath = (base_dir / target).resolve()
    try:
        spec = importlib.util.spec_from_file_location("_embedded_page", str(abspath))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # page will render at import time
        else:
            st.error(f"Cannot load page module for {page_name}")
    except Exception as e:
        st.error(f"Failed to render {page_name}: {e}")

_choice = st.sidebar.radio("Go to", list(_pages.keys()), index=0)
if _choice != "Home":
    try:
        st.switch_page(_pages[_choice])
        st.stop()
    except Exception:
        st.sidebar.info("Navigation fallback active. Rendering page here.")
        _render_fallback(_choice)
        st.stop()

st.title("🧪 ReactorSim")
st.caption("Modeling and simulation for Batch, CSTR, and PFR with Arrhenius kinetics")

# --- Quick navigation buttons ---
st.markdown("### Quick navigation")
cols = st.columns(6)
labels = ["Data", "Kinetics", "Reactors", "Solver", "Plots", "Analytics"]
targets = ["pages/1_Data.py","pages/2_Kinetics.py","pages/3_Reactors.py","pages/4_Solver.py","pages/5_Plots.py","pages/6_Analytics.py"]
for i, (label, target) in enumerate(zip(labels, targets)):
    with cols[i]:
        if st.button(label, use_container_width=True):
            try:
                st.switch_page(target)
                st.stop()
            except Exception:
                st.info("Navigation fallback active. Rendering page here.")
                # Map target path back to page name key
                name_lookup = {v: k for k, v in _pages.items() if v is not None}
                page_name = name_lookup.get(target)
                if page_name:
                    _render_fallback(page_name)
                    st.stop()
                else:
                    st.error("Unknown page target.")

st.divider()

# --- Session status ---
st.markdown("### Session status")
left, mid, right = st.columns(3)
with left:
    df = st.session_state.get("cfd_df")
    st.metric("CFD dataset", f"{len(df):,} rows" if df is not None else "Not loaded")
with mid:
    kin = st.session_state.get("kinetics")
    st.metric("Kinetics", f"{len(kin['species'])} species / {len(kin['stoich'])} rxns" if kin else "Not defined")
with right:
    setup = st.session_state.get("reactor_setup")
    st.metric("Reactor setup", setup.get("type", "Defined") if setup else "Not selected")

# --- Quickstart ---
st.markdown("### Quickstart")
st.markdown(
    "- Step 1: (Optional) Load CFD dataset in Data.\n"
    "- Step 2: Define species and reactions in Kinetics.\n"
    "- Step 3: Choose reactor type/params in Reactors.\n"
    "- Step 4: Run in Solver and export CSV.\n"
    "- Step 5: Visualize in Plots.\n"
    "- Step 6: Compare physics vs ML vs hybrid in Analytics."
)

with st.expander("Command-line examples (optional)"):
    st.code(
        "python -m reactorsim.cli pfr --species A B --stoich=-1,1 --A 1 --Ea 0 --T 300 --cin 1 0 --tau_end 3 --dt 0.05 --csv pfr.csv\n"
        "python -m reactorsim.cli batch-adiabatic --species A --stoich=-1 --A 1 --Ea 0 --T0 300 --c0 1 --rho_cp 4000 --dH -50000 --tend 2 --dt 0.02 --csv batch_adiabatic.csv",
        language="bash",
    )

# --- One-click examples ---
st.markdown("### One-click examples")
ex_cols = st.columns(2)
with ex_cols[0]:
    if st.button("Load example: A → B (Batch, isothermal)", use_container_width=True):
        st.session_state["kinetics"] = {"species": ["A", "B"], "stoich": ["-1,1"], "A": [1.0], "Ea": [0.0]}
        st.session_state["reactor_setup"] = {"type": "Batch (isothermal)", "params": {"T": 300.0, "c0": [1.0, 0.0], "tend": 5.0, "dt": 0.1}}
        try:
            st.switch_page("pages/4_Solver.py")
        except Exception:
            st.info("Use the sidebar to open Solver tab.")
with ex_cols[1]:
    if st.button("Load example: A (CSTR, adiabatic)", use_container_width=True):
        st.session_state["kinetics"] = {"species": ["A"], "stoich": ["-1"], "A": [1.0], "Ea": [0.0]}
        st.session_state["reactor_setup"] = {
            "type": "CSTR (adiabatic)",
            "params": {"Tin": 300.0, "c0": [0.0], "cin": [1.0], "tau": 1.0, "rho_cp": 4000.0, "dH": [-50000.0], "tend": 5.0, "dt": 0.05},
        }
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
