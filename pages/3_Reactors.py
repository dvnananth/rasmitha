from typing import Dict, List

import streamlit as st

try:
    st.set_page_config(page_title="ReactorSim - Reactors", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Reactors")
st.caption("Configure reactor type and parameters.")

setup = st.session_state.get("reactor_setup", {})
sel_type = st.selectbox(
    "Reactor type",
    ["Batch (isothermal)", "CSTR (adiabatic)", "PFR (isothermal)"],
    index=["Batch (isothermal)", "CSTR (adiabatic)", "PFR (isothermal)"].index(setup.get("type", "Batch (isothermal)"))
)

params = dict(setup.get("params", {}))

if sel_type == "Batch (isothermal)":
    st.subheader("Batch parameters")
    T = st.number_input("Temperature T [K]", value=float(params.get("T", 300.0)))
    c0_text = st.text_input("Initial concentrations c0 (per species, comma-separated)", value=", ".join(map(str, params.get("c0", [1.0]))))
    tend = st.number_input("End time [s]", value=float(params.get("tend", 5.0)))
    dt = st.number_input("Time step [s]", value=float(params.get("dt", 0.05)))
    params_out = {"T": T, "c0": [float(x) for x in c0_text.split(",") if x.strip()], "tend": tend, "dt": dt}

elif sel_type == "CSTR (adiabatic)":
    st.subheader("CSTR parameters (dynamic, adiabatic)")
    Tin = st.number_input("Feed temperature Tin [K]", value=float(params.get("Tin", 300.0)))
    tau = st.number_input("Residence time tau [s]", value=float(params.get("tau", 1.0)))
    c0_text = st.text_input("Initial concentrations c0 (comma-separated)", value=", ".join(map(str, params.get("c0", [0.0]))))
    cin_text = st.text_input("Feed concentrations cin (comma-separated)", value=", ".join(map(str, params.get("cin", [1.0]))))
    rho_cp = st.number_input("rho*cp [J/(m^3*K)]", value=float(params.get("rho_cp", 4000.0)))
    dH_text = st.text_input("Reaction enthalpies dH [J/mol] (per reaction, comma-separated)", value=", ".join(map(str, params.get("dH", [-5e4]))))
    tend = st.number_input("End time [s]", value=float(params.get("tend", 5.0)))
    dt = st.number_input("Time step [s]", value=float(params.get("dt", 0.05)))
    params_out = {
        "Tin": Tin,
        "tau": tau,
        "c0": [float(x) for x in c0_text.split(",") if x.strip()],
        "cin": [float(x) for x in cin_text.split(",") if x.strip()],
        "rho_cp": rho_cp,
        "dH": [float(x) for x in dH_text.split(",") if x.strip()],
        "tend": tend,
        "dt": dt,
    }

else:  # PFR (isothermal)
    st.subheader("PFR parameters (isothermal)")
    T = st.number_input("Temperature T [K]", value=float(params.get("T", 300.0)))
    cin_text = st.text_input("Inlet concentrations cin (comma-separated)", value=", ".join(map(str, params.get("cin", [1.0]))))
    tau_end = st.number_input("End residence coordinate [s]", value=float(params.get("tau_end", 3.0)))
    dt = st.number_input("Step [s]", value=float(params.get("dt", 0.05)))
    params_out = {"T": T, "cin": [float(x) for x in cin_text.split(",") if x.strip()], "tau_end": tau_end, "dt": dt}

if st.button("Save setup", use_container_width=True):
    st.session_state["reactor_setup"] = {"type": sel_type, "params": params_out}
    st.success("Reactor setup saved.")

st.divider()

st.subheader("Current setup")
st.write(st.session_state.get("reactor_setup"))
