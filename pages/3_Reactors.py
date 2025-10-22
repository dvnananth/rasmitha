from typing import Dict, List

import streamlit as st

try:
    st.set_page_config(page_title="ReactorSim - Reactors", page_icon="🧪", layout="wide")
except Exception:
    pass

def _adjust_length(values, expected_len, name):
    vals = [float(x) for x in values]
    if expected_len and len(vals) != expected_len:
        st.warning(f"{name} length {len(vals)} does not match expected {expected_len}. Auto-adjusting.")
        if len(vals) < expected_len:
            vals = vals + [0.0] * (expected_len - len(vals))
        else:
            vals = vals[: expected_len]
    return vals

st.title("Reactors")
st.caption("Configure reactor type and parameters.")

setup = st.session_state.get("reactor_setup", {})
sel_type = st.selectbox(
    "Reactor type",
    ["Batch (isothermal)", "CSTR (adiabatic)", "PFR (isothermal)"],
    index=["Batch (isothermal)", "CSTR (adiabatic)", "PFR (isothermal)"].index(setup.get("type", "Batch (isothermal)"))
)

params = dict(setup.get("params", {}))
kin = st.session_state.get("kinetics")
num_species = len(kin["species"]) if kin and "species" in kin else 0
num_rxn = len(kin["stoich"]) if kin and "stoich" in kin else 0

if sel_type == "Batch (isothermal)":
    st.subheader("Batch parameters")
    T = st.number_input("Temperature T [K]", value=float(params.get("T", 300.0)))
    c0_text = st.text_input("Initial concentrations c0 (per species, comma-separated)", value=", ".join(map(str, params.get("c0", [1.0]))))
    tend = st.number_input("End time [s]", value=float(params.get("tend", 5.0)))
    dt = st.number_input("Time step [s]", value=float(params.get("dt", 0.05)))
    c0_vals = _adjust_length([x for x in c0_text.split(",") if x.strip()], num_species, "c0")
    params_out = {"T": T, "c0": c0_vals, "tend": tend, "dt": dt}

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
    # Normalize vector lengths
    c0_vals = _adjust_length([x for x in c0_text.split(",") if x.strip()], num_species, "c0")
    cin_vals = _adjust_length([x for x in cin_text.split(",") if x.strip()], num_species, "cin")
    dH_vals = _adjust_length([x for x in dH_text.split(",") if x.strip()], num_rxn, "dH")
    params_out = {
        "Tin": Tin,
        "tau": tau,
        "c0": c0_vals,
        "cin": cin_vals,
        "rho_cp": rho_cp,
        "dH": dH_vals,
        "tend": tend,
        "dt": dt,
    }

else:  # PFR (isothermal)
    st.subheader("PFR parameters (isothermal)")
    T = st.number_input("Temperature T [K]", value=float(params.get("T", 300.0)))
    cin_text = st.text_input("Inlet concentrations cin (comma-separated)", value=", ".join(map(str, params.get("cin", [1.0]))))
    tau_end = st.number_input("End residence coordinate [s]", value=float(params.get("tau_end", 3.0)))
    dt = st.number_input("Step [s]", value=float(params.get("dt", 0.05)))
    cin_vals = _adjust_length([x for x in cin_text.split(",") if x.strip()], num_species, "cin")
    params_out = {"T": T, "cin": cin_vals, "tau_end": tau_end, "dt": dt}

if st.button("Save setup", use_container_width=True):
    st.session_state["reactor_setup"] = {"type": sel_type, "params": params_out}
    st.success("Reactor setup saved.")

st.divider()

st.subheader("Current setup")
st.write(st.session_state.get("reactor_setup"))
