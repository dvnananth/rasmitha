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
kin = st.session_state.get("kinetics")
num_species = len(kin["species"]) if kin and "species" in kin else 0
num_rxn = len(kin["stoich"]) if kin and "stoich" in kin else 0

if sel_type == "Batch (isothermal)":
    st.subheader("Batch parameters")
    T = st.number_input("Temperature T [K]", value=float(params.get("T", 300.0)))
    c0_text = st.text_input("Initial concentrations c0 (per species, comma-separated)", value=", ".join(map(str, params.get("c0", [1.0]))))
    tend = st.number_input("End time [s]", value=float(params.get("tend", 5.0)))
    dt = st.number_input("Time step [s]", value=float(params.get("dt", 0.05)))
    c0_vals = [float(x) for x in c0_text.split(",") if x.strip()]
    if num_species and len(c0_vals) != num_species:
        st.warning(f"c0 length {len(c0_vals)} does not match species count {num_species}. Auto-adjusting by pad/truncate.")
        if len(c0_vals) < num_species:
            c0_vals = c0_vals + [0.0] * (num_species - len(c0_vals))
        else:
            c0_vals = c0_vals[: num_species]
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
    c0_vals = [float(x) for x in c0_text.split(",") if x.strip()]
    cin_vals = [float(x) for x in cin_text.split(",") if x.strip()]
    if num_species:
        for name, arr in [("c0", c0_vals), ("cin", cin_vals)]:
            if len(arr) != num_species:
                st.warning(f"{name} length {len(arr)} != species count {num_species}. Auto-adjusting.")
                if len(arr) < num_species:
                    arr += [0.0] * (num_species - len(arr))
                else:
                    del arr[num_species:]
    dH_vals = [float(x) for x in dH_text.split(",") if x.strip()]
    if num_rxn and len(dH_vals) != num_rxn:
        st.warning(f"dH length {len(dH_vals)} != reaction count {num_rxn}. Auto-adjusting.")
        if len(dH_vals) < num_rxn:
            dH_vals += [0.0] * (num_rxn - len(dH_vals))
        else:
            del dH_vals[num_rxn:]
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
    cin_vals = [float(x) for x in cin_text.split(",") if x.strip()]
    if num_species and len(cin_vals) != num_species:
        st.warning(f"cin length {len(cin_vals)} != species count {num_species}. Auto-adjusting.")
        if len(cin_vals) < num_species:
            cin_vals += [0.0] * (num_species - len(cin_vals))
        else:
            del cin_vals[num_species:]
    params_out = {"T": T, "cin": cin_vals, "tau_end": tau_end, "dt": dt}

if st.button("Save setup", use_container_width=True):
    st.session_state["reactor_setup"] = {"type": sel_type, "params": params_out}
    st.success("Reactor setup saved.")

st.divider()

st.subheader("Current setup")
st.write(st.session_state.get("reactor_setup"))
