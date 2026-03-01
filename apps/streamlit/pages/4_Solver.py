import streamlit as st
import numpy as np
import csv

from reactorsim.kinetics import Arrhenius, Reaction, Network
from reactorsim.reactors import BatchIsothermal, CSTRIsothermal, PFRIsothermal, BatchAdiabatic, CSTRAdiabatic
from reactorsim.solver import integrate_ode

st.title("Solver")

if "kinetics" not in st.session_state or "reactor_setup" not in st.session_state:
    st.warning("Define kinetics and reactor setup first.")
    st.stop()

kin = st.session_state["kinetics"]
setup = st.session_state["reactor_setup"]
species = kin["species"]

# Build network
reactions = []
for r_str, A, Ea in zip(kin["stoich"], kin["A"], kin["Ea"]):
    nums = [float(x) for x in r_str.split(',')]
    sto = {sp: nu for sp, nu in zip(species, nums)}
    reactions.append(Reaction(forward=Arrhenius(A=A, Ea=Ea), stoich=sto))
net = Network(species=species, reactions=reactions)

rx_type = setup["type"]
params = setup["params"]

run = st.button("Run Simulation")

if run:
    if rx_type == "Batch (isothermal)":
        model = BatchIsothermal(network=net, T_K=params["T"])
        t_eval = np.arange(0.0, params["tend"] + 1e-12, params["dt"])
        res = integrate_ode(model.rhs, y0=params["c0"], t_span=(0.0, params["tend"]), t_eval=t_eval)
        header = ["time"] + species
        rows = [[t] + [res.y[i][k] for i in range(len(species))] for k, t in enumerate(res.t)]
    elif rx_type == "CSTR (isothermal)":
        model = CSTRIsothermal(network=net, T_K=params["T"], residence_time_h=params["tau"], feed_conc=params["cin"])
        t_eval = np.arange(0.0, params["tend"] + 1e-12, params["dt"])
        res = integrate_ode(model.rhs, y0=params["c0"], t_span=(0.0, params["tend"]), t_eval=t_eval)
        header = ["time"] + species
        rows = [[t] + [res.y[i][k] for i in range(len(species))] for k, t in enumerate(res.t)]
    elif rx_type == "PFR (isothermal)":
        model = PFRIsothermal(network=net, T_K=params["T"])
        t_eval = np.arange(0.0, params["tau_end"] + 1e-12, params["dt"])
        res = integrate_ode(model.rhs, y0=params["cin"], t_span=(0.0, params["tau_end"]), t_eval=t_eval)
        header = ["tau"] + species
        rows = [[t] + [res.y[i][k] for i in range(len(species))] for k, t in enumerate(res.t)]
    elif rx_type == "Batch (adiabatic)":
        model = BatchAdiabatic(network=net, T0_K=params["T0"], rho_cp=params["rho_cp"], dH_rxn_J_per_mol=params["dH"])
        y0 = list(params["c0"]) + [params["T0"]]
        t_eval = np.arange(0.0, params["tend"] + 1e-12, params["dt"])
        res = integrate_ode(model.rhs, y0=y0, t_span=(0.0, params["tend"]), t_eval=t_eval)
        header = ["time"] + species + ["T_K"]
        rows = [[t] + [res.y[i][k] for i in range(len(species))] + [res.y[len(species)][k]] for k, t in enumerate(res.t)]
    elif rx_type == "CSTR (adiabatic)":
        model = CSTRAdiabatic(network=net, T_in_K=params["Tin"], residence_time_h=params["tau"], feed_conc=params["cin"], rho_cp=params["rho_cp"], dH_rxn_J_per_mol=params["dH"])
        y0 = list(params["c0"]) + [params["Tin"]]
        t_eval = np.arange(0.0, params["tend"] + 1e-12, params["dt"])
        res = integrate_ode(model.rhs, y0=y0, t_span=(0.0, params["tend"]), t_eval=t_eval)
        header = ["time"] + species + ["T_K"]
        rows = [[t] + [res.y[i][k] for i in range(len(species))] + [res.y[len(species)][k]] for k, t in enumerate(res.t)]
    else:
        st.error("Unsupported reactor type")
        st.stop()

    st.session_state["results_header"] = header
    st.session_state["results_rows"] = rows

    st.success("Simulation complete. Go to Plots tab.")

    # Offer CSV download
    import io
    import pandas as pd
    df = pd.DataFrame(rows, columns=header)
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv_bytes, file_name="results.csv", mime="text/csv")
