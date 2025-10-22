import streamlit as st
import numpy as np

st.title("Reactors")

if "kinetics" not in st.session_state:
    st.warning("Define kinetics first in the Kinetics tab.")
    st.stop()

kin = st.session_state["kinetics"]
species = kin["species"]

reactor_type = st.selectbox("Reactor type", ["Batch (isothermal)", "CSTR (isothermal)", "PFR (isothermal)", "Batch (adiabatic)", "CSTR (adiabatic)"])

params = {}
if reactor_type == "Batch (isothermal)":
    params["T"] = st.number_input("Temperature (K)", value=300.0)
    params["c0"] = [st.number_input(f"c0 {s}", value=1.0 if i == 0 else 0.0) for i, s in enumerate(species)]
    params["tend"] = st.number_input("End time (h)", value=5.0)
    params["dt"] = st.number_input("dt (h)", value=0.1)
elif reactor_type == "CSTR (isothermal)":
    params["T"] = st.number_input("Temperature (K)", value=300.0)
    params["c0"] = [st.number_input(f"Initial {s}", value=0.0) for s in species]
    params["cin"] = [st.number_input(f"Feed {s}", value=1.0 if i == 0 else 0.0) for i, s in enumerate(species)]
    params["tau"] = st.number_input("Residence time tau (h)", value=1.0)
    params["tend"] = st.number_input("End time (h)", value=10.0)
    params["dt"] = st.number_input("dt (h)", value=0.05)
elif reactor_type == "PFR (isothermal)":
    params["T"] = st.number_input("Temperature (K)", value=300.0)
    params["cin"] = [st.number_input(f"Inlet {s}", value=1.0 if i == 0 else 0.0) for i, s in enumerate(species)]
    params["tau_end"] = st.number_input("Tau end (h)", value=3.0)
    params["dt"] = st.number_input("dt (h)", value=0.05)
elif reactor_type == "Batch (adiabatic)":
    params["T0"] = st.number_input("Initial Temperature (K)", value=300.0)
    params["c0"] = [st.number_input(f"c0 {s}", value=1.0 if i == 0 else 0.0) for i, s in enumerate(species)]
    params["rho_cp"] = st.number_input("rho*Cp (J/L-K)", value=4000.0)
    params["dH"] = [st.number_input(f"dH reaction {i+1} (J/mol)", value=-50000.0) for i in range(len(kin["stoich"]))]
    params["tend"] = st.number_input("End time (h)", value=2.0)
    params["dt"] = st.number_input("dt (h)", value=0.02)
elif reactor_type == "CSTR (adiabatic)":
    params["Tin"] = st.number_input("Inlet Temperature (K)", value=300.0)
    params["c0"] = [st.number_input(f"Initial {s}", value=0.0) for s in species]
    params["cin"] = [st.number_input(f"Feed {s}", value=1.0 if i == 0 else 0.0) for i, s in enumerate(species)]
    params["tau"] = st.number_input("Residence time tau (h)", value=1.0)
    params["rho_cp"] = st.number_input("rho*Cp (J/L-K)", value=4000.0)
    params["dH"] = [st.number_input(f"dH reaction {i+1} (J/mol)", value=-50000.0) for i in range(len(kin["stoich"]))]
    params["tend"] = st.number_input("End time (h)", value=5.0)
    params["dt"] = st.number_input("dt (h)", value=0.05)

if st.button("Save Reactor Setup"):
    st.session_state["reactor_setup"] = {"type": reactor_type, "params": params}
    st.success("Reactor setup saved. Proceed to Solver tab.")
