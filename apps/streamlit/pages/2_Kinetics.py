import streamlit as st
from typing import List

st.title("Kinetics")

# Define species list
species_text = st.text_input("Species (comma-separated)", value="A,B")
species = [s.strip() for s in species_text.split(',') if s.strip()]

st.markdown("Define reactions as stoichiometry vectors aligned with the species order. Example for A->B: -1,1")

num_rxn = st.number_input("Number of reactions", min_value=1, max_value=10, value=1, step=1)

stoich_list: List[str] = []
A_list: List[float] = []
Ea_list: List[float] = []

for i in range(num_rxn):
    cols = st.columns([3,2,2])
    with cols[0]:
        sto = st.text_input(f"Stoich r{i+1}", value="-1,1" if i == 0 and len(species) >= 2 else ",".join(["0"]*len(species)))
    with cols[1]:
        A = st.number_input(f"A r{i+1}", value=1.0, step=0.1, format="%f")
    with cols[2]:
        Ea = st.number_input(f"Ea r{i+1} (J/mol)", value=0.0, step=1000.0, format="%f")
    stoich_list.append(sto)
    A_list.append(A)
    Ea_list.append(Ea)

if st.button("Save Kinetics"):
    st.session_state["kinetics"] = {
        "species": species,
        "stoich": stoich_list,
        "A": A_list,
        "Ea": Ea_list,
    }
    st.success("Kinetics saved. Proceed to Reactors tab.")
