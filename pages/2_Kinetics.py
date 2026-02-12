from typing import List

import streamlit as st

try:
    st.set_page_config(page_title="ReactorSim - Kinetics", page_icon="🧪", layout="wide")
except Exception:
    pass

st.title("Kinetics")
st.caption("Define species and elementary Arrhenius reactions.")

existing = st.session_state.get("kinetics")
init_species = ", ".join(existing["species"]) if existing else "A, B"
init_rxn = len(existing["stoich"]) if existing else 1
init_A = ", ".join(map(str, existing.get("A", []))) if existing else "1.0"
init_Ea = ", ".join(map(str, existing.get("Ea", []))) if existing else "0.0"
init_stoich_text = "\n".join(existing.get("stoich", [])) if existing else "-1, 1"

species_text = st.text_input("Species (comma-separated)", value=init_species)
num_rxn = st.number_input("Number of reactions", min_value=1, step=1, value=int(init_rxn))
stoich_text = st.text_area(
    "Stoichiometry per reaction (one line per reaction; entries per species)",
    value=init_stoich_text,
    height=120,
    help="Example for A -> B with species A,B: '-1, 1'",
)
A_text = st.text_input("Arrhenius A (per reaction, comma-separated)", value=init_A)
Ea_text = st.text_input("Activation energies Ea [J/mol] (per reaction, comma-separated)", value=init_Ea)

col1, col2 = st.columns(2)
with col1:
    if st.button("Save kinetics", use_container_width=True):
        try:
            species = [s.strip() for s in species_text.split(",") if s.strip()]
            lines = [ln.strip() for ln in stoich_text.strip().splitlines() if ln.strip()]
            # Ensure count matches num_rxn
            if len(lines) < num_rxn:
                # pad with zeros
                while len(lines) < num_rxn:
                    lines.append(", ".join(["0"] * len(species)))
            elif len(lines) > num_rxn:
                lines = lines[: num_rxn]
            A_vals = [float(x) for x in A_text.split(",") if x.strip()]
            Ea_vals = [float(x) for x in Ea_text.split(",") if x.strip()]
            if len(A_vals) != num_rxn or len(Ea_vals) != num_rxn:
                st.error("Lengths of A and Ea must equal number of reactions")
            else:
                st.session_state["kinetics"] = {
                    "species": species,
                    "stoich": lines,
                    "A": A_vals,
                    "Ea": Ea_vals,
                }
                st.success("Kinetics saved.")
        except Exception as e:
            st.exception(e)

with col2:
    if st.button("Load example: A → B (1st order)", use_container_width=True):
        st.session_state["kinetics"] = {
            "species": ["A", "B"],
            "stoich": ["-1, 1"],
            "A": [1.0],
            "Ea": [0.0],
        }
        st.success("Example kinetics loaded.")

st.divider()

st.subheader("Current kinetics")
st.write(st.session_state.get("kinetics"))
