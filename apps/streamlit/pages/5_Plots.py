import streamlit as st
import plotly.express as px
import pandas as pd

st.title("Plots")

if "results_rows" not in st.session_state or "results_header" not in st.session_state:
    st.info("Run a simulation in the Solver tab first.")
    st.stop()

rows = st.session_state["results_rows"]
header = st.session_state["results_header"]

df = pd.DataFrame(rows, columns=header)

x_col = header[0]
y_cols = [c for c in header[1:]]
sel = st.multiselect("Series to plot", y_cols, default=y_cols[:min(3, len(y_cols))])

fig = px.line(df, x=x_col, y=sel)
st.plotly_chart(fig, use_container_width=True)
