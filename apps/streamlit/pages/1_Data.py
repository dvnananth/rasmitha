import io
import pandas as pd
import streamlit as st

st.title("Data")

st.markdown("Upload or load your stirred reactor CFD dataset and view summary statistics.")

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
path_input = st.text_input("Or path to CSV on disk", value="")

if uploaded is not None:
    content = uploaded.read()
    df = pd.read_csv(io.BytesIO(content))
    st.session_state["cfd_df"] = df
    st.success(f"Loaded {len(df)} rows from upload")
elif path_input:
    try:
        df = pd.read_csv(path_input)
        st.session_state["cfd_df"] = df
        st.success(f"Loaded {len(df)} rows from {path_input}")
    except Exception as e:
        st.error(str(e))

if "cfd_df" in st.session_state:
    df = st.session_state["cfd_df"]
    st.subheader("Preview")
    st.dataframe(df.head())
    st.subheader("Describe")
    st.write(df.describe(include='all'))
    st.subheader("Correlation (numeric)")
    st.write(df.corr(numeric_only=True))
