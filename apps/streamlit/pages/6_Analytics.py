import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from reactorsim.kinetics import Arrhenius, Reaction, Network
from reactorsim.analytics import generate_cstr_isothermal_dataset

st.title("Analytics: Physics vs ML vs Hybrid")

st.markdown("""
This page compares:
- Physics (high fidelity): slow but accurate (fine solver dt)
- Physics (low fidelity): fast but approximate (coarse solver dt)
- ML: trained on high-fidelity labels
- Hybrid: baseline low-fidelity corrected by ML on residuals (y_true - y_low)
""")

# Kinetics setup (simple A -> B first-order)
species = ["A", "B"]
rxn = Reaction(forward=Arrhenius(A=1.0, Ea=0.0), stoich={"A": -1.0, "B": 1.0})
net = Network(species=species, reactions=[rxn])

st.sidebar.header("Dataset options")
n_samples = st.sidebar.slider("Samples", min_value=50, max_value=2000, value=400, step=50)
t_end = st.sidebar.number_input("t_end (h)", value=10.0)
dt_high = st.sidebar.number_input("dt_high", value=0.01, format="%f")
dt_low = st.sidebar.number_input("dt_low", value=0.1, format="%f")
T_min, T_max = st.sidebar.number_input("T min", value=280.0), st.sidebar.number_input("T max", value=340.0)
tau_min, tau_max = st.sidebar.number_input("tau min", value=0.2), st.sidebar.number_input("tau max", value=5.0)
feed_min, feed_max = st.sidebar.number_input("c_in min", value=0.0), st.sidebar.number_input("c_in max", value=2.0)

if st.button("Generate and Train"):
    with st.spinner("Simulating datasets..."):
        ds = generate_cstr_isothermal_dataset(
            network=net,
            species_index=1,  # predict B
            T_range=(T_min, T_max),
            tau_range=(tau_min, tau_max),
            feed_ranges=[(feed_min, feed_max), (0.0, 0.0)],
            n_samples=n_samples,
            t_end_h=t_end,
            dt_high=dt_high,
            dt_low=dt_low,
            seed=42,
        )

    # Split
    X_train, X_test, y_true_train, y_true_test, y_low_train, y_low_test = train_test_split(
        ds.X, ds.y_true, ds.y_low, test_size=0.2, random_state=42
    )

    # Train ML on true
    t0 = time.perf_counter()
    ml = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    ml.fit(X_train, y_true_train)
    ml_train_time = time.perf_counter() - t0

    # Train hybrid on residuals
    resid_train = y_true_train - y_low_train
    t1 = time.perf_counter()
    hy = RandomForestRegressor(n_estimators=300, random_state=43, n_jobs=-1)
    hy.fit(X_train, resid_train)
    hy_train_time = time.perf_counter() - t1

    # Predict
    y_true_pred = ml.predict(X_test)
    y_low_pred = y_low_test  # baseline physics
    y_hybrid_pred = y_low_test + hy.predict(X_test)

    # Metrics
    def metrics(y, yhat, name):
        return {
            "name": name,
            "MAE": mean_absolute_error(y, yhat),
            "R2": r2_score(y, yhat),
        }

    rows = [
        {"name": "Physics (low)", "MAE": mean_absolute_error(y_true_test, y_low_pred), "R2": r2_score(y_true_test, y_low_pred)},
        metrics(y_true_test, y_true_pred, "ML"),
        metrics(y_true_test, y_hybrid_pred, "Hybrid (low + ML residual)"),
    ]

    perf = pd.DataFrame(rows)
    st.subheader("Accuracy comparison (test set)")
    st.dataframe(perf)

    st.subheader("Timing summary (seconds)")
    times = pd.DataFrame([
        {"what": "Physics (low) sim", "seconds": ds.low_pred_time_s},
        {"what": "Physics (high) sim", "seconds": ds.true_time_s},
        {"what": "ML train", "seconds": ml_train_time},
        {"what": "Hybrid train", "seconds": hy_train_time},
    ])
    st.dataframe(times)

    st.subheader("Parity plots")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.scatter(x=y_true_test, y=y_low_pred, labels={"x": "True", "y": "Low-phys pred"}, title="Low-phys vs True")
        fig.add_shape(type="line", x0=min(y_true_test), y0=min(y_true_test), x1=max(y_true_test), y1=max(y_true_test))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(x=y_true_test, y=y_true_pred, labels={"x": "True", "y": "ML pred"}, title="ML vs True")
        fig.add_shape(type="line", x0=min(y_true_test), y0=min(y_true_test), x1=max(y_true_test), y1=max(y_true_test))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.scatter(x=y_true_test, y=y_hybrid_pred, labels={"x": "True", "y": "Hybrid pred"}, title="Hybrid vs True")
        fig.add_shape(type="line", x0=min(y_true_test), y0=min(y_true_test), x1=max(y_true_test), y1=max(y_true_test))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature importance (ML)")
    imp = pd.DataFrame({"feature": ds.feature_names, "importance": ml.feature_importances_}).sort_values("importance", ascending=False)
    st.dataframe(imp)

    st.success("Done.")
