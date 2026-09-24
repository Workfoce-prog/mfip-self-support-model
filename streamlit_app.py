"""Streamlit front end for the MFIP combined research prototype."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from mfip_model import run

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="MFIP Self Support Model", layout="wide")
st.title("MFIP Self Support Model")
st.caption("Research prototype. Example data are synthetic; results are not official HSPM determinations.")

with st.expander("How the three components work", expanded=True):
    st.markdown(
        "**1. Hierarchical risk adjustment:** Compare observed Self Support Index with a "
        "service area expected range.\n\n"
        "**3. Multiple outcomes:** Review cash exit, employment, earnings, and return to assistance separately.\n\n"
        "**4. Absolute goal:** Display whether the observed Self Support Index reaches a chosen goal."
    )

uploaded = st.file_uploader("Upload adult-quarter CSV to run the model", type="csv")
goal = st.slider("Illustrative absolute goal", min_value=0.30, max_value=0.95,
                 value=0.70, step=0.01, format="%0.2f")
st.caption("The goal is a demonstration parameter. It does not change HSPM PIP rules.")
st.caption("Use synthetic data for public demos. Upload participant data only in an approved, access-controlled deployment.")

if uploaded is None:
    st.info("Showing precomputed results from synthetic example data. Upload a CSV and select Run model to calculate new results.")
    area = pd.read_csv(ROOT / "results" / "area_summary.csv")
    validation = pd.read_csv(ROOT / "results" / "validation.csv")
    diagnostics = json.loads((ROOT / "results" / "model_diagnostics.json").read_text())
else:
    st.caption("Your uploaded file is processed only when you select Run model.")
    if st.button("Run model", type="primary"):
        try:
            data = pd.read_csv(uploaded)
            with st.spinner("Fitting outcome models and calculating area results"):
                with TemporaryDirectory() as temp:
                    folder = Path(temp)
                    run(data, goal, folder)
                    st.session_state["area"] = pd.read_csv(folder / "area_summary.csv")
                    st.session_state["validation"] = pd.read_csv(folder / "validation.csv")
                    st.session_state["diagnostics"] = json.loads((folder / "model_diagnostics.json").read_text())
                    st.session_state["upload_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Could not run model: {exc}")
    if st.session_state.get("upload_name") != uploaded.name or "area" not in st.session_state:
        st.stop()
    area = st.session_state["area"]
    validation = st.session_state["validation"]
    diagnostics = st.session_state["diagnostics"]

area = area.copy()
area["absolute_goal"] = goal
area["goal_result"] = area["observed_ssi"].ge(goal).map({True: "Met", False: "Below"})
st.subheader("Service area results")
st.dataframe(area, use_container_width=True, hide_index=True)
st.download_button("Download area results", area.to_csv(index=False),
                   file_name="mfip_area_results.csv", mime="text/csv")

st.subheader("Three year Self Support Index")
plot = area.set_index("service_area")[["observed_ssi", "expected_ssi", "absolute_goal"]]
st.bar_chart(plot)
st.caption("Expected lower and upper limits, along with the classification, appear in the service area table.")

st.subheader("Temporal holdout validation")
st.dataframe(validation, use_container_width=True, hide_index=True)
st.caption(f"Latest baseline year held out: {diagnostics['holdout_cohort_year']}. "
           f"Training records: {diagnostics['train_rows']:,}; holdout records: {diagnostics['holdout_rows']:,}.")
st.warning(diagnostics["interval_warning"])
with st.expander("Input data and definitions"):
    st.markdown((ROOT / "README.md").read_text().split("## Model and interpretation")[0])
