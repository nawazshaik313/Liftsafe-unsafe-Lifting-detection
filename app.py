from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.charts import bar_figure, line_figure, render_glass_chart
from src.lifting_inference import compare_model_predictions
from src.ui import (
    inject_app_styles,
    render_page_header,
    render_section_heading,
    render_sidebar_brand,
    render_theme_toggle,
)

st.set_page_config(
    page_title="LiftSafe | Posture Analysis",
    page_icon="🏋️",
    layout="wide",
)
render_sidebar_brand()
render_theme_toggle()
inject_app_styles()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data/processed/inail_lifting_sequences"
MODEL_PATH = ROOT_DIR / "outputs/inail_cnn/li_condition_logistic_regression.joblib"
RF_MODEL_PATH = ROOT_DIR / "outputs/inail_cnn/li_condition_random_forest.joblib"
METADATA_PATH = ROOT_DIR / "outputs/inail_cnn/li_condition_model_metadata.json"

FEATURE_NAMES = [
    "Trunk inclination (°)",
    "Supplementary hip angle (°)",
    "Knee flexion (°)",
    "Trunk angular change (°/s)",
]
FEATURE_KEYS = [
    "Trunk inclination",
    "Supplementary hip angle",
    "Knee flexion",
    "Trunk angular change",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_rf_model():
    return joblib.load(RF_MODEL_PATH)


@st.cache_data
def load_data():
    sequences = np.load(DATA_DIR / "sequences.npy")
    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    return sequences, metadata


render_page_header(
    "Lifting motion, made clearer",
    "Review recorded trials, posture measurements, and experimental model scores.",
    eyebrow="LIFTSAFE  /  OVERVIEW",
)

st.warning(
    "**Research prototype:** The model predicts experimental LI conditions "
    "(LI 1, LI 2, LI 3). These are not verified safe/unsafe posture labels, "
    "and the model is not a certified ergonomic or medical assessment tool."
)

if not DATA_DIR.exists() or not MODEL_PATH.exists() or not RF_MODEL_PATH.exists():
    st.error(
        "Required data or model files are missing. Run the dataset preparation "
        "and baseline-model scripts first."
    )
    st.stop()

try:
    sequences, metadata = load_data()
    model = load_model()
    rf_model = load_rf_model()
except Exception as exc:
    st.error(f"Could not load the project data or model: {exc}")
    st.stop()

if sequences.ndim != 3 or sequences.shape[1:] != (64, 4):
    st.error(f"Expected sequences shaped (trials, 64, 4); found {sequences.shape}.")
    st.stop()

if len(sequences) != len(metadata):
    st.error("The sequence count does not match the metadata row count.")
    st.stop()

render_section_heading("Project status", "Saved assets and dataset snapshot available locally")
status_cols = st.columns(4)
status_cols[0].metric("Recorded trials", len(metadata))
status_cols[1].metric("Subjects", metadata["subject_id"].nunique())
status_cols[2].metric("Saved classifiers", "2 ready")
status_cols[3].metric("Sequence input", f"{sequences.shape[1]} × {sequences.shape[2]}")

required_metadata = {
    "subject_id", "trial", "li_condition", "lifting_duration_seconds"
}
if not required_metadata.issubset(metadata.columns):
    st.error("The lifting metadata is missing required trial fields.")
    st.stop()

with st.sidebar:
    with st.container(border=True):
        st.header("Trial selection")
        subject_options = sorted(metadata["subject_id"].unique())
        subject = st.selectbox("Subject", subject_options)

        subject_rows = metadata[metadata["subject_id"] == subject]
        trial_options = subject_rows["trial"].tolist()
        trial_name = st.selectbox("Lifting trial", trial_options)

matches = subject_rows.index[subject_rows["trial"] == trial_name].tolist()
if not matches:
    st.error("The selected trial was not found.")
    st.stop()

idx = matches[0]
row = metadata.iloc[idx]
sequence = sequences[idx]

try:
    comparison = compare_model_predictions(model, rf_model, sequence)
    prediction = comparison["logistic_prediction"]
    rf_prediction = comparison["random_forest_prediction"]
    probabilities = model.predict_proba(comparison["logistic_features"])[0]
    rf_probabilities = rf_model.predict_proba(
        comparison["random_forest_features"]
    )[0]
except Exception as exc:
    st.error(f"The selected trial could not be scored by both models: {exc}")
    st.stop()

if (
    probabilities.shape != (len(model.classes_),)
    or rf_probabilities.shape != (len(rf_model.classes_),)
    or not np.isfinite(probabilities).all()
    or not np.isfinite(rf_probabilities).all()
    or (probabilities < 0).any()
    or (probabilities > 1).any()
    or (rf_probabilities < 0).any()
    or (rf_probabilities > 1).any()
    or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)
    or not np.isclose(rf_probabilities.sum(), 1.0, atol=1e-4)
):
    st.error("A saved model returned invalid scores for its class labels.")
    st.stop()

render_section_heading("Selected recorded trial", f"{subject} · {trial_name}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Subject", row["subject_id"])
c2.metric("Actual LI condition", f"LI {int(row['li_condition'])}")
c3.metric("Logistic Regression", f"LI {prediction}")
c4.metric("Random Forest", f"LI {rf_prediction}")
c5.metric("Lifting duration", f"{row['lifting_duration_seconds']:.2f} s")

render_section_heading(
    "Model scores",
    "Experimental LI-condition predictions · scores are not calibrated safety probabilities.",
)

if comparison["agree"]:
    st.success(
        f"Model agreement: both predict LI {prediction}. Agreement is not "
        "evidence that the prediction is correct or that the lift is safe."
    )
else:
    st.warning(
        f"Model disagreement: Logistic Regression predicts LI {prediction}; "
        f"Random Forest predicts LI {rf_prediction}. Review the trial and "
        "measurements; neither prediction establishes safety."
    )

training_subjects = {f"S{i:02d}" for i in range(1, 11)}
if row["subject_id"] in training_subjects:
    st.caption(
        "This subject was included in the final models' training data. "
        "The selected-trial prediction is not a held-out evaluation."
    )
else:
    st.caption(
        "This subject is outside the saved models' training subjects; this "
        "single-trial prediction is still exploratory, not an evaluation."
    )

left, right = st.columns(2)
with left:
    render_section_heading("Logistic Regression", "Experimental LI-condition model scores")
    probability_frame = pd.DataFrame({
        "LI condition": [f"LI {int(c)}" for c in model.classes_],
        "Model score": probabilities,
    })
    render_glass_chart(
        bar_figure(
            probability_frame["LI condition"].tolist(),
            probability_frame["Model score"].tolist(),
            name="Model score",
            y_title="Model score (not calibrated probability)",
        ),
        key="dashboard_logistic_scores",
    )

with right:
    render_section_heading("Random Forest", "Experimental LI-condition model scores")
    rf_probability_frame = pd.DataFrame({
        "LI condition": [f"LI {int(c)}" for c in rf_model.classes_],
        "Model score": rf_probabilities,
    })
    render_glass_chart(
        bar_figure(
            rf_probability_frame["LI condition"].tolist(),
            rf_probability_frame["Model score"].tolist(),
            name="Model score",
            y_title="Model score (not calibrated probability)",
        ),
        key="dashboard_forest_scores",
    )

render_section_heading("Posture measurements", "Recorded motion-capture sequence · 64 resampled steps")

# Plot angles together because they share the same unit: degrees.
angle_data = pd.DataFrame({
    FEATURE_NAMES[j]: sequence[:, j] for j in range(3)
})
angle_data.index.name = "Resampled time step (0–63)"

render_section_heading("Trunk and joint angles", "Degrees · three motion features")
render_glass_chart(
    line_figure(
        angle_data,
        None,
        FEATURE_NAMES[:3],
        x_title="Resampled sequence step",
        y_title="Angle (degrees)",
    ),
    key="dashboard_motion_angles",
)

# Plot angular change separately because its unit is degrees per second.
speed_data = pd.DataFrame({
    FEATURE_NAMES[3]: sequence[:, 3]
})
speed_data.index.name = "Resampled time step (0–63)"

render_section_heading("Trunk angular change", "Degrees per second")
render_glass_chart(
    line_figure(
        speed_data,
        None,
        [FEATURE_NAMES[3]],
        x_title="Resampled sequence step",
        y_title="Angular change (°/s)",
        height=280,
    ),
    key="dashboard_trunk_velocity",
)

render_section_heading("Trial details", "Summary of the selected sequence")
details = pd.DataFrame({
    "Measurement": FEATURE_NAMES,
    "Mean": [sequence[:, j].mean() for j in range(4)],
    "Minimum": [sequence[:, j].min() for j in range(4)],
    "Maximum": [sequence[:, j].max() for j in range(4)],
})
for col in ["Mean", "Minimum", "Maximum"]:
    details[col] = details[col].round(2)
st.dataframe(details, hide_index=True, width="stretch")

with st.expander("About this model"):
    st.markdown(
        """
        - **Dataset:** INAIL lifting and carrying motion-capture trials.
        - **Input:** 64 time steps and four geometric features per trial.
        - **Model:** Logistic regression on 16 summary statistics.
        - **Cross-validation:** 76.67% out-of-fold accuracy using five
          subject-wise folds over ten subjects.
        - **Limitations:** The dataset is small; the validation result is
          exploratory. The model predicts LI condition, not safe versus unsafe
          posture. Supplementary hip and knee angles are geometric proxies.
        """
    )

st.caption(
    "LiftSafe research prototype · Predictions should not replace a qualified "
    "ergonomic assessment."
)

# Download the selected trial's analysis report.
report = details.copy()
report.insert(0, "Subject", row["subject_id"])
report.insert(1, "Trial", row["trial"])
report["Actual LI condition"] = int(row["li_condition"])
report["Predicted LI condition"] = prediction
report["Random Forest predicted LI condition"] = rf_prediction
report["Models agree"] = comparison["agree"]

for class_id, score in zip(model.classes_, probabilities):
    report[f"Logistic Regression score LI {int(class_id)}"] = round(
        float(score), 4
    )
for class_id, score in zip(rf_model.classes_, rf_probabilities):
    report[f"Random Forest score LI {int(class_id)}"] = round(float(score), 4)

st.download_button(
    label="Download trial report (CSV)",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"{row['subject_id']}_lifting_report.csv",
    mime="text/csv",
)

