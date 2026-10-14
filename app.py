from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LiftSafe | Posture Analysis",
    page_icon="🏋️",
    layout="wide",
)

DATA_DIR = Path("data/processed/inail_lifting_sequences")
MODEL_PATH = Path("outputs/inail_cnn/li_condition_logistic_regression.joblib")
RF_MODEL_PATH = Path("outputs/inail_cnn/li_condition_random_forest.joblib")
METADATA_PATH = Path("outputs/inail_cnn/li_condition_model_metadata.json")

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


def summarize_sequence(sequence):
    """Keep feature extraction identical to the saved baseline pipeline."""
    features = []
    for j in range(sequence.shape[1]):
        signal = sequence[:, j]
        features.extend([
            signal.mean(),
            signal.std(),
            signal.min(),
            signal.max(),
        ])
    return np.asarray(features, dtype=np.float64).reshape(1, -1)


def summarize_sequence_for_rf(sequence):
    """Match RF training order: all means, stds, mins, then maxes."""
    return np.concatenate([
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.min(axis=0),
        sequence.max(axis=0),
    ]).astype(np.float64).reshape(1, -1)


st.title("🏋️ LiftSafe")
st.subheader("Lifting Motion & Posture Analysis")
st.write(
    "Explore recorded lifting trials, examine posture-related measurements, "
    "and view the model's predicted INAIL lifting condition."
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

if len(sequences) != len(metadata):
    st.error("The sequence count does not match the metadata row count.")
    st.stop()

st.sidebar.header("Trial selection")
subject_options = sorted(metadata["subject_id"].unique())
subject = st.sidebar.selectbox("Subject", subject_options)

subject_rows = metadata[metadata["subject_id"] == subject]
trial_options = subject_rows["trial"].tolist()
trial_name = st.sidebar.selectbox("Lifting trial", trial_options)

matches = metadata.index[metadata["trial"] == trial_name].tolist()
if not matches:
    st.error("The selected trial was not found.")
    st.stop()

idx = matches[0]
row = metadata.iloc[idx]
sequence = sequences[idx]

features = summarize_sequence(sequence)
prediction = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]
rf_features = summarize_sequence_for_rf(sequence)
rf_prediction = int(rf_model.predict(rf_features)[0])
rf_probabilities = rf_model.predict_proba(rf_features)[0]

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Subject", row["subject_id"])
c2.metric("Actual LI condition", f"LI {int(row['li_condition'])}")
c3.metric("Logistic Regression", f"LI {prediction}")
c4.metric("Random Forest", f"LI {rf_prediction}")
c5.metric("Lifting duration", f"{row['lifting_duration_seconds']:.2f} s")

st.subheader("Model prediction comparison")
st.write(
    "These models predict experimental LI conditions, not safe/unsafe labels. "
    "Their scores are not calibrated safety probabilities."
)

left, right = st.columns(2)
with left:
    st.markdown("#### Logistic Regression")
    probability_frame = pd.DataFrame({
        "LI condition": [f"LI {int(c)}" for c in model.classes_],
        "Model score": probabilities,
    })
    st.bar_chart(
        probability_frame, x="LI condition", y="Model score", width="stretch"
    )

with right:
    st.markdown("#### Random Forest")
    rf_probability_frame = pd.DataFrame({
        "LI condition": [f"LI {int(c)}" for c in rf_model.classes_],
        "Model score": rf_probabilities,
    })
    st.bar_chart(
        rf_probability_frame, x="LI condition", y="Model score", width="stretch"
    )

st.subheader("Posture measurements over the lifting sequence")

# Plot angles together because they share the same unit: degrees.
angle_data = pd.DataFrame({
    FEATURE_NAMES[j]: sequence[:, j] for j in range(3)
})
angle_data.index.name = "Resampled time step (0–63)"

st.markdown("#### Trunk and joint angles (degrees)")
st.line_chart(angle_data, width="stretch")

# Plot angular change separately because its unit is degrees per second.
speed_data = pd.DataFrame({
    FEATURE_NAMES[3]: sequence[:, 3]
})
speed_data.index.name = "Resampled time step (0–63)"

st.markdown("#### Trunk angular change (degrees/second)")
st.line_chart(speed_data, width="stretch")

st.subheader("Trial details")
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

for class_id, probability in zip(model.classes_, probabilities):
    report[f"Probability LI {int(class_id)}"] = round(
        float(probability), 4
    )

st.download_button(
    label="Download trial report (CSV)",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"{row['subject_id']}_lifting_report.csv",
    mime="text/csv",
)

