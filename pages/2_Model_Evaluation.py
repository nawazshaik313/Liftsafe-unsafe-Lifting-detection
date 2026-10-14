from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

st.set_page_config(
    page_title="LiftSafe | Model Evaluation",
    page_icon="📊",
    layout="wide",
)

OUTPUT_DIR = Path("outputs/inail_cnn")
MODEL_FILES = {
    "Temporal CNN": OUTPUT_DIR / "group_cv_predictions.csv",
    "Logistic Regression": OUTPUT_DIR / "baseline_cv_predictions.csv",
    "Random Forest": OUTPUT_DIR / "random_forest_cv_predictions.csv",
}

st.title("📊 Model Evaluation")
st.caption("Subject-wise 5-fold cross-validation on INAIL lifting trials")

st.warning(
    "These are exploratory results from a small dataset of 90 trials "
    "and 10 subjects. The models predict LI condition, not verified "
    "safe/unsafe posture."
)

missing = [name for name, path in MODEL_FILES.items() if not path.exists()]
if missing:
    st.error(
        "Missing evaluation files for: " + ", ".join(missing)
        + ". Run the corresponding cross-validation scripts first."
    )
    st.stop()

frames = {name: pd.read_csv(path) for name, path in MODEL_FILES.items()}

required = {
    "subject_id",
    "trial",
    "li_condition",
    "predicted_li_condition",
}

for name, frame in frames.items():
    if not required.issubset(frame.columns):
        st.error(f"{name} predictions are missing required columns.")
        st.stop()

    if frame[list(required)].isna().any().any():
        st.error(f"{name} predictions contain missing values.")
        st.stop()

reference_keys = frames["Temporal CNN"][
    ["subject_id", "trial", "li_condition"]
].sort_values(["subject_id", "trial"]).reset_index(drop=True)

for name, frame in frames.items():
    keys = frame[
        ["subject_id", "trial", "li_condition"]
    ].sort_values(["subject_id", "trial"]).reset_index(drop=True)

    if not keys.equals(reference_keys):
        st.error(f"{name} does not contain the same trial records as the CNN.")
        st.stop()

labels = [1, 2, 3]
class_names = ["LI 1", "LI 2", "LI 3"]
metrics = {}

for name, frame in frames.items():
    y_true = frame["li_condition"].astype(int)
    y_pred = frame["predicted_li_condition"].astype(int)

    metrics[name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "correct": int((y_true == y_pred).sum()),
    }

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trials evaluated", len(frames["Temporal CNN"]))
c2.metric("Temporal CNN", f'{metrics["Temporal CNN"]["accuracy"] * 100:.2f}%')
c3.metric(
    "Logistic Regression",
    f'{metrics["Logistic Regression"]["accuracy"] * 100:.2f}%'
)
c4.metric(
    "Random Forest",
    f'{metrics["Random Forest"]["accuracy"] * 100:.2f}%'
)

st.subheader("Out-of-fold model comparison")

comparison = pd.DataFrame([
    {
        "Model": name,
        "Accuracy (%)": values["accuracy"] * 100,
        "Macro F1": values["macro_f1"],
    }
    for name, values in metrics.items()
])

st.dataframe(
    comparison.sort_values("Accuracy (%)", ascending=False).round(3),
    hide_index=True,
    width="stretch",
)

st.bar_chart(
    comparison.set_index("Model")[["Accuracy (%)"]],
    width="stretch",
)

st.caption(
    "Each trial is predicted by a model trained without that trial's subject. "
    "Macro F1 gives equal weight to the three LI classes."
)

tabs = st.tabs(list(frames.keys()))

for tab, (name, frame) in zip(tabs, frames.items()):
    with tab:
        y_true = frame["li_condition"].astype(int)
        y_pred = frame["predicted_li_condition"].astype(int)

        st.markdown(f"### {name}: per-class performance")

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).T
        report_df = report_df.loc[
            ["LI 1", "LI 2", "LI 3", "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"],
        ]

        st.dataframe(report_df.round(3), width="stretch")

        st.markdown("### Confusion matrix")
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in class_names],
            columns=[f"Predicted {label}" for label in class_names],
        )
        st.dataframe(matrix_df, width="stretch")

        correct = int((y_true == y_pred).sum())
        st.write(f"Correct predictions: **{correct} / {len(frame)}**")

        st.download_button(
            f"Download {name} predictions (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{name.lower().replace(' ', '_')}_cv_predictions.csv",
            mime="text/csv",
            key=f"download_{name.lower().replace(' ', '_')}_predictions",
        )

st.subheader("Interpretation")
st.markdown(
    """
    - Random Forest currently has the highest exploratory accuracy among
      these three models on the saved out-of-fold predictions.
    - LI 1 and LI 2 are often confused with one another.
    - LI 3 was classified consistently in these experiments.
    - Results from 10 subjects may not generalize to new people or real-time
      camera footage. Further validation is required.
    """
)

st.info(
    "These models use features extracted from recorded motion-capture data. "
    "They do not yet classify live camera footage, and LI conditions are not "
    "equivalent to validated safe/unsafe labels."
)
