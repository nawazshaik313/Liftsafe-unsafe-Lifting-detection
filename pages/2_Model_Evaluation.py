from pathlib import Path

import pandas as pd
import streamlit as st

from src.charts import confusion_matrix_figure, model_metric_figure, render_glass_chart
from src.ui import inject_app_styles, render_page_header, render_section_heading, render_sidebar_brand, render_theme_toggle
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
render_sidebar_brand()
render_theme_toggle()
inject_app_styles()

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs/inail_cnn"
MODEL_FILES = {
    "Temporal CNN": OUTPUT_DIR / "group_cv_predictions.csv",
    "Logistic Regression": OUTPUT_DIR / "baseline_cv_predictions.csv",
    "Random Forest": OUTPUT_DIR / "random_forest_cv_predictions.csv",
}
HOLDOUT_FILE = OUTPUT_DIR / "test_predictions.csv"

render_page_header(
    "Model evaluation",
    "Subject-wise cross-validation results from recorded INAIL lifting trials.",
    eyebrow="LIFTSAFE  /  VALIDATION",
)

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
c2.metric("Subjects", frames["Temporal CNN"]["subject_id"].nunique())
c3.metric("Random Forest accuracy", f'{metrics["Random Forest"]["accuracy"] * 100:.2f}%')
c4.metric("Random Forest macro F1", f'{metrics["Random Forest"]["macro_f1"]:.3f}')

render_section_heading("Out-of-fold model comparison", "Accuracy and macro F1 across three condition classes")

comparison = pd.DataFrame([
    {
        "Model": name,
        "Accuracy (%)": values["accuracy"] * 100,
        "Macro F1": values["macro_f1"],
    }
    for name, values in metrics.items()
])

with st.container(border=True):
    st.dataframe(
        comparison.sort_values("Accuracy (%)", ascending=False).round(3),
        hide_index=True,
        width="stretch",
    )

accuracy_chart, f1_chart = st.columns(2)
with accuracy_chart:
    st.markdown("#### Accuracy")
    render_glass_chart(
        model_metric_figure(
            comparison["Model"].tolist(),
            comparison["Accuracy (%)"].tolist(),
            "Accuracy (%)",
            percent=True,
        ),
        key="evaluation_accuracy",
    )
with f1_chart:
    st.markdown("#### Macro F1")
    render_glass_chart(
        model_metric_figure(
            comparison["Model"].tolist(),
            comparison["Macro F1"].tolist(),
            "Macro F1",
        ),
        key="evaluation_macro_f1",
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

        with st.container(border=True):
            st.dataframe(report_df.round(3), width="stretch")

        render_section_heading("Confusion matrix", "Rows are actual classes · columns are model predictions")
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in class_names],
            columns=[f"Predicted {label}" for label in class_names],
        )
        render_glass_chart(
            confusion_matrix_figure(matrix, class_names),
            key=f"confusion_matrix_{name.lower().replace(' ', '_')}",
        )
        with st.expander("View confusion matrix values"):
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

render_section_heading("Interpretation", "What these saved predictions do and do not show")
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

if HOLDOUT_FILE.exists():
    holdout = pd.read_csv(HOLDOUT_FILE)
    if not required.issubset(holdout.columns) or holdout.empty:
        st.warning("The separate CNN holdout prediction file is incomplete and was not summarized.")
    else:
        holdout_true = holdout["li_condition"].astype(int)
        holdout_pred = holdout["predicted_li_condition"].astype(int)
        holdout_accuracy = accuracy_score(holdout_true, holdout_pred)
        holdout_f1 = f1_score(
            holdout_true, holdout_pred, labels=labels, average="macro", zero_division=0
        )

        render_section_heading(
            "Separate Temporal CNN holdout",
            "Small held-out subject set · shown separately from out-of-fold results",
        )
        st.warning(
            "This result uses a different evaluation split and only a few held-out "
            "subjects. Do not compare it directly with the 90-trial out-of-fold scores."
        )
        holdout_columns = st.columns(4)
        holdout_columns[0].metric("Holdout trials", len(holdout))
        holdout_columns[1].metric("Held-out subjects", holdout["subject_id"].nunique())
        holdout_columns[2].metric("Holdout accuracy", f"{holdout_accuracy * 100:.2f}%")
        holdout_columns[3].metric("Holdout macro F1", f"{holdout_f1:.3f}")
        holdout_matrix = confusion_matrix(holdout_true, holdout_pred, labels=labels)
        render_glass_chart(
            confusion_matrix_figure(holdout_matrix, class_names),
            key="confusion_matrix_temporal_cnn_holdout",
        )
        st.download_button(
            "Download separate holdout predictions (CSV)",
            data=holdout.to_csv(index=False).encode("utf-8"),
            file_name="temporal_cnn_holdout_predictions.csv",
            mime="text/csv",
            key="download_temporal_cnn_holdout_predictions",
        )
