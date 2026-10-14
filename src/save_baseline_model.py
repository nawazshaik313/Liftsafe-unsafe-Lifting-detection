from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed/inail_lifting_sequences")
OUTPUT_DIR = Path("outputs/inail_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_DIR / "sequences.npy").astype(np.float64)
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# Use the same ten subjects and feature extraction as cross-validation.
eligible = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

feature_names = [
    "trunk_inclination",
    "supplementary_hip",
    "knee_flexion",
    "trunk_angular_change",
]
stat_names = ["mean", "std", "min", "max"]

rows = []
for sequence in X:
    row = []
    for j in range(X.shape[2]):
        signal = sequence[:, j]
        row.extend([
            signal.mean(),
            signal.std(),
            signal.min(),
            signal.max(),
        ])
    rows.append(row)

X_summary = np.asarray(rows)
y = metadata["li_condition"].to_numpy(dtype=np.int64)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42),
)
model.fit(X_summary, y)

model_path = OUTPUT_DIR / "li_condition_logistic_regression.joblib"
joblib.dump(model, model_path)

metadata_path = OUTPUT_DIR / "li_condition_model_metadata.json"
model_metadata = {
    "task": "INAIL LI condition classification",
    "classes": {
        "1": "LI 1",
        "2": "LI 2",
        "3": "LI 3",
    },
    "input_shape_per_trial": [64, 4],
    "input_features": feature_names,
    "statistics_per_feature": stat_names,
    "summary_feature_count": int(X_summary.shape[1]),
    "training_subjects": eligible,
    "training_trials": int(len(y)),
    "cross_validation": {
        "method": "5-fold GroupKFold by subject",
        "out_of_fold_accuracy": 0.7667,
        "note": "Exploratory estimate; small dataset, not safe/unsafe validation.",
    },
    "warning": (
        "Predicts experimental LI condition, not a validated safe/unsafe label. "
        "This final artifact was fitted on all 90 eligible trials; do not use "
        "its training accuracy as independent test performance."
    ),
}
metadata_path.write_text(json.dumps(model_metadata, indent=2))

print("Saved pipeline:", model_path)
print("Saved metadata:", metadata_path)
print("Input sequences:", X.shape)
print("Summary feature matrix:", X_summary.shape)
print("Classes:", sorted(np.unique(y).tolist()))
print("Training trials:", len(y))
print("Cross-validation accuracy (separate evaluation): 76.67%")
print("Final model fit on all 90 trials; no independent test score computed.")
