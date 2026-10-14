from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path("data/processed/inail_lifting_sequences")
OUTPUT_DIR = Path("outputs/inail_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_DIR / "sequences.npy").astype(np.float32)
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# Match the same eligible subjects used in cross-validation.
eligible = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

# Four summary statistics per feature: mean, std, minimum, maximum.
X_summary = np.concatenate(
    [
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
    ],
    axis=1,
)

y = metadata["li_condition"].to_numpy(dtype=np.int64)

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_summary, y)

model_path = OUTPUT_DIR / "li_condition_random_forest.joblib"
joblib.dump(model, model_path)

metadata_path = OUTPUT_DIR / "random_forest_model_metadata.json"
model_metadata = {
    "task": "INAIL LI condition classification",
    "classes": {
        "1": "LI 1",
        "2": "LI 2",
        "3": "LI 3",
    },
    "input_shape_per_trial": [64, 4],
    "summary_feature_count": int(X_summary.shape[1]),
    "feature_statistics_order": [
        "all feature means",
        "all feature standard deviations",
        "all feature minima",
        "all feature maxima",
    ],
    "training_subjects": eligible,
    "training_trials": int(len(y)),
    "model": {
        "type": "RandomForestClassifier",
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "cross_validation": {
        "method": "5-fold GroupKFold by subject",
        "out_of_fold_accuracy": 0.80,
        "macro_f1": 0.798,
        "note": (
            "Exploratory estimate from 90 trials and 10 subjects; "
            "not safe/unsafe validation."
        ),
    },
    "warning": (
        "Predicts experimental LI condition, not a validated safe/unsafe label. "
        "The final artifact is fitted on all 90 eligible trials. "
        "Its training accuracy is not independent test performance."
    ),
}
metadata_path.write_text(json.dumps(model_metadata, indent=2))

print("Saved model:", model_path)
print("Saved metadata:", metadata_path)
print("Summary feature matrix:", X_summary.shape)
print("Classes:", sorted(np.unique(y).tolist()))
print("Training trials:", len(y))
print("Separate subject-wise cross-validation accuracy: 80.00%")
print("Final model fitted on all 90 trials.")
