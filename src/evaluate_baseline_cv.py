from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.load("data/processed/inail_lifting_sequences/sequences.npy").astype(np.float64)
metadata = pd.read_csv("data/processed/inail_lifting_sequences/metadata.csv")

eligible = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

# Summarize each sequence into simple statistical features.
feature_names = [
    "trunk_inclination",
    "supplementary_hip",
    "knee_flexion",
    "trunk_angular_change",
]
summary_features = []

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
    summary_features.append(row)

X_summary = np.asarray(summary_features)
y = metadata["li_condition"].to_numpy(dtype=np.int64)
groups = metadata["subject_id"].to_numpy()

cv = GroupKFold(n_splits=5)
oof_predictions = np.full(len(y), -1, dtype=np.int64)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_summary, y, groups), start=1):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(X_summary[train_idx], y[train_idx])
    predictions = model.predict(X_summary[test_idx])
    oof_predictions[test_idx] = predictions
    accuracy = accuracy_score(y[test_idx], predictions)
    print(
        f"Fold {fold}: subjects="
        f"{sorted(set(groups[test_idx]))}, accuracy={accuracy * 100:.2f}%"
    )

print("\n=== Logistic regression baseline ===")
print(f"Out-of-fold accuracy: {accuracy_score(y, oof_predictions) * 100:.2f}%")
print(classification_report(
    y, oof_predictions,
    labels=[1, 2, 3],
    target_names=["LI 1", "LI 2", "LI 3"],
    zero_division=0,
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y, oof_predictions, labels=[1, 2, 3]))

output = Path("outputs/inail_cnn")
output.mkdir(parents=True, exist_ok=True)
result = metadata[["subject_id", "trial", "li_condition"]].copy()
result["predicted_li_condition"] = oof_predictions
result["correct"] = result["li_condition"].to_numpy() == oof_predictions
result.to_csv(output / "baseline_cv_predictions.csv", index=False)
print("Saved:", output / "baseline_cv_predictions.csv")
