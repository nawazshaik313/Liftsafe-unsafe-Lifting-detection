from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

DATA_DIR = Path("data/processed/inail_lifting_sequences")
OUTPUT_DIR = Path("outputs/inail_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X = np.load(DATA_DIR / "sequences.npy").astype(np.float32)
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

eligible = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

# Four statistics per posture signal: mean, std, minimum, maximum.
features = np.concatenate(
    [
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
    ],
    axis=1,
)

y = metadata["li_condition"].to_numpy(dtype=np.int64)
groups = metadata["subject_id"].to_numpy()
predictions = np.full(len(y), -1, dtype=np.int64)

cv = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(features, y, groups), start=1
):
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features[train_idx], y[train_idx])
    predictions[test_idx] = model.predict(features[test_idx])

    fold_accuracy = accuracy_score(y[test_idx], predictions[test_idx])
    print(f"Fold {fold}: accuracy={fold_accuracy * 100:.2f}%")

assert np.all(predictions >= 0)

results = metadata[
    ["subject_id", "trial", "li_condition"]
].copy()
results["predicted_li_condition"] = predictions
results["correct"] = results["li_condition"].to_numpy() == predictions
results.to_csv(OUTPUT_DIR / "random_forest_cv_predictions.csv", index=False)

accuracy = accuracy_score(y, predictions)
macro_f1 = f1_score(y, predictions, average="macro", labels=[1, 2, 3])
report = classification_report(
    y,
    predictions,
    labels=[1, 2, 3],
    target_names=["LI1", "LI2", "LI3"],
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "random_forest_cv_report.csv"
)

print(f"\nOut-of-fold accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1:.3f}")
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y, predictions, labels=[1, 2, 3]))
print("Saved predictions:", OUTPUT_DIR / "random_forest_cv_predictions.csv")
print("Saved report:", OUTPUT_DIR / "random_forest_cv_report.csv")
print("Exploratory LI-condition classification, not safe/unsafe validation.")
