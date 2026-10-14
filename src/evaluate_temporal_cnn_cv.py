from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from temporal_cnn import TemporalCNN

SEED = 42
EPOCHS = 40
BATCH_SIZE = 12
OUTPUT_DIR = Path("outputs/inail_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X = np.load("data/processed/inail_lifting_sequences/sequences.npy").astype(np.float32)
metadata = pd.read_csv("data/processed/inail_lifting_sequences/metadata.csv")

eligible = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

y = metadata["li_condition"].to_numpy(dtype=np.int64) - 1
groups = metadata["subject_id"].to_numpy()

cv = GroupKFold(n_splits=5)
oof_predictions = np.full(len(y), -1, dtype=np.int64)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    # Reset seeds so the experiment is reproducible.
    random.seed(SEED + fold)
    np.random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)

    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_x = torch.from_numpy(X_train).transpose(1, 2).contiguous()
    test_x = torch.from_numpy(X_test).transpose(1, 2).contiguous()
    train_y = torch.from_numpy(y_train).long()

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + fold),
    )

    model = TemporalCNN(input_features=4, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()

    oof_predictions[test_idx] = predictions
    accuracy = accuracy_score(y_test, predictions)
    test_subjects = sorted(set(groups[test_idx]))
    fold_results.append((fold, test_subjects, len(test_idx), accuracy))
    print(f"Fold {fold}: subjects={test_subjects}, trials={len(test_idx)}, accuracy={accuracy:.3f}")

assert np.all(oof_predictions >= 0)

print("\n=== Grouped cross-validation results ===")
for fold, subjects, count, accuracy in fold_results:
    print(f"Fold {fold}: {accuracy * 100:.2f}%")

overall_accuracy = accuracy_score(y, oof_predictions)
print(f"\nOut-of-fold accuracy: {overall_accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(
    y, oof_predictions,
    labels=[0, 1, 2],
    target_names=["LI 1", "LI 2", "LI 3"],
    zero_division=0,
))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y, oof_predictions, labels=[0, 1, 2]))

results = metadata[["subject_id", "trial", "li_condition"]].copy()
results["predicted_li_condition"] = oof_predictions + 1
results["correct"] = results["li_condition"].to_numpy() == oof_predictions + 1
results.to_csv(OUTPUT_DIR / "group_cv_predictions.csv", index=False)
print("\nSaved out-of-fold predictions:", OUTPUT_DIR / "group_cv_predictions.csv")
print("Note: Small dataset; results are exploratory, not safe/unsafe validation.")
