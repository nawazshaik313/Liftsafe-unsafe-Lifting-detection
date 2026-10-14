import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset

from temporal_cnn import TemporalCNN


SEED = 42
EPOCHS = 60
BATCH_SIZE = 12
LEARNING_RATE = 0.001
TEST_SUBJECTS = ["S03", "S07"]

DATA_DIR = Path("data/processed/inail_lifting_sequences")
OUTPUT_DIR = Path("outputs/inail_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load sequences: (trials, time steps, features)
X = np.load(DATA_DIR / "sequences.npy").astype(np.float32)
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# Use only S01-S10, who have all three LI conditions.
eligible_subjects = [f"S{i:02d}" for i in range(1, 11)]
keep = metadata["subject_id"].isin(eligible_subjects).to_numpy()
X = X[keep]
metadata = metadata.loc[keep].reset_index(drop=True)

# Split by subject, never by individual trial.
test_mask = metadata["subject_id"].isin(TEST_SUBJECTS).to_numpy()
train_mask = ~test_mask

X_train = X[train_mask]
X_test = X[test_mask]

y_all = metadata["li_condition"].to_numpy(dtype=np.int64) - 1
y_train = y_all[train_mask]
y_test = y_all[test_mask]

train_subjects = set(metadata.loc[train_mask, "subject_id"])
test_subjects = set(metadata.loc[test_mask, "subject_id"])

assert not (train_subjects & test_subjects), "Subject leakage detected!"
assert set(np.unique(y_train)) == {0, 1, 2}
assert set(np.unique(y_test)) == {0, 1, 2}
assert np.isfinite(X_train).all() and np.isfinite(X_test).all()

# Normalize each feature using training data only.
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)
std[std < 1e-6] = 1.0

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Save normalization values for later inference.
np.savez(OUTPUT_DIR / "normalization.npz", mean=mean, std=std)

# CNN expects (batch, features, time steps).
train_x = torch.from_numpy(X_train).transpose(1, 2).contiguous()
test_x = torch.from_numpy(X_test).transpose(1, 2).contiguous()
train_y = torch.from_numpy(y_train).long()
test_y = torch.from_numpy(y_test).long()

generator = torch.Generator().manual_seed(SEED)
loader = DataLoader(
    TensorDataset(train_x, train_y),
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

model = TemporalCNN(input_features=4, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Device:", device)
print("Sequences:", X.shape)
print("Training trials:", len(y_train), "| subjects:", sorted(train_subjects))
print("Testing trials:", len(y_test), "| subjects:", sorted(test_subjects))
print("Training class counts:", np.bincount(y_train, minlength=3).tolist())
print("Testing class counts:", np.bincount(y_test, minlength=3).tolist())
print("Training CNN...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(batch_y)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| loss: {running_loss / len(train_y):.4f}"
        )

model.eval()
with torch.no_grad():
    logits = model(test_x.to(device))
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)

accuracy = accuracy_score(y_test, predictions)
labels = [0, 1, 2]

print("\n=== Held-out test results ===")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report (classes correspond to LI 1, LI 2, LI 3):")
print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=["LI 1", "LI 2", "LI 3"],
        zero_division=0,
    )
)
print("Confusion matrix (rows=true, columns=predicted; LI 1, LI 2, LI 3):")
print(confusion_matrix(y_test, predictions, labels=labels))

# Save weights and reproducibility details.
torch.save(model.state_dict(), OUTPUT_DIR / "temporal_cnn_li3.pt")

results = metadata.loc[test_mask, ["subject_id", "trial", "li_condition"]].copy()
results["predicted_li_condition"] = predictions + 1
results["correct"] = results["li_condition"].to_numpy() == predictions + 1
for i, name in enumerate(["probability_li1", "probability_li2", "probability_li3"]):
    results[name] = probabilities[:, i]
results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

print("\nSaved model:", OUTPUT_DIR / "temporal_cnn_li3.pt")
print("Saved test predictions:", OUTPUT_DIR / "test_predictions.csv")
print("Saved normalization:", OUTPUT_DIR / "normalization.npz")
print("\nNote: This predicts experimental LI condition, not safe/unsafe posture.")
print("The test set contains only two subjects; results are preliminary.")
