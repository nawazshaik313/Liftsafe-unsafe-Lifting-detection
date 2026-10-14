
import numpy as np


class SequenceStandardizer:
    """Standardize sequence features using training data only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        X shape: (videos, time_steps, features).
        Fit only on the training split.
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "Expected shape (videos, time_steps, features)"
            )

        if not np.isfinite(X).all():
            raise ValueError("Input contains NaN or infinite values")

        self.mean_ = X.mean(axis=(0, 1))
        self.std_ = X.std(axis=(0, 1))

        # Prevent division by zero for constant features.
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)

        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X, dtype=np.float32)

        if X.shape[-1] != len(self.mean_):
            raise ValueError("Unexpected number of features")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    X = np.load("data/processed/cnn_sequences.npy")

    # Pipeline smoke test only. This is not a training/test split.
    # Do not use these statistics for model evaluation.
    scaler = SequenceStandardizer()
    X_scaled = scaler.fit_transform(X)

    print("Original shape:", X.shape)
    print("Scaled shape:", X_scaled.shape)
    print("All finite:", np.isfinite(X_scaled).all())
    print("Feature means:", X_scaled.mean(axis=(0, 1)).round(4))
    print("Feature stds:", X_scaled.std(axis=(0, 1)).round(4))
    print("Preprocessing smoke test passed.")
    print("No model training performed.")
