"""Feature summaries shared by the dashboard's trained LI models."""

import numpy as np


def summarize_for_logistic_regression(sequence):
    """Feature order used by the Logistic Regression training pipeline."""
    sequence = _validate_sequence(sequence)
    features = []
    for feature_index in range(sequence.shape[1]):
        signal = sequence[:, feature_index]
        features.extend((signal.mean(), signal.std(), signal.min(), signal.max()))
    return np.asarray(features, dtype=np.float64).reshape(1, -1)


def summarize_for_random_forest(sequence):
    """Feature order used by the Random Forest training pipeline."""
    sequence = _validate_sequence(sequence)
    return np.concatenate((
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.min(axis=0),
        sequence.max(axis=0),
    )).astype(np.float64).reshape(1, -1)


def compare_model_predictions(logistic_model, random_forest_model, sequence):
    """Predict with each model using its own verified 16-feature layout."""
    logistic_features = summarize_for_logistic_regression(sequence)
    forest_features = summarize_for_random_forest(sequence)
    _check_model_input(logistic_model, logistic_features, "Logistic Regression")
    _check_model_input(random_forest_model, forest_features, "Random Forest")
    _check_model_classes(logistic_model, "Logistic Regression")
    _check_model_classes(random_forest_model, "Random Forest")

    logistic_prediction = _predict_one(
        logistic_model, logistic_features, "Logistic Regression"
    )
    forest_prediction = _predict_one(
        random_forest_model, forest_features, "Random Forest"
    )
    return {
        "logistic_features": logistic_features,
        "random_forest_features": forest_features,
        "logistic_prediction": logistic_prediction,
        "random_forest_prediction": forest_prediction,
        "agree": logistic_prediction == forest_prediction,
    }


def _check_model_input(model, features, name):
    expected = getattr(model, "n_features_in_", features.shape[1])
    if features.shape != (1, 16) or expected != 16:
        raise ValueError(
            f"{name} expects {expected} features; this dashboard provides 16."
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{name} input contains non-finite features.")


def _check_model_classes(model, name):
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (3,) or set(classes.tolist()) != {1, 2, 3}:
        raise ValueError(f"{name} must provide all three LI classes (1, 2, 3).")


def _predict_one(model, features, name):
    prediction = np.asarray(model.predict(features))
    if prediction.shape != (1,):
        raise ValueError(f"{name} returned an invalid prediction shape.")
    try:
        value = int(prediction[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} returned an invalid LI class.") from exc
    if value not in (1, 2, 3) or value != prediction[0]:
        raise ValueError(f"{name} returned an LI class outside 1, 2, 3.")
    return value


def _validate_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] != 64 or sequence.shape[1] != 4:
        raise ValueError("Expected one lifting sequence with shape (64, 4).")
    if not np.isfinite(sequence).all():
        raise ValueError("Lifting sequence contains NaN or infinite values.")
    return sequence
