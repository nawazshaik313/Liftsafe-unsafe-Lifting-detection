import unittest

import numpy as np

from src.lifting_inference import (
    compare_model_predictions,
    summarize_for_logistic_regression,
    summarize_for_random_forest,
)


class DummyModel:
    n_features_in_ = 16
    classes_ = np.array([1, 2, 3])

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        assert features.shape == (1, 16)
        return np.array([self.prediction])


class LiftingInferenceTests(unittest.TestCase):
    def setUp(self):
        self.sequence = np.arange(64 * 4, dtype=np.float64).reshape(64, 4)

    def test_model_specific_feature_orders(self):
        logistic = summarize_for_logistic_regression(self.sequence)
        forest = summarize_for_random_forest(self.sequence)
        self.assertEqual(logistic.shape, (1, 16))
        self.assertEqual(forest.shape, (1, 16))
        self.assertFalse(np.array_equal(logistic, forest))
        self.assertEqual(logistic[0, 0], self.sequence[:, 0].mean())
        self.assertEqual(forest[0, 0], self.sequence[:, 0].mean())
        self.assertEqual(forest[0, 4], self.sequence[:, 0].std())

    def test_agreement_and_disagreement_are_computed(self):
        agreement = compare_model_predictions(
            DummyModel(2), DummyModel(2), self.sequence
        )
        disagreement = compare_model_predictions(
            DummyModel(1), DummyModel(3), self.sequence
        )
        self.assertTrue(agreement["agree"])
        self.assertFalse(disagreement["agree"])
        self.assertEqual(disagreement["logistic_prediction"], 1)
        self.assertEqual(disagreement["random_forest_prediction"], 3)

    def test_invalid_shape_or_model_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_for_logistic_regression(self.sequence[:10])

        incompatible = DummyModel(1)
        incompatible.n_features_in_ = 8
        with self.assertRaisesRegex(ValueError, "expects 8 features"):
            compare_model_predictions(incompatible, DummyModel(1), self.sequence)

    def test_non_finite_sequence_and_missing_model_class_are_rejected(self):
        non_finite = self.sequence.copy()
        non_finite[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            compare_model_predictions(
                DummyModel(1), DummyModel(1), non_finite
            )

        missing_class = DummyModel(1)
        missing_class.classes_ = np.array([1, 2])
        with self.assertRaisesRegex(ValueError, "all three LI classes"):
            compare_model_predictions(
                missing_class, DummyModel(1), self.sequence
            )

    def test_invalid_prediction_and_model_exception_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside 1, 2, 3"):
            compare_model_predictions(
                DummyModel(4), DummyModel(1), self.sequence
            )

        class FailingModel(DummyModel):
            def predict(self, _features):
                raise RuntimeError("synthetic inference failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            compare_model_predictions(
                FailingModel(1), DummyModel(1), self.sequence
            )


if __name__ == "__main__":
    unittest.main()
