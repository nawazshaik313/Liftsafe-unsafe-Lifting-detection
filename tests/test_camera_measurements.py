import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.camera_measurements import (
    calculate_angle,
    calculate_landmark_confidence,
    calculate_measurements,
    is_fresh_timestamp,
)


class CameraMeasurementTests(unittest.TestCase):
    def test_live_timestamp_freshness_rejects_old_future_and_invalid_times(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertTrue(is_fresh_timestamp(
            (now - timedelta(seconds=1)).isoformat(), now=now
        ))
        self.assertFalse(is_fresh_timestamp(
            (now - timedelta(seconds=3)).isoformat(), now=now
        ))
        self.assertFalse(is_fresh_timestamp(
            (now + timedelta(seconds=1)).isoformat(), now=now
        ))
        self.assertFalse(is_fresh_timestamp("not-a-timestamp", now=now))
        self.assertFalse(is_fresh_timestamp(None, now=now))

    def test_key_landmark_confidence_is_bounded_and_handles_missing_values(self):
        landmarks = [
            SimpleNamespace(visibility=0.8),
            SimpleNamespace(visibility=1.2),
            SimpleNamespace(visibility=float("nan")),
        ]
        self.assertAlmostEqual(
            calculate_landmark_confidence(landmarks, {"A": 0, "B": 1, "C": 2}),
            0.9,
        )
        self.assertIsNone(calculate_landmark_confidence([], {"A": 0}))

    def test_angle_calculation_and_degenerate_inputs(self):
        self.assertAlmostEqual(calculate_angle((1, 0), (0, 0), (0, 1)), 90.0)
        self.assertIsNone(calculate_angle((0, 0), (0, 0), (1, 1)))
        self.assertIsNone(calculate_angle((float("nan"), 0), (0, 0), (1, 1)))

    def test_measurements_require_visible_landmarks(self):
        names = (
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
        )
        indices = {name: i for i, name in enumerate(names)}
        points = {
            "LEFT_SHOULDER": (0.45, 0.20), "RIGHT_SHOULDER": (0.55, 0.20),
            "LEFT_HIP": (0.45, 0.50), "RIGHT_HIP": (0.55, 0.50),
            "LEFT_KNEE": (0.45, 0.75), "RIGHT_KNEE": (0.55, 0.75),
            "LEFT_ANKLE": (0.45, 0.95), "RIGHT_ANKLE": (0.55, 0.95),
        }
        landmarks = [
            SimpleNamespace(x=points[name][0], y=points[name][1], visibility=1.0)
            for name in names
        ]

        result = calculate_measurements(landmarks, 100, 200, indices)
        self.assertAlmostEqual(result["Trunk inclination (°)"], 0.0)
        self.assertIn("Left hip angle (°)", result)
        self.assertIn("Right hip angle (°)", result)
        self.assertIn("Left knee angle (°)", result)
        self.assertIn("Right knee angle (°)", result)

        landmarks[indices["LEFT_ANKLE"]].visibility = 0.1
        result = calculate_measurements(landmarks, 100, 200, indices)
        self.assertNotIn("Left knee angle (°)", result)
        self.assertIn("Right knee angle (°)", result)

    def test_invalid_frame_dimensions_and_missing_points_are_empty(self):
        indices = {"LEFT_SHOULDER": 0}
        landmark = [SimpleNamespace(x=0.2, y=0.3, visibility=1.0)]
        self.assertEqual(calculate_measurements(landmark, 0, 10, indices), {})
        self.assertEqual(calculate_measurements(landmark, 10, 10, indices), {})


if __name__ == "__main__":
    unittest.main()
