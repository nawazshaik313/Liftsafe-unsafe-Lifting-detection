import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.pose_estimation import PoseFrame, analyze_pose_frame, create_pose_estimator


class FakePose:
    def __init__(self, result=None, error=None):
        self.result = result or SimpleNamespace(pose_landmarks=None)
        self.error = error
        self.last_image = None

    def process(self, image):
        self.last_image = image
        if self.error:
            raise self.error
        return self.result


class PoseEstimationTests(unittest.TestCase):
    def test_no_person_returns_empty_current_measurements(self):
        result = analyze_pose_frame(FakePose(), np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertEqual(result, PoseFrame(None, False, "no_person", None, {}))

    def test_input_is_converted_from_bgr_to_rgb(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :] = (5, 20, 220)
        pose = FakePose()
        analyze_pose_frame(pose, image)
        self.assertTupleEqual(tuple(pose.last_image[0, 0]), (220, 20, 5))

    def test_inference_error_propagates_for_actionable_page_handling(self):
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            analyze_pose_frame(
                FakePose(error=RuntimeError("inference failed")),
                np.zeros((20, 20, 3), dtype=np.uint8),
            )

    def test_empty_or_invalid_image_is_rejected_before_inference(self):
        pose = FakePose()
        for image in (None, np.empty((0, 0, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)):
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "BGR image"):
                analyze_pose_frame(pose, image)
        self.assertIsNone(pose.last_image)

    @unittest.skipUnless(
        os.environ.get("LIFTSAFE_REAL_MEDIAPIPE_TEST") == "1"
        and Path("outputs/pose_sample.jpg").is_file(),
        "Set LIFTSAFE_REAL_MEDIAPIPE_TEST=1 to run real MediaPipe inference on the local pose sample.",
    )
    def test_real_mediapipe_smoke_on_local_pose_image(self):
        image = cv2.imread("outputs/pose_sample.jpg")
        self.assertIsNotNone(image)
        pose = create_pose_estimator(static_image_mode=True)
        try:
            result = analyze_pose_frame(pose, image)
        finally:
            pose.close()
        self.assertTrue(result.pose_detected)
        self.assertIsNotNone(result.landmarks)


if __name__ == "__main__":
    unittest.main()
