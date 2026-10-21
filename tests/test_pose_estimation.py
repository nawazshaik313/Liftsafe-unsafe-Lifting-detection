import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src import pose_estimation
from src.pose_estimation import (
    PoseFrame,
    PoseTargetTracker,
    analyze_pose_frame,
    create_pose_estimator,
)


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


def make_person(center_x, *, trunk_angle=0.0, height=0.65, visibility=0.95):
    """Create deterministic 33-point normalized landmarks for one person."""
    landmarks = [SimpleNamespace(x=center_x, y=0.5, visibility=0.0) for _ in range(33)]
    shoulder_y = 0.62 - height * 0.6
    hip_y = 0.62 - height * 0.3
    shift = (hip_y - shoulder_y) * np.tan(np.radians(trunk_angle))
    positions = {
        11: (center_x + shift - 0.07, shoulder_y),
        12: (center_x + shift + 0.07, shoulder_y),
        23: (center_x - 0.06, hip_y),
        24: (center_x + 0.06, hip_y),
        25: (center_x - 0.05, 0.62),
        26: (center_x + 0.05, 0.62),
        27: (center_x - 0.05, min(0.98, 0.62 + height * 0.38)),
        28: (center_x + 0.05, min(0.98, 0.62 + height * 0.38)),
    }
    for index, (x, y) in positions.items():
        landmarks[index] = SimpleNamespace(x=x, y=y, visibility=visibility)
    return landmarks


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

    def test_selects_bending_lifter_over_larger_standing_bystander(self):
        bystander = make_person(0.25, trunk_angle=2, height=0.85, visibility=0.99)
        lifter = make_person(0.73, trunk_angle=55, height=0.62, visibility=0.84)
        tracker = PoseTargetTracker()
        selected = tracker.select([bystander, lifter], 1280, 720)
        self.assertEqual(selected.state, "selected")
        self.assertIs(selected.landmarks, lifter)

    def test_tracks_moving_lifter_when_stationary_bystander_becomes_more_bent(self):
        tracker = PoseTargetTracker()
        lifter = make_person(0.72, trunk_angle=45, height=0.62)
        first = tracker.select([make_person(0.25, trunk_angle=0, height=0.8), lifter], 640, 360)
        moved_lifter = make_person(0.75, trunk_angle=48, height=0.62)
        second = tracker.select(
            [make_person(0.25, trunk_angle=70, height=0.8), moved_lifter], 640, 360
        )
        self.assertEqual(second.target_id, first.target_id)
        self.assertIs(second.landmarks, moved_lifter)

    def test_short_landmark_dropout_pauses_without_switching_and_resumes_same_id(self):
        tracker = PoseTargetTracker(max_misses=2)
        first = tracker.select([make_person(0.7, trunk_angle=40)], 640, 360)
        missing = tracker.select([], 640, 360)
        resumed = tracker.select([make_person(0.72, trunk_angle=42)], 640, 360)
        self.assertEqual(missing.state, "target_lost")
        self.assertEqual(resumed.target_id, first.target_id)
        self.assertEqual(resumed.state, "selected")

    def test_ambiguous_people_are_not_silently_measured(self):
        left = make_person(0.46, trunk_angle=30)
        right = make_person(0.54, trunk_angle=30)
        result = analyze_pose_frame(
            FakePose(SimpleNamespace(pose_landmarks=[left, right])),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        self.assertEqual(result.state, "target_ambiguous")
        self.assertFalse(result.pose_detected)
        self.assertEqual(result.measurements, {})

    def test_measurements_and_drawable_landmarks_belong_to_same_selected_candidate(self):
        bystander = make_person(0.25, trunk_angle=0, height=0.8)
        lifter = make_person(0.73, trunk_angle=55, height=0.62)
        result = analyze_pose_frame(
            FakePose(SimpleNamespace(pose_landmarks=[bystander, lifter])),
            np.zeros((720, 1280, 3), dtype=np.uint8),
            PoseTargetTracker(),
        )
        self.assertIs(result.landmarks.landmark, lifter)
        self.assertGreater(result.measurements["Trunk inclination (°)"], 40)

    def test_ambiguous_target_can_be_manually_steered_by_frame_region(self):
        left = make_person(0.2, trunk_angle=30)
        right = make_person(0.8, trunk_angle=30)
        selected = PoseTargetTracker(preferred_region="right").select(
            [left, right], 640, 360
        )
        self.assertIs(selected.landmarks, right)

    def test_pose_coordinates_and_selection_are_resolution_independent(self):
        candidate = make_person(0.72, trunk_angle=45)
        small = PoseTargetTracker().select([candidate], 320, 180)
        large = PoseTargetTracker().select([candidate], 1280, 720)
        self.assertAlmostEqual(small.confidence, large.confidence)

    def test_target_tracker_resets_between_videos(self):
        tracker = PoseTargetTracker()
        first = tracker.select([make_person(0.2, trunk_angle=35)], 640, 360)
        tracker.reset()
        next_video = tracker.select([make_person(0.8, trunk_angle=35)], 640, 360)
        self.assertEqual(first.target_id, next_video.target_id)

    def test_long_target_exit_releases_old_track_before_reacquiring(self):
        tracker = PoseTargetTracker(max_misses=1)
        original = tracker.select([make_person(0.72, trunk_angle=45)], 640, 360)
        self.assertEqual(tracker.select([], 640, 360).state, "target_lost")
        self.assertEqual(tracker.select([], 640, 360).state, "no_person")
        entrant = tracker.select([make_person(0.25, trunk_angle=5)], 640, 360)
        self.assertEqual(entrant.state, "selected")
        self.assertNotEqual(entrant.target_id, original.target_id)

    def test_mediapipe_estimator_requests_multiple_people_on_cpu(self):
        class DummyLandmarker:
            def close(self):
                pass

        with patch.object(
            pose_estimation.mp.tasks.vision.PoseLandmarker,
            "create_from_options",
            return_value=DummyLandmarker(),
        ) as create:
            estimator = create_pose_estimator(static_image_mode=True)
            options = create.call_args.args[0]
            self.assertEqual(options.num_poses, 4)
            self.assertEqual(
                options.base_options.delegate,
                pose_estimation.mp.tasks.BaseOptions.Delegate.CPU,
            )
            estimator.close()

    def test_no_stale_pose_or_measurement_after_selected_person_disappears(self):
        pose = FakePose(SimpleNamespace(pose_landmarks=[make_person(0.7, trunk_angle=45)]))
        tracker = PoseTargetTracker()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        first = analyze_pose_frame(pose, image, tracker)
        pose.result = SimpleNamespace(pose_landmarks=[])
        second = analyze_pose_frame(pose, image, tracker)
        self.assertTrue(first.measurements)
        self.assertEqual(second.state, "target_lost")
        self.assertIsNone(second.landmarks)
        self.assertEqual(second.measurements, {})

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
