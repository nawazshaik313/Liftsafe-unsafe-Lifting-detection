"""Shared CPU-only MediaPipe Tasks inference for video and camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.camera_measurements import calculate_landmark_confidence, calculate_measurements


mp_pose = mp.solutions.pose
POSE_LANDMARK_INDICES = {
    name: mp_pose.PoseLandmark[name].value
    for name in (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )
}
POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "pose_landmarker_lite.task"


@dataclass(frozen=True)
class PoseFrame:
    """Pose and measurements for one BGR image."""

    landmarks: Any | None
    pose_detected: bool
    state: str
    visibility: float | None
    measurements: dict[str, float]


class CpuPoseEstimator:
    """Adapt MediaPipe Tasks' explicit CPU delegate to a small process API."""

    def __init__(self, *, static_image_mode: bool, model_complexity: int):
        del model_complexity  # The shipped task model is the Lite model.
        if not POSE_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"MediaPipe pose model is missing: {POSE_MODEL_PATH}"
            )
        vision = mp.tasks.vision
        self._static_image_mode = static_image_mode
        self._last_timestamp_ms = -1
        running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(POSE_MODEL_PATH),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        rgb_frame = np.ascontiguousarray(rgb_frame)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._static_image_mode:
            result = self._landmarker.detect(image)
        else:
            timestamp_ms = max(time.monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        # Keep the same ``pose_landmarks.landmark`` shape used by MediaPipe
        # Solutions so both pages consume one stable interface.
        return SimpleNamespace(
            pose_landmarks=(SimpleNamespace(landmark=landmarks) if landmarks else None)
        )

    def close(self):
        self._landmarker.close()


def create_pose_estimator(*, static_image_mode: bool = False, model_complexity: int = 1):
    """Create one explicitly CPU-backed estimator for a video/camera session."""
    return CpuPoseEstimator(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
    )


def analyze_pose_frame(pose, bgr_frame) -> PoseFrame:
    """Run real Pose inference and return quality-aware image-space angles."""
    if (
        bgr_frame is None
        or getattr(bgr_frame, "ndim", 0) != 3
        or bgr_frame.shape[2] != 3
        or bgr_frame.size == 0
    ):
        raise ValueError("Expected a non-empty BGR image with three color channels.")

    height, width = bgr_frame.shape[:2]
    results = pose.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    pose_landmarks = getattr(results, "pose_landmarks", None)
    if pose_landmarks is None:
        return PoseFrame(None, False, "no_person", None, {})

    landmarks = pose_landmarks.landmark
    visibility = calculate_landmark_confidence(landmarks, POSE_LANDMARK_INDICES)
    measurements = calculate_measurements(
        landmarks, width, height, POSE_LANDMARK_INDICES
    )
    state = (
        "measurements_available"
        if measurements.get("Trunk inclination (°)") is not None
        else "incomplete_landmarks"
    )
    return PoseFrame(pose_landmarks, True, state, visibility, measurements)


def draw_pose_landmarks(image, landmarks) -> None:
    """Draw visible Tasks landmarks and standard MediaPipe pose connections."""
    height, width = image.shape[:2]
    for start, end in mp_pose.POSE_CONNECTIONS:
        first = landmarks[start]
        second = landmarks[end]
        if min(getattr(first, "visibility", 0.0), getattr(second, "visibility", 0.0)) < 0.35:
            continue
        p1 = (int(first.x * width), int(first.y * height))
        p2 = (int(second.x * width), int(second.y * height))
        cv2.line(image, p1, p2, (245, 245, 245), 2, cv2.LINE_AA)
    for landmark in landmarks:
        if getattr(landmark, "visibility", 0.0) >= 0.35:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(image, point, 3, (35, 90, 255), -1, cv2.LINE_AA)
