"""Shared CPU-only MediaPipe Tasks inference for video and camera frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    person_count: int = 0
    target_id: int | None = None
    target_score: float | None = None


@dataclass(frozen=True)
class TargetSelection:
    landmarks: Any | None
    state: str
    target_id: int | None
    confidence: float | None
    person_count: int


def _candidate_features(landmarks, width, height):
    required = tuple(POSE_LANDMARK_INDICES.values())
    visible = []
    for index in required:
        if index >= len(landmarks):
            continue
        point = landmarks[index]
        try:
            visibility = float(getattr(point, "visibility", 0.0) or 0.0)
            x = float(getattr(point, "x", math.nan))
            y = float(getattr(point, "y", math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y) and visibility >= 0.25:
            visible.append((x, y, visibility))
    if len(visible) < 2:
        return None

    xs, ys = [p[0] for p in visible], [p[1] for p in visible]
    box = (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    visibility = float(np.mean([p[2] for p in visible]))
    completeness = len(visible) / max(len(required), 1)

    def midpoint(a, b):
        first, second = landmarks[POSE_LANDMARK_INDICES[a]], landmarks[POSE_LANDMARK_INDICES[b]]
        if min(float(getattr(first, "visibility", 0)), float(getattr(second, "visibility", 0))) < 0.25:
            return None
        return ((float(first.x) + float(second.x)) / 2, (float(first.y) + float(second.y)) / 2)

    shoulder, hip = midpoint("LEFT_SHOULDER", "RIGHT_SHOULDER"), midpoint("LEFT_HIP", "RIGHT_HIP")
    trunk = None
    if shoulder is not None and hip is not None:
        trunk = math.degrees(math.atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1])))
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return {
        "landmarks": landmarks, "box": box, "center": center, "area": area,
        "visibility": visibility, "completeness": completeness, "trunk": trunk,
    }


def _box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class PoseTargetTracker:
    """Select and follow a likely lifting person without switching on a missed frame."""

    def __init__(self, *, max_misses=8, ambiguity_margin=0.08, preferred_region="auto"):
        self.max_misses = max_misses
        self.ambiguity_margin = ambiguity_margin
        self.preferred_region = preferred_region
        self._box = None
        self._trunk = None
        self._target_id = None
        self._next_id = 1
        self._misses = 0

    def reset(self):
        self._box = self._trunk = self._target_id = None
        self._next_id = 1
        self._misses = 0

    def select(self, candidates, width, height):
        features = [f for pose in candidates if (f := _candidate_features(pose, width, height))]
        count = len(features)
        if not count:
            if self._target_id is not None:
                self._misses += 1
                if self._misses <= self.max_misses:
                    return TargetSelection(None, "target_lost", self._target_id, None, 0)
                next_id = self._next_id
                self.reset()
                self._next_id = next_id
            return TargetSelection(None, "no_person", None, None, 0)

        largest_area = max(f["area"] for f in features) or 1.0
        for feature in features:
            trunk = feature["trunk"]
            bend = min(max((trunk or 0.0) / 75.0, 0.0), 1.0)
            movement = 0.0
            if self._trunk is not None and trunk is not None:
                movement = min(abs(trunk - self._trunk) / 35.0, 1.0)
            # Area and center are deliberately low-weight. Visible, complete
            # poses and lifting-like trunk posture/motion carry more evidence.
            feature["score"] = (
                0.25 * feature["visibility"]
                + 0.17 * feature["completeness"]
                + 0.12 * math.sqrt(feature["area"] / largest_area)
                + 0.04 * (1.0 - min(math.dist(feature["center"], (0.5, 0.5)) / 0.71, 1.0))
                + 0.22 * bend
                + 0.10 * movement
                + 0.10 * self._region_score(feature["center"][0])
            )

        # If a target was previously selected, continuity wins over a stronger
        # one-frame score from a bystander. Do not substitute another person
        # during short occlusions or detector dropouts.
        if self._box is not None:
            matched = []
            for feature in features:
                distance = math.dist(feature["center"], ((self._box[0] + self._box[2]) / 2,
                                                         (self._box[1] + self._box[3]) / 2))
                continuity = 0.65 * _box_iou(self._box, feature["box"]) + 0.35 * max(0.0, 1.0 - distance / 0.35)
                matched.append((continuity, feature))
            continuity, chosen = max(matched, key=lambda item: item[0])
            if continuity >= 0.22:
                self._misses = 0
                self._box = chosen["box"]
                self._trunk = chosen["trunk"]
                return TargetSelection(chosen["landmarks"], "selected", self._target_id,
                                       float(0.5 * continuity + 0.5 * chosen["score"]), count)
            self._misses += 1
            if self._misses <= self.max_misses:
                return TargetSelection(None, "target_lost", self._target_id, None, count)
            next_id = self._next_id
            self.reset()
            self._next_id = next_id

        if self.preferred_region in {"left", "center", "right"}:
            in_region = [
                feature for feature in features
                if self._region_score(feature["center"][0]) == 1.0
            ]
            if in_region:
                features = in_region

        ranked = sorted(features, key=lambda f: f["score"], reverse=True)
        gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
        if ranked[0]["score"] < 0.38 or (len(ranked) > 1 and gap < self.ambiguity_margin):
            return TargetSelection(None, "target_ambiguous", None, float(ranked[0]["score"]), count)

        chosen = ranked[0]
        self._target_id = self._next_id
        self._next_id += 1
        self._box, self._trunk, self._misses = chosen["box"], chosen["trunk"], 0
        return TargetSelection(chosen["landmarks"], "selected", self._target_id,
                               float(chosen["score"]), count)

    def _region_score(self, center_x):
        if self.preferred_region == "left":
            return 1.0 if center_x < 1 / 3 else 0.0
        if self.preferred_region == "center":
            return 1.0 if 1 / 3 <= center_x <= 2 / 3 else 0.0
        if self.preferred_region == "right":
            return 1.0 if center_x > 2 / 3 else 0.0
        return 0.5


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
            num_poses=4,
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
        people = [SimpleNamespace(landmark=pose) for pose in result.pose_landmarks]
        return SimpleNamespace(
            pose_landmarks=people
        )

    def close(self):
        self._landmarker.close()


def create_pose_estimator(*, static_image_mode: bool = False, model_complexity: int = 1):
    """Create one explicitly CPU-backed estimator for a video/camera session."""
    return CpuPoseEstimator(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
    )


def analyze_pose_frame(pose, bgr_frame, tracker: PoseTargetTracker | None = None) -> PoseFrame:
    """Infer all detected people, select one target, and measure only that target."""
    if (
        bgr_frame is None
        or getattr(bgr_frame, "ndim", 0) != 3
        or bgr_frame.shape[2] != 3
        or bgr_frame.size == 0
    ):
        raise ValueError("Expected a non-empty BGR image with three color channels.")

    height, width = bgr_frame.shape[:2]
    results = pose.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    raw_poses = getattr(results, "pose_landmarks", None)
    if not raw_poses:
        candidates = []
    elif hasattr(raw_poses, "landmark"):
        candidates = [raw_poses.landmark]
    else:
        candidates = [getattr(item, "landmark", item) for item in raw_poses]
    tracker = tracker or PoseTargetTracker()
    target = tracker.select(candidates, width, height)
    if target.landmarks is None:
        return PoseFrame(None, False, target.state, None, {}, target.person_count,
                         target.target_id, target.confidence)

    landmarks = target.landmarks
    visibility = calculate_landmark_confidence(landmarks, POSE_LANDMARK_INDICES)
    measurements = calculate_measurements(
        landmarks, width, height, POSE_LANDMARK_INDICES
    )
    state = (
        "measurements_available"
        if measurements.get("Trunk inclination (°)") is not None
        else "incomplete_landmarks"
    )
    return PoseFrame(SimpleNamespace(landmark=landmarks), True, state, visibility,
                     measurements, target.person_count, target.target_id,
                     target.confidence)


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
