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
    candidate_number: int | None = None
    target_bbox: tuple[float, float, float, float] | None = None
    selection_reason: str | None = None
    candidates: tuple[dict, ...] = ()


@dataclass(frozen=True)
class TargetSelection:
    landmarks: Any | None
    state: str
    target_id: int | None
    confidence: float | None
    person_count: int
    candidate_number: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    reason: str = ""
    candidates: tuple[dict, ...] = ()


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

    def __init__(
        self, *, max_misses=8, ambiguity_margin=0.08, preferred_region="auto",
        manual_target_center=None,
    ):
        # Keep the configured value for callers that inspect tracker settings.
        # A lock is deliberately not released after a timeout: doing so can
        # silently transfer measurements to a bystander.
        self.max_misses = max_misses
        self.ambiguity_margin = ambiguity_margin
        self.preferred_region = preferred_region
        self.manual_target_center = manual_target_center
        self._manual_request = None
        self._manual_initial_applied = False
        self._box = None
        self._trunk = None
        self._target_id = None
        self._next_id = 1
        self._misses = 0

    def reset(self):
        self._box = self._trunk = self._target_id = None
        self._next_id = 1
        self._misses = 0
        self._manual_request = None

    def request_manual_candidate(self, candidate_number):
        """Lock the next detected frame to a left-to-right numbered candidate."""
        self._manual_request = candidate_number

    def select(self, candidates, width, height):
        features = [f for pose in candidates if (f := _candidate_features(pose, width, height))]
        features.sort(key=lambda feature: feature["center"][0])
        for candidate_number, feature in enumerate(features, start=1):
            feature["candidate_number"] = candidate_number
        count = len(features)
        if not count:
            if self._target_id is not None:
                self._misses += 1
                return TargetSelection(None, "target_lost", self._target_id, None, 0,
                                       reason="Selected target not detected; holding lock until it is matched again or manually changed.")
            return TargetSelection(None, "no_person", None, None, 0,
                                   reason="No valid multi-pose candidates detected.")

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

        summaries = tuple({
            "candidate_number": f["candidate_number"],
            "bbox": f["box"],
        "center": f["center"],
            "visibility": f["visibility"],
            "score": f["score"],
        } for f in features)

        requested = self._manual_request
        if requested is None and not self._manual_initial_applied and self.manual_target_center is not None:
            requested = min(
                features,
                key=lambda f: math.dist(f["center"], self.manual_target_center),
            )["candidate_number"]
            self._manual_initial_applied = True
        if requested is not None:
            self._manual_request = None
            chosen = next((f for f in features if f["candidate_number"] == requested), None)
            if chosen is None:
                self._manual_request = requested
                return TargetSelection(None, "target_ambiguous", self._target_id, None,
                                       count, reason="The selected candidate is no longer present.",
                                       candidates=summaries)
            self._target_id = self._next_id
            self._next_id += 1
            self._box, self._trunk, self._misses = chosen["box"], chosen["trunk"], 0
            return TargetSelection(chosen["landmarks"], "selected", self._target_id,
                                   float(chosen["score"]), count,
                                   candidate_number=chosen["candidate_number"],
                                   bbox=chosen["box"], reason="User-selected candidate lock.",
                                   candidates=summaries)

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
                                       float(0.5 * continuity + 0.5 * chosen["score"]), count,
                                       candidate_number=chosen["candidate_number"],
                                       bbox=chosen["box"], reason="Matched prior target by box overlap and center continuity.",
                                       candidates=summaries)
            self._misses += 1
            return TargetSelection(None, "target_lost", self._target_id, None, count,
                                   reason="No candidate matched the tracked person's box; refusing to switch.",
                                   candidates=summaries)

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
            return TargetSelection(None, "target_ambiguous", None, float(ranked[0]["score"]), count,
                                   candidate_number=ranked[0]["candidate_number"],
                                   bbox=ranked[0]["box"],
                                   reason="Top candidate scores are too close; manual selection required.",
                                   candidates=summaries)

        chosen = ranked[0]
        self._target_id = self._next_id
        self._next_id += 1
        self._box, self._trunk, self._misses = chosen["box"], chosen["trunk"], 0
        return TargetSelection(chosen["landmarks"], "selected", self._target_id,
                               float(chosen["score"]), count,
                               candidate_number=chosen["candidate_number"],
                               bbox=chosen["box"], reason="Highest combined posture, visibility, size, and position score.",
                               candidates=summaries)

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
                         target.target_id, target.confidence,
                         target.candidate_number, target.bbox, target.reason,
                         target.candidates)

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
                     target.confidence, target.candidate_number, target.bbox,
                     target.reason, target.candidates)


def detect_pose_candidates(pose, bgr_frame):
    """Return visible multi-pose candidate boxes, numbered left-to-right."""
    if (bgr_frame is None or getattr(bgr_frame, "ndim", 0) != 3
            or bgr_frame.shape[2] != 3 or bgr_frame.size == 0):
        raise ValueError("Expected a non-empty BGR image with three color channels.")
    height, width = bgr_frame.shape[:2]
    results = pose.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    raw = getattr(results, "pose_landmarks", None)
    if not raw:
        return ()
    poses = [raw.landmark] if hasattr(raw, "landmark") else [
        getattr(item, "landmark", item) for item in raw
    ]
    features = [feature for landmarks in poses
                if (feature := _candidate_features(landmarks, width, height))]
    features.sort(key=lambda feature: feature["center"][0])
    return tuple({
        "candidate_number": number,
        "bbox": feature["box"],
        "center": feature["center"],
        "visibility": feature["visibility"],
        "landmarks": feature["landmarks"],
    } for number, feature in enumerate(features, start=1))


def draw_candidate_boxes(image, candidates, selected_number=None):
    """Draw numbered candidate boxes in the image's original coordinate space."""
    height, width = image.shape[:2]
    for candidate in candidates:
        left, top, right, bottom = candidate["bbox"]
        p1 = (round(left * width), round(top * height))
        p2 = (round(right * width), round(bottom * height))
        selected = candidate["candidate_number"] == selected_number
        color = (40, 230, 80) if selected else (255, 190, 40)
        cv2.rectangle(image, p1, p2, color, 3 if selected else 2, cv2.LINE_AA)
        label = f"P{candidate['candidate_number']}" + (" SELECTED" if selected else "")
        cv2.putText(image, label, (p1[0], max(24, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


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
