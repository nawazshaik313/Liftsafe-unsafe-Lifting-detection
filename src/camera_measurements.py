"""Camera-independent calculations for MediaPipe pose landmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math

import numpy as np


def is_fresh_timestamp(timestamp, *, max_age_seconds=2.5, now=None):
    """Return whether an ISO timestamp is recent enough for live UI display."""
    if not timestamp:
        return False
    try:
        measured_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    age = (current_time - measured_at).total_seconds()
    return 0 <= age <= max_age_seconds


def calculate_angle(a, b, c):
    """Return angle ABC in degrees, or None for invalid/degenerate points."""
    try:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        c = np.asarray(c, dtype=float)
    except (TypeError, ValueError):
        return None

    if a.shape != (2,) or b.shape != (2,) or c.shape != (2,):
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return None

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _point(landmarks, index, width, height, visibility_threshold):
    if index is None or index < 0 or index >= len(landmarks):
        return None
    landmark = landmarks[index]
    visibility = getattr(landmark, "visibility", 0.0)
    x = getattr(landmark, "x", float("nan"))
    y = getattr(landmark, "y", float("nan"))
    if not all(math.isfinite(float(value)) for value in (visibility, x, y)):
        return None
    if visibility < visibility_threshold:
        return None
    return (float(x) * width, float(y) * height)


def calculate_landmark_confidence(
    landmarks: Sequence,
    landmark_indices: Mapping[str, int],
    names: Sequence[str] | None = None,
) -> float | None:
    """Return mean MediaPipe visibility for key landmarks, in the 0–1 range."""
    selected_names = names or tuple(landmark_indices)
    values = []
    for name in selected_names:
        index = landmark_indices.get(name)
        if index is None or index < 0 or index >= len(landmarks):
            continue
        try:
            visibility = float(getattr(landmarks[index], "visibility", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(visibility):
            values.append(min(1.0, max(0.0, visibility)))
    return float(np.mean(values)) if values else None


def calculate_measurements(
    landmarks: Sequence,
    width: int,
    height: int,
    landmark_indices: Mapping[str, int],
    visibility_threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate image-space trunk and knee angles from visible landmarks."""
    if width <= 0 or height <= 0:
        return {}

    required = (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )
    points = {
        name: _point(
            landmarks,
            landmark_indices.get(name),
            width,
            height,
            visibility_threshold,
        )
        for name in required
    }

    measurements = {}
    shoulders = [points[n] for n in ("LEFT_SHOULDER", "RIGHT_SHOULDER") if points[n] is not None]
    hips = [points[n] for n in ("LEFT_HIP", "RIGHT_HIP") if points[n] is not None]
    if shoulders and hips:
        shoulder = np.mean(shoulders, axis=0)
        hip = np.mean(hips, axis=0)
        dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
        measurements["Trunk inclination (°)"] = float(
            np.degrees(np.arctan2(abs(dx), abs(dy)))
        )

    for side in ("LEFT", "RIGHT"):
        shoulder, hip, knee = (
            points[f"{side}_{part}"] for part in ("SHOULDER", "HIP", "KNEE")
        )
        if shoulder is not None and hip is not None and knee is not None:
            angle = calculate_angle(shoulder, hip, knee)
            if angle is not None:
                measurements[f"{side.title()} hip angle (°)"] = angle

        hip, knee, ankle = (points[f"{side}_{part}"] for part in ("HIP", "KNEE", "ANKLE"))
        if hip is not None and knee is not None and ankle is not None:
            angle = calculate_angle(hip, knee, ankle)
            if angle is not None:
                measurements[f"{side.title()} knee angle (°)"] = angle

    return measurements
