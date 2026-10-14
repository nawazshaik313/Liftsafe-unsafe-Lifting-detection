
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math

VIDEO_PATH = "data/raw/video_env_1_anonymized/ap_0016.mp4"
OUTPUT_PATH = "data/processed/pose_angles.csv"

VISIBILITY_THRESHOLD = 0.5
SIDE_SWITCH_MARGIN = 0.15

mp_pose = mp.solutions.pose
Landmark = mp_pose.PoseLandmark


def angle_between(a, b, c):
    """Calculate angle ABC in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(float(cosine)))


def get_landmark(landmarks, name, width, height):
    """Return pixel coordinates and visibility."""
    point = landmarks[Landmark[name].value]

    x = point.x * width
    y = point.y * height

    return (x, y), point.visibility


def main():
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True, exist_ok=True
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    rows = []
    frame_number = 0
    detected_frames = 0
    best_visibility = 0.0
    previous_side = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            height, width = frame.shape[:2]

            result = pose.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            if result.pose_landmarks:
                detected_frames += 1
                landmarks = result.pose_landmarks.landmark
                candidates = {}

                for side in ("LEFT", "RIGHT"):
                    shoulder, sv = get_landmark(
                        landmarks, f"{side}_SHOULDER",
                        width, height
                    )
                    hip, hv = get_landmark(
                        landmarks, f"{side}_HIP",
                        width, height
                    )
                    knee, kv = get_landmark(
                        landmarks, f"{side}_KNEE",
                        width, height
                    )
                    ankle, av = get_landmark(
                        landmarks, f"{side}_ANKLE",
                        width, height
                    )

                    confidence = min(sv, hv, kv, av)

                    candidates[side] = (
                        confidence, shoulder, hip, knee, ankle
                    )

                # Prefer the previous side unless the other side
                # has clearly better visibility.
                if previous_side is None:
                    side = max(
                        candidates,
                        key=lambda s: candidates[s][0]
                    )
                else:
                    other_side = (
                        "RIGHT" if previous_side == "LEFT"
                        else "LEFT"
                    )

                    current_score = candidates[previous_side][0]
                    other_score = candidates[other_side][0]

                    if other_score > current_score + SIDE_SWITCH_MARGIN:
                        side = other_side
                    else:
                        side = previous_side

                confidence, shoulder, hip, knee, ankle = (
                    candidates[side]
                )

                best_visibility = max(
                    best_visibility, confidence
                )

                if confidence >= VISIBILITY_THRESHOLD:
                    # Positive trunk angle means the shoulder
                    # is to the right of the hip in the image.
                    dx = shoulder[0] - hip[0]
                    dy = hip[1] - shoulder[1]

                    trunk = abs(math.degrees(math.atan2(dx, dy)))

                    hip_angle = angle_between(
                        shoulder, hip, knee
                    )
                    knee_angle = angle_between(
                        hip, knee, ankle
                    )

                    if all(math.isfinite(v) for v in (
                        trunk, hip_angle, knee_angle
                    )):
                        rows.append({
                            "frame": frame_number,
                            "side": side,
                            "min_visibility": round(
                                confidence, 3
                            ),
                            "trunk_inclination_deg": trunk,
                            "hip_angle_deg": hip_angle,
                            "knee_angle_deg": knee_angle
                        })

                        previous_side = side

            frame_number += 1

    cap.release()

    columns = [
        "frame",
        "side",
        "min_visibility",
        "trunk_inclination_deg",
        "hip_angle_deg",
        "knee_angle_deg"
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Pose diagnostics ---")
    print("Frames processed:", frame_number)
    print("Frames with detected poses:", detected_frames)
    print("Frames with usable angles:", len(df))
    print("Best minimum landmark visibility:",
          round(best_visibility, 3))
    print("Saved CSV:", OUTPUT_PATH)

    if not df.empty:
        print("\nFirst five measurements:")
        print(df.head().round(2).to_string(index=False))

        print("\nAngle summary:")
        print(df[
            [
                "trunk_inclination_deg",
                "hip_angle_deg",
                "knee_angle_deg"
            ]
        ].describe().round(2))

        print("\nSelected body sides:")
        print(df["side"].value_counts().to_string())
    else:
        print(
            "\nNo usable measurements. Check video quality, "
            "landmark visibility and camera view."
        )


if __name__ == "__main__":
    main()
