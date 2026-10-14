import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = Path("data/raw/video_env_1_anonymized/ap_0016.mp4")
OUTPUT_DIR = Path("outputs/pose_video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATED_PATH = OUTPUT_DIR / "annotated_video.mp4"
CSV_PATH = OUTPUT_DIR / "pose_features.csv"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def angle_from_vertical(a, b):
    """Angle of vector a->b relative to the vertical image axis."""
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-8)))


def joint_angle(a, b, c):
    """Angle ABC in degrees, using 2D image coordinates."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


if not VIDEO_PATH.exists():
    raise SystemExit(f"Video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if np.isfinite(fps) and fps > 0 else 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    str(ANNOTATED_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

rows = []
frame_index = 0

try:
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            row = {
                "frame": frame_index,
                "time_seconds": frame_index / fps,
                "pose_detected": False,
            }

            if results.pose_landmarks:
                row["pose_detected"] = True
                landmarks = results.pose_landmarks.landmark

                def point(name):
                    lm = landmarks[mp_pose.PoseLandmark[name].value]
                    return np.array([lm.x * width, lm.y * height])

                # Use image-space landmarks. These are 2D approximations.
                left_shoulder = point("LEFT_SHOULDER")
                right_shoulder = point("RIGHT_SHOULDER")
                left_hip = point("LEFT_HIP")
                right_hip = point("RIGHT_HIP")
                left_knee = point("LEFT_KNEE")
                right_knee = point("RIGHT_KNEE")
                left_ankle = point("LEFT_ANKLE")
                right_ankle = point("RIGHT_ANKLE")

                shoulder = (left_shoulder + right_shoulder) / 2
                hip = (left_hip + right_hip) / 2

                row["trunk_inclination_2d_deg"] = angle_from_vertical(
                    shoulder, hip
                )
                row["left_knee_angle_2d_deg"] = joint_angle(
                    left_hip, left_knee, left_ankle
                )
                row["right_knee_angle_2d_deg"] = joint_angle(
                    right_hip, right_knee, right_ankle
                )
                row["left_hip_angle_2d_deg"] = joint_angle(
                    left_shoulder, left_hip, left_knee
                )
                row["right_hip_angle_2d_deg"] = joint_angle(
                    right_shoulder, right_hip, right_knee
                )

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

            rows.append(row)
            writer.write(frame)
            frame_index += 1

            if frame_index % 100 == 0:
                print(f"Processed {frame_index} frames...")

finally:
    cap.release()
    writer.release()

with CSV_PATH.open("w", newline="") as file:
    fieldnames = sorted({key for row in rows for key in row})
    csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(rows)

detected_count = sum(row["pose_detected"] for row in rows)

print("\nVideo processing complete.")
print("Frames processed:", frame_index)
print("Frames with pose detected:", detected_count)
print("FPS:", round(fps, 2))
print("Annotated video:", ANNOTATED_PATH)
print("Pose feature CSV:", CSV_PATH)
print("\nNote: These are 2D image-space estimates, not validated 3D ergonomic angles.")
print("They are not yet compatible with the INAIL LI classifier.")
