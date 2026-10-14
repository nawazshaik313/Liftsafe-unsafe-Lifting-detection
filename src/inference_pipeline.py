
import cv2
import math
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

TRUNK_LIMIT = 30.0
HIP_LIMIT = 90.0
KNEE_LIMIT = 90.0
VISIBILITY_THRESHOLD = 0.5

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
Landmark = mp_pose.PoseLandmark


def angle_between(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    return math.degrees(
        math.acos(float(np.clip(cosine, -1.0, 1.0)))
    )


def landmark_point(landmarks, name, width, height):
    point = landmarks[Landmark[name].value]
    return (
        (point.x * width, point.y * height),
        float(point.visibility),
    )


def calculate_frame_angles(landmarks, width, height):
    candidates = {}

    for side in ("LEFT", "RIGHT"):
        shoulder, sv = landmark_point(
            landmarks, f"{side}_SHOULDER", width, height
        )
        hip, hv = landmark_point(
            landmarks, f"{side}_HIP", width, height
        )
        knee, kv = landmark_point(
            landmarks, f"{side}_KNEE", width, height
        )
        ankle, av = landmark_point(
            landmarks, f"{side}_ANKLE", width, height
        )

        confidence = min(sv, hv, kv, av)
        candidates[side] = (
            confidence, shoulder, hip, knee, ankle
        )

    side = max(candidates, key=lambda key: candidates[key][0])
    confidence, shoulder, hip, knee, ankle = candidates[side]

    if confidence < VISIBILITY_THRESHOLD:
        return None

    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]

    trunk = abs(math.degrees(math.atan2(dx, dy)))
    hip_angle = angle_between(shoulder, hip, knee)
    knee_angle = angle_between(hip, knee, ankle)

    if not all(math.isfinite(value) for value in (
        trunk, hip_angle, knee_angle
    )):
        return None

    warnings = []

    if trunk > TRUNK_LIMIT:
        warnings.append("High trunk inclination")
    if hip_angle < HIP_LIMIT:
        warnings.append("Deep hip flexion")
    if knee_angle < KNEE_LIMIT:
        warnings.append("Deep knee flexion")

    return {
        "side": side,
        "min_visibility": confidence,
        "trunk_inclination_deg": trunk,
        "hip_angle_deg": hip_angle,
        "knee_angle_deg": knee_angle,
        "status": (
            "REVIEW POSTURE" if warnings
            else "NO THRESHOLD TRIGGERED"
        ),
        "warnings": "; ".join(warnings) if warnings else "None",
    }


def analyze_video(video_path, output_dir="outputs"):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = output_dir / f"{video_path.stem}_analyzed.mp4"
    output_csv = output_dir / f"{video_path.stem}_measurements.csv"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated video.")

    rows = []
    frame_number = 0

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                record = None

                if result.pose_landmarks:
                    record = calculate_frame_angles(
                        result.pose_landmarks.landmark,
                        width,
                        height,
                    )

                    mp_draw.draw_landmarks(
                        frame,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                if record is not None:
                    record["frame"] = frame_number
                    record["time_seconds"] = frame_number / fps
                    rows.append(record)

                    flagged = record["status"] == "REVIEW POSTURE"
                    color = (0, 0, 255) if flagged else (0, 180, 0)
                    label = (
                        "EXPERIMENTAL: REVIEW POSTURE"
                        if flagged else "NO THRESHOLD TRIGGERED"
                    )
                    warning_text = record["warnings"]
                else:
                    color = (0, 165, 255)
                    label = "NO RELIABLE MEASUREMENT"
                    warning_text = "Check visibility / pose"

                cv2.rectangle(
                    frame, (10, 10), (width - 10, 85),
                    (25, 25, 25), -1
                )
                cv2.putText(
                    frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    color, 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, str(warning_text)[:95], (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

                writer.write(frame)
                frame_number += 1
    finally:
        cap.release()
        writer.release()

    columns = [
        "frame", "time_seconds", "side", "min_visibility",
        "trunk_inclination_deg", "hip_angle_deg",
        "knee_angle_deg", "status", "warnings"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    return {
        "measurements": df,
        "video_path": output_video,
        "csv_path": output_csv,
        "frames_processed": frame_number,
        "frames_measured": len(df),
    }


if __name__ == "__main__":
    sample_video = "data/raw/video_env_1_anonymized/ap_0016.mp4"
    result = analyze_video(sample_video)

    print("Frames processed:", result["frames_processed"])
    print("Frames measured:", result["frames_measured"])
    print("Annotated video:", result["video_path"])
    print("Measurements CSV:", result["csv_path"])
    print("\nStatus counts:")
    if not result["measurements"].empty:
        print(result["measurements"]["status"].value_counts().to_string())
    else:
        print("No reliable pose measurements.")
