
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math

VIDEO_DIR = Path("data/raw/video_env_1_anonymized")
OUTPUT_FILE = Path("data/processed/all_pose_angles.csv")

mp_pose = mp.solutions.pose
Landmark = mp_pose.PoseLandmark


def angle(a, b, c):
    a, b, c = map(lambda p: np.array(p, dtype=float), (a, b, c))
    ba, bc = a - b, c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return float("nan")

    cosine = np.clip(np.dot(ba, bc) / denominator, -1, 1)
    return math.degrees(math.acos(float(cosine)))


def point(landmarks, name):
    p = landmarks[Landmark[name].value]
    return (p.x, p.y), p.visibility


def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        raise SystemExit(f"No MP4 files found in {VIDEO_DIR}")

    all_rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            frame_no = 0
            video_rows = []

            if not cap.isOpened():
                print("Skipping unreadable video:", video_path.name)
                continue

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark
                    candidates = []

                    for side in ("LEFT", "RIGHT"):
                        shoulder, sv = point(lm, f"{side}_SHOULDER")
                        hip, hv = point(lm, f"{side}_HIP")
                        knee, kv = point(lm, f"{side}_KNEE")
                        ankle, av = point(lm, f"{side}_ANKLE")

                        confidence = min(sv, hv, kv, av)
                        candidates.append(
                            (confidence, shoulder, hip, knee, ankle, side)
                        )

                    confidence, shoulder, hip, knee, ankle, side = max(
                        candidates, key=lambda x: x[0]
                    )

                    if confidence >= 0.3:
                        trunk = math.degrees(math.atan2(
                            abs(shoulder[0] - hip[0]),
                            abs(hip[1] - shoulder[1])
                        ))

                        video_rows.append({
                            "video": video_path.name,
                            "frame": frame_no,
                            "side": side,
                            "min_visibility": confidence,
                            "trunk_inclination_deg": trunk,
                            "hip_angle_deg": angle(shoulder, hip, knee),
                            "knee_angle_deg": angle(hip, knee, ankle)
                        })

                frame_no += 1

            cap.release()
            all_rows.extend(video_rows)

            print(
                f"{video_path.name}: {frame_no} frames, "
                f"{len(video_rows)} usable measurements"
            )

    columns = [
        "video", "frame", "side", "min_visibility",
        "trunk_inclination_deg", "hip_angle_deg", "knee_angle_deg"
    ]

    df = pd.DataFrame(all_rows, columns=columns)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nVideos processed:", len(videos))
    print("Total measurements:", len(df))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
