
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

VIDEO_PATH = Path(
    "data/raw/video_env_1_anonymized/ap_0016.mp4"
)
RESULTS_PATH = Path(
    "data/processed/posture_risk_results.csv"
)
OUTPUT_PATH = Path("outputs/posture_risk_demo.mp4")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def main():
    results = pd.read_csv(RESULTS_PATH)
    results_by_frame = results.set_index("frame").to_dict("index")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit("Could not create output video.")

    frame_number = 0

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)

            record = results_by_frame.get(frame_number, {})
            status = record.get(
                "status", "NO MEASUREMENT"
            )
            warnings = record.get("warnings", "None")

            if pose_result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            if status == "REVIEW POSTURE":
                color = (0, 0, 255)
                label = "REVIEW POSTURE - EXPERIMENTAL"
            elif status == "NO THRESHOLD TRIGGERED":
                color = (0, 180, 0)
                label = "NO THRESHOLD TRIGGERED"
            else:
                color = (0, 165, 255)
                label = "NO MEASUREMENT"

            cv2.rectangle(
                frame, (10, 10), (width - 10, 105),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame, label, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, f"Frame: {frame_number}",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA
            )

            cv2.putText(
                frame, str(warnings)[:100],
                (20, 94), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

            writer.write(frame)
            frame_number += 1

    cap.release()
    writer.release()

    print("Frames annotated:", frame_number)
    print("Saved annotated video:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
