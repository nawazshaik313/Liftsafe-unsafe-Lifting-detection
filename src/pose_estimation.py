
import cv2
import mediapipe as mp
from pathlib import Path

# File paths
video_path = Path("data/raw/video_env_1_anonymized/ap_0016.mp4")
output_path = Path("outputs/pose_sample.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Initialize MediaPipe Pose

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {video_path}")

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detected = False

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        # OpenCV uses BGR; MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw detected body landmarks and connections
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Save the first frame with a detected person
            cv2.imwrite(str(output_path), frame)

            print("Person detected!")
            print("Number of pose landmarks:",len(results.pose_landmarks.landmark))
            print("Saved annotated image:", output_path)

            detected = True
            break

    if not detected:
        print("No person detected in this video.")

finally:
    cap.release()
    pose.close()

print("Pose estimation test finished.")
