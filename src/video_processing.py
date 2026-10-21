"""Uploaded-video pose analysis with bounded temporary resource handling."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.camera_measurements import calculate_landmark_confidence, calculate_measurements

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_VIDEO_PROCESSING_SECONDS = 600
POSE_LANDMARK_INDICES = {
    name: mp_pose.PoseLandmark[name].value
    for name in (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )
}

def process_video(uploaded_file, progress_bar, status, threshold=45):
    """Process one upload and return an in-memory annotated MP4 and measurements."""
    video_bytes = uploaded_file.getvalue()
    if not video_bytes:
        raise ValueError("The uploaded video is empty. Choose a video with frames.")
    if len(video_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Video is larger than the 500 MB analysis limit.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to prepare a browser-compatible MP4.")

    with tempfile.TemporaryDirectory(prefix="liftsafe_") as temp_name:
        temp_dir = Path(temp_name)
        input_path = temp_dir / "input_video"
        output_path = temp_dir / "annotated.mp4"
        compatible_path = temp_dir / "annotated_video_h264.mp4"
        input_path.write_bytes(video_bytes)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("OpenCV could not read this video. Try a valid MP4, MOV, or AVI file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("The uploaded video has invalid dimensions.")

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise ValueError("Could not create the annotated output video.")

        rows = []
        frame_index = 0
        deadline = time.monotonic() + MAX_VIDEO_PROCESSING_SECONDS
        try:
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                while True:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Video analysis exceeded the 10 minute processing limit."
                        )
                    success, frame = cap.read()
                    if not success:
                        break
                    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    row = {
                        "frame": frame_index,
                        "time_seconds": frame_index / fps,
                        "pose_detected": False,
                        "landmark_confidence": np.nan,
                        "posture_review_flag": False,
                        "trunk_inclination_2d_deg": np.nan,
                        "left_hip_angle_2d_deg": np.nan,
                        "right_hip_angle_2d_deg": np.nan,
                        "left_knee_angle_2d_deg": np.nan,
                        "right_knee_angle_2d_deg": np.nan,
                    }
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        confidence = calculate_landmark_confidence(
                            landmarks, POSE_LANDMARK_INDICES
                        )
                        if confidence is not None:
                            row["landmark_confidence"] = confidence

                        measured = calculate_measurements(
                            landmarks, width, height, POSE_LANDMARK_INDICES
                        )
                        row["pose_detected"] = bool(measured)
                        row["left_hip_angle_2d_deg"] = measured.get(
                            "Left hip angle (°)", np.nan
                        )
                        row["right_hip_angle_2d_deg"] = measured.get(
                            "Right hip angle (°)", np.nan
                        )
                        row["trunk_inclination_2d_deg"] = measured.get(
                            "Trunk inclination (°)", np.nan
                        )
                        row["left_knee_angle_2d_deg"] = measured.get(
                            "Left knee angle (°)", np.nan
                        )
                        row["right_knee_angle_2d_deg"] = measured.get(
                            "Right knee angle (°)", np.nan
                        )
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        trunk = row["trunk_inclination_2d_deg"]
                        if np.isfinite(trunk) and trunk >= threshold:
                            row["posture_review_flag"] = True
                            cv2.rectangle(frame, (0, 0), (width, 72), (0, 140, 255), -1)
                            cv2.putText(frame, "POSTURE REVIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"Trunk inclination: {trunk:.1f} deg (threshold {threshold} deg)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    rows.append(row)
                    writer.write(frame)
                    frame_index += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_index / total_frames, 1.0))
                        status.caption(f"Processed {frame_index} of {total_frames} frames")
        finally:
            cap.release()
            writer.release()

        if frame_index == 0:
            raise ValueError("No readable frames were found in the video.")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise ValueError("The annotated video could not be generated.")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_path), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                 str(compatible_path)],
                capture_output=True, text=True, check=True, timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg conversion exceeded the 5 minute limit.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg conversion failed: {exc.stderr[-1500:]}") from exc
        if not compatible_path.exists() or compatible_path.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce a playable annotated video.")
        return compatible_path.read_bytes(), pd.DataFrame(rows), fps
