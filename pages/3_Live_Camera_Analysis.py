
from datetime import datetime
from threading import Lock

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="LiftSafe | Live Camera",
    page_icon="📷",
    layout="wide",
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(
        np.clip(cosine, -1.0, 1.0)
    )))


def get_point(landmarks, landmark_name, width, height):
    point = landmarks[
        mp_pose.PoseLandmark[landmark_name].value
    ]

    if point.visibility < 0.5:
        return None

    return (int(point.x * width), int(point.y * height))


def calculate_measurements(landmarks, width, height):
    names = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
    ]

    points = {
        name: get_point(landmarks, name, width, height)
        for name in names
    }

    measurements = {}

    shoulders = [
        points[name] for name in
        ("LEFT_SHOULDER", "RIGHT_SHOULDER")
        if points[name] is not None
    ]
    hips = [
        points[name] for name in ("LEFT_HIP", "RIGHT_HIP")
        if points[name] is not None
    ]

    if shoulders and hips:
        shoulder = np.mean(shoulders, axis=0)
        hip = np.mean(hips, axis=0)

        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]

        measurements["Trunk inclination (°)"] = float(
            np.degrees(np.arctan2(abs(dx), abs(dy)))
        )

    for side in ("LEFT", "RIGHT"):
        hip = points[f"{side}_HIP"]
        knee = points[f"{side}_KNEE"]
        ankle = points[f"{side}_ANKLE"]

        if hip is not None and knee is not None and ankle is not None:
            angle = calculate_angle(hip, knee, ankle)
            if angle is not None:
                measurements[
                    f"{side.title()} knee angle (°)"
                ] = angle

    return measurements


class LiftSafeProcessor:
    """Processes browser camera frames in the WebRTC worker."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.lock = Lock()
        self.history = []
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        height, width = image.shape[:2]

        results = self.pose.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        measurements = {}

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
            measurements = calculate_measurements(
                results.pose_landmarks.landmark,
                width,
                height,
            )

        trunk = measurements.get("Trunk inclination (°)")
        review = trunk is not None and trunk >= self.threshold

        if trunk is None:
            label = "BODY NOT CLEARLY DETECTED"
            color = (0, 165, 255)
        elif review:
            label = "POSTURE REVIEW"
            color = (0, 0, 255)
        else:
            label = "MEASUREMENTS AVAILABLE"
            color = (0, 180, 0)

        cv2.rectangle(image, (10, 10), (width - 10, 58), color, -1)
        cv2.putText(
            image, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        y = 88
        for name, value in measurements.items():
            cv2.putText(
                image, f"{name}: {value:.1f}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            y += 28

        if measurements:
            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                **measurements,
                "review_flag": bool(review),
            }
            with self.lock:
                self.history.append(row)
                # Bound memory use during long sessions.
                if len(self.history) > 10000:
                    self.history = self.history[-10000:]

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )

    def get_history(self):
        with self.lock:
            return list(self.history)

    def close(self):
        self.pose.close()


st.title("📷 LiftSafe — Live Camera Analysis")
st.write(
    "Use your browser camera to view body landmarks and "
    "approximate posture angles. Keep your full body visible "
    "in a well-lit area."
)

st.warning(
    "**Research prototype:** These 2D measurements depend on "
    "camera position and landmark visibility. A threshold flag "
    "is for review only, not a validated ergonomic or "
    "safe/unsafe lifting assessment."
)

threshold = st.slider(
    "Trunk inclination review threshold (degrees)",
    min_value=10, max_value=80, value=45, step=5,
)

st.caption(
    "The threshold is experimental and is not a validated "
    "safety limit."
)

ctx = webrtc_streamer(
    key="liftsafe-browser-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: LiftSafeProcessor(threshold),
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.subheader("Measurement history")

if ctx.video_processor:
    history = ctx.video_processor.get_history()

    if history:
        log_df = pd.DataFrame(history)
        st.dataframe(log_df.tail(100), width="stretch")

        st.download_button(
            "Download measurement history (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="liftsafe_live_camera_measurements.csv",
            mime="text/csv",
            key="download_live_camera_history",
        )
    else:
        st.info(
            "Waiting for visible body landmarks. "
            "Allow camera access and keep your body in view."
        )
else:
    st.info(
        "Start the camera above and allow your browser "
        "to access the camera when prompted."
    )
