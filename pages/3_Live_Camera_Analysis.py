
from datetime import datetime, timezone
from threading import Lock

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.camera_measurements import calculate_landmark_confidence, calculate_measurements
from src.ui import (
    inject_app_styles,
    render_empty_state,
    render_page_header,
    render_section_heading,
    render_sidebar_brand,
    render_theme_toggle,
)

st.set_page_config(
    page_title="LiftSafe | Live Camera",
    page_icon="📷",
    layout="wide",
)
render_sidebar_brand()
render_theme_toggle()
inject_app_styles()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


POSE_LANDMARK_INDICES = {
    name: mp_pose.PoseLandmark[name].value
    for name in (
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    )
}


class LiftSafeProcessor(VideoProcessorBase):
    """Processes browser camera frames in the WebRTC worker."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.lock = Lock()
        self.history = []
        self._closed = False
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
            confidence = calculate_landmark_confidence(
                results.pose_landmarks.landmark, POSE_LANDMARK_INDICES
            )
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )
            measurements = calculate_measurements(
                results.pose_landmarks.landmark,
                width,
                height,
                POSE_LANDMARK_INDICES,
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
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "landmark_confidence": confidence,
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
        if not self._closed:
            self._closed = True
            self.pose.close()

    def on_ended(self):
        """Release MediaPipe resources when the browser stream stops."""
        self.close()


render_page_header(
    "Live camera analysis",
    "View pose landmarks and approximate posture angles from your browser camera.",
    eyebrow="LIFTSAFE  /  LIVE REVIEW",
)

with st.container(border=True):
    render_section_heading("Research prototype", "Experimental image-space measurements")
    st.warning(
        "These 2D measurements depend on camera position and landmark visibility. "
        "A threshold flag is for review only, not a validated ergonomic or "
        "safe/unsafe lifting assessment."
    )
    threshold = st.slider(
        "Trunk inclination review threshold (degrees)",
        min_value=10, max_value=80, value=45, step=5,
    )
    st.caption("The threshold is experimental and is not a validated safety limit.")

ctx = webrtc_streamer(
    key="liftsafe-browser-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: LiftSafeProcessor(threshold),
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_html_attrs={
        "autoPlay": True,
        "controls": False,
        "playsInline": True,
        "style": {
            "width": "100%",
            "aspectRatio": "16 / 9",
            "objectFit": "contain",
            "backgroundColor": "#080d19",
            "borderRadius": "16px",
        },
    },
    translations={
        "start": "Start camera",
        "stop": "Stop camera",
        "select_device": "Choose camera",
        "device_access_denied": "Camera permission was denied. Allow camera access in browser settings, then try again.",
        "device_not_available": "No camera was found. Connect or enable a camera and try again.",
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("Camera connected. Press Stop to end this session.")
else:
    with st.container(border=True):
        render_empty_state(
            "◎",
            "Camera preview is ready",
            "Use the Start camera control in the preview above and allow browser "
            "camera access. For remote use, open LiftSafe over HTTPS.",
        )
        st.info(
            "If the preview does not connect, check camera permission and network "
            "access. Some restricted networks require a TURN relay."
        )

render_section_heading("Measurement history", "Updates during a live camera session")

@st.fragment(run_every="1s")
def render_measurement_history():
    processor = ctx.video_processor
    if processor:
        history = processor.get_history()

        if history:
            log_df = pd.DataFrame(history)
            latest = log_df.iloc[-1]
            render_section_heading("Current measurements", "Most recent frame with visible landmarks")
            metric_columns = st.columns(4)
            metric_specs = (
                ("Trunk inclination (°)", "Trunk inclination", False),
                ("Left knee angle (°)", "Left knee", False),
                ("Right knee angle (°)", "Right knee", False),
                ("landmark_confidence", "Keypoint visibility", True),
            )
            for column, (key, label, is_confidence) in zip(metric_columns, metric_specs):
                value = latest.get(key)
                if pd.notna(value):
                    display = f"{value * 100:.0f}%" if is_confidence else f"{value:.1f}°"
                else:
                    display = "—"
                column.metric(label, display)
            st.dataframe(log_df.tail(100), width="stretch")

            st.download_button(
                "Download measurement history (CSV)",
                data=log_df.to_csv(index=False).encode("utf-8"),
                file_name="liftsafe_live_camera_measurements.csv",
                mime="text/csv",
                key="download_live_camera_history",
            )
        else:
            render_empty_state(
                "◉",
                "Waiting for visible landmarks",
                "Check camera permission and keep your full body in view."
            )
    else:
        render_empty_state(
            "◉",
            "Measurements will appear here",
            "Start the camera and keep visible landmarks in frame to begin the live history.",
        )


render_measurement_history()
