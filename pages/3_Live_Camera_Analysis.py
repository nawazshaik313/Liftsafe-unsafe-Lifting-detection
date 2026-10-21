
from datetime import datetime, timezone
from threading import Lock

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.camera_measurements import calculate_measurements
from src.ui import inject_app_styles

st.set_page_config(
    page_title="LiftSafe | Live Camera",
    page_icon="📷",
    layout="wide",
)
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
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
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


st.title("📷 LiftSafe — Live Camera Analysis")
st.write(
    "Use your browser camera to view body landmarks and "
    "approximate posture angles. Allow camera permission in your browser, "
    "keep your full body visible, and use a well-lit area."
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
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("Camera connected. Press Stop to end this session.")
else:
    st.info(
        "Camera is stopped. Press Start and allow browser camera access. "
        "If it does not connect, check permissions, HTTPS, and network "
        "restrictions; some networks require TURN."
    )

st.subheader("Measurement history")

# Keep the history in the Streamlit session after WebRTC stops its processor.
if "live_camera_history" not in st.session_state:
    st.session_state.live_camera_history = []
if "live_camera_history_timestamps" not in st.session_state:
    st.session_state.live_camera_history_timestamps = set()

@st.fragment(run_every="1s")
def render_measurement_history():
    processor = ctx.video_processor
    if processor:
        processor_history = processor.get_history()
        saved_history = st.session_state.live_camera_history
        seen_timestamps = st.session_state.live_camera_history_timestamps
        for row in processor_history:
            timestamp = row.get("timestamp")
            if timestamp not in seen_timestamps:
                saved_history.append(row)
                seen_timestamps.add(timestamp)
        if len(saved_history) > 10000:
            del saved_history[:-10000]

    history = st.session_state.live_camera_history
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
    elif processor:
        st.info(
            "Waiting for visible body landmarks. Check camera permission "
            "and keep your body in view."
        )
    else:
        st.caption("Start the camera to collect measurements.")


render_measurement_history()
