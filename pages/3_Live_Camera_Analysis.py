
from datetime import datetime, timezone
import logging
from threading import Lock

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.pose_estimation import (
    PoseTargetTracker,
    analyze_pose_frame,
    create_pose_estimator,
    draw_pose_landmarks,
)
from src.ui import inject_app_styles, render_sidebar_brand, render_theme_toggle

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LiftSafe | Live Camera",
    page_icon="📷",
    layout="wide",
)
render_sidebar_brand()
render_theme_toggle()
inject_app_styles()

class LiftSafeProcessor(VideoProcessorBase):
    """Processes browser camera frames in the WebRTC worker."""

    def __init__(self, threshold, target_region="auto"):
        self.threshold = threshold
        self.lock = Lock()
        self.history = []
        self.target_tracker = PoseTargetTracker(preferred_region=target_region)
        self._closed = False
        self.latest = {
            "state": "starting",
            "timestamp": None,
            "visibility": None,
            "measurements": {},
            "review": None,
            "error": None,
        }
        try:
            self.pose = create_pose_estimator()
        except Exception as exc:
            self.pose = None
            self.latest.update(state="inference_error", error=f"{type(exc).__name__}: {exc}")

    def recv(self, frame):
        image = None
        try:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)
            height, width = image.shape[:2]
            if width > 960:
                scale = 960 / width
                image = cv2.resize(
                    image, (960, max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                height, width = image.shape[:2]

            if self.pose is None:
                raise RuntimeError("MediaPipe Pose could not be initialized.")
            pose_result = analyze_pose_frame(self.pose, image, self.target_tracker)
            measurements = pose_result.measurements
            trunk = measurements.get("Trunk inclination (°)")
            review = None if trunk is None else trunk >= self.threshold
            timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            latest = {
                "state": pose_result.state,
                "timestamp": timestamp,
                "visibility": pose_result.visibility,
                "measurements": dict(measurements),
                "review": review,
                "error": None,
                "person_count": pose_result.person_count,
                "target_id": pose_result.target_id,
                "target_score": pose_result.target_score,
            }
            if pose_result.landmarks is not None:
                draw_pose_landmarks(image, pose_result.landmarks.landmark)
            if pose_result.state == "target_ambiguous":
                label, color = "TARGET AMBIGUOUS - FRAME ONE PERSON", (0, 165, 255)
            elif pose_result.state == "target_lost":
                label, color = "TRACKED PERSON TEMPORARILY LOST", (0, 165, 255)
            elif not pose_result.pose_detected:
                label, color = "NO PERSON DETECTED", (0, 165, 255)
            elif trunk is None:
                label, color = "POSE DETECTED - VISIBILITY INSUFFICIENT", (0, 165, 255)
            elif review:
                label, color = "REVIEW THRESHOLD EXCEEDED", (0, 0, 255)
            else:
                label, color = "WITHIN CONFIGURED REVIEW THRESHOLD", (0, 150, 0)

            cv2.rectangle(image, (10, 10), (width - 10, 58), color, -1)
            cv2.putText(
                image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
            y = 88
            for name, value in measurements.items():
                cv2.putText(
                    image, f"{name}: {value:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )
                y += 28

            with self.lock:
                self.latest = latest
                if trunk is not None:
                    self.history.append({
                        "timestamp": timestamp,
                        "landmark_confidence": pose_result.visibility,
                        **measurements,
                        "review_flag": bool(review),
                    })
                    if len(self.history) > 10000:
                        self.history = self.history[-10000:]
        except Exception as exc:
            logger.exception("Live camera frame processing failed")
            if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                image = np.zeros((360, 640, 3), dtype=np.uint8)
            height, width = image.shape[:2]
            with self.lock:
                self.latest = {
                    "state": "inference_error",
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    "visibility": None,
                    "measurements": {},
                    "review": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            cv2.rectangle(image, (10, 10), (width - 10, 58), (0, 0, 180), -1)
            cv2.putText(
                image, "POSE DETECTION ERROR", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_history(self):
        with self.lock:
            return list(self.history)

    def get_latest(self):
        with self.lock:
            return dict(self.latest)

    def close(self):
        if not self._closed and self.pose is not None:
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

target_region = st.selectbox(
    "Target person's starting position",
    options=["auto", "left", "center", "right"],
    format_func=lambda value: {
        "auto": "Automatic selection (recommended)",
        "left": "Left third",
        "center": "Center third",
        "right": "Right third",
    }[value],
    help="Choose the lifter's initial position when multiple people appear in view.",
)

st.caption(
    "The threshold is experimental and is not a validated "
    "safety limit."
)

ctx = webrtc_streamer(
    key="liftsafe-browser-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: LiftSafeProcessor(threshold, target_region),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 960},
            "height": {"ideal": 540},
            "frameRate": {"ideal": 15, "max": 24},
        },
        "audio": False,
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

st.subheader("Live pose status and measurement history")

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

    with st.container(border=True):
        st.markdown("### Current camera frame")
        if not ctx.state.playing:
            st.info(
                "Camera stopped. Press Start and allow browser camera access. "
                "If permission was denied, allow it in browser settings and retry. "
                "Use HTTPS when opening LiftSafe remotely."
            )
        elif processor is None:
            st.info("Camera connected. Waiting for the pose processor to start.")
        else:
            latest = processor.get_latest()
            state = latest.get("state")
            if state == "no_person":
                st.warning(
                    "No person detected — move into the camera frame and ensure your full upper body is visible."
                )
            elif state == "target_ambiguous":
                st.warning(
                    "Several people are visible and the lifting target is ambiguous. "
                    "Frame the lifter clearly or move others out of view; measurements are paused."
                )
            elif state == "target_lost":
                st.warning(
                    "The tracked person is temporarily lost. Keep them visible; measurements are paused instead of switching to another person."
                )
            elif state == "incomplete_landmarks":
                st.warning(
                    "Pose measurements unavailable — improve lighting, camera angle, or body visibility."
                )
            elif state == "inference_error":
                st.error("Pose detection could not process the latest camera frame.")
                if latest.get("error"):
                    with st.expander("Pose detection details"):
                        st.code(latest["error"])
            elif state == "starting":
                st.info("Camera connected. Waiting for the first frame.")
            elif state == "measurements_available":
                measurements = latest.get("measurements", {})
                trunk = measurements.get("Trunk inclination (°)")
                review = latest.get("review")
                if review is True:
                    st.warning(
                        f"Review threshold exceeded: trunk inclination {trunk:.1f}° "
                        f"is at or above the configured {threshold}° threshold."
                    )
                else:
                    st.success(
                        f"Pose detected. Trunk inclination {trunk:.1f}° is within "
                        f"the configured {threshold}° review threshold."
                    )
                metric_columns = st.columns(4)
                metrics = (
                    ("Trunk inclination (°)", "Trunk inclination"),
                    ("Left knee angle (°)", "Left knee"),
                    ("Right knee angle (°)", "Right knee"),
                    ("Left hip angle (°)", "Left hip"),
                )
                for column, (key, label) in zip(metric_columns, metrics):
                    value = measurements.get(key)
                    column.metric(label, f"{value:.1f}°" if value is not None else "—")
                visibility = latest.get("visibility")
                if visibility is not None:
                    st.caption(f"Keypoint visibility/tracking quality: {visibility:.0%}")
                if latest.get("person_count", 0) > 1:
                    st.caption(
                        f"Tracking person {latest.get('target_id')} among "
                        f"{latest['person_count']} detected people; heuristic selection "
                        f"score {latest.get('target_score') or 0:.2f}."
                    )
                st.caption(
                    f"Experimental threshold: {threshold}°. This image-space review is not a validated ergonomic assessment."
                )
            if latest.get("timestamp"):
                st.caption(f"Latest frame: {latest['timestamp']} UTC")

    history = st.session_state.live_camera_history
    if history:
        st.markdown("### Measurement history")
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
        st.caption("Pose measurements will be logged here when trunk landmarks are reliable.")


render_measurement_history()
