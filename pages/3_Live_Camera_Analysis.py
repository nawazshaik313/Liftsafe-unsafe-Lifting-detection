
from datetime import datetime, timezone
import logging
from threading import Lock

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.camera_measurements import is_fresh_timestamp
from src.charts import line_figure, render_glass_chart
from src.pose_estimation import (
    PoseTargetTracker,
    analyze_pose_frame,
    create_pose_estimator,
    draw_candidate_boxes,
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
        self.target_lock = Lock()
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
            with self.target_lock:
                pose_result = analyze_pose_frame(self.pose, image, self.target_tracker)
            measurements = pose_result.measurements
            trunk = measurements.get("Trunk inclination (°)")
            trunk_valid = trunk is not None and np.isfinite(trunk)
            review = bool(trunk >= self.threshold) if trunk_valid else None
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
                "candidate_number": pose_result.candidate_number,
                "selection_reason": pose_result.selection_reason,
                "candidates": pose_result.candidates,
            }
            if pose_result.candidates:
                draw_candidate_boxes(image, pose_result.candidates, pose_result.candidate_number)
            if pose_result.landmarks is not None:
                draw_pose_landmarks(image, pose_result.landmarks.landmark)
            logger.debug(
                "live prediction pose_model=MediaPipe Tasks PoseLandmarker Lite (num_poses=4) pose_state=%s candidates=%s selected_track=%s candidate=%s bbox=%s reason=%s overlay_same_pose=%s classifier_model=none cnn_invoked=false threshold_review_model=trunk-angle-rule threshold=%s review=%s measurements=%s",
                pose_result.state,
                pose_result.candidates, pose_result.target_id,
                pose_result.candidate_number, pose_result.target_bbox,
                pose_result.selection_reason, pose_result.landmarks is not None,
                self.threshold, review, measurements,
            )
            if pose_result.state == "target_ambiguous":
                label, color = "TARGET AMBIGUOUS - FRAME ONE PERSON", (0, 165, 255)
            elif pose_result.state == "target_lost":
                label, color = "TRACKED PERSON TEMPORARILY LOST", (0, 165, 255)
            elif not pose_result.pose_detected:
                label, color = "NO PERSON DETECTED", (0, 165, 255)
            elif not trunk_valid:
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
                if trunk_valid:
                    self.history.append({
                        "timestamp": timestamp,
                        "landmark_confidence": pose_result.visibility,
                        "review_threshold": self.threshold,
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

    def lock_candidate(self, candidate_number):
        with self.target_lock:
            self.target_tracker.request_manual_candidate(candidate_number)

    def reset_target(self):
        with self.target_lock:
            self.target_tracker.reset()

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
            timestamp = latest.get("timestamp")
            stale = (
                ctx.state.playing
                and timestamp is not None
                and not is_fresh_timestamp(timestamp)
            )
            if stale:
                try:
                    age_seconds = max(
                        0.0,
                        (datetime.now(timezone.utc) - datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )).total_seconds(),
                    )
                except (TypeError, ValueError):
                    age_seconds = None
                age_text = f" ({age_seconds:.1f}s old)" if age_seconds is not None else ""
                st.warning(
                    "Camera frames are stale" + age_text
                    + ". The latest measurement is not current; check the camera stream."
                )
            elif state == "no_person":
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
                elif review is False:
                    st.success(
                        f"Pose detected. Trunk inclination {trunk:.1f}° is within "
                        f"the configured {threshold}° review threshold."
                    )
                else:
                    st.warning("Trunk measurement is unavailable or non-finite for this frame.")
                metric_columns = st.columns(4)
                metrics = (
                    ("Trunk inclination (°)", "Trunk inclination"),
                    ("Left knee angle (°)", "Left knee"),
                    ("Right knee angle (°)", "Right knee"),
                    ("Left hip angle (°)", "Left hip"),
                )
                for column, (key, label) in zip(metric_columns, metrics):
                    value = measurements.get(key)
                    column.metric(
                        label,
                        f"{value:.1f}°" if value is not None and np.isfinite(value) else "—",
                    )
                visibility = latest.get("visibility")
                if visibility is not None:
                    st.caption(f"Keypoint visibility/tracking quality: {visibility:.0%}")
                if latest.get("person_count", 0) > 1:
                    st.caption(
                        f"Tracked person ID {latest.get('target_id')} via candidate "
                        f"P{latest.get('candidate_number')}; selection score "
                        f"{latest.get('target_score') or 0:.2f}. "
                        f"{latest.get('selection_reason')}"
                    )
                st.caption(
                    f"Experimental threshold: {threshold}°. This image-space review is not a validated ergonomic assessment."
                )
            if latest.get("timestamp"):
                st.caption(f"Latest frame: {latest['timestamp']} UTC")
            candidates = latest.get("candidates", ())
            if candidates and not stale:
                st.caption(
                    "Camera candidates are numbered left-to-right. Lock the lifter if "
                    "automatic tracking chose a different person."
                )
                candidate_columns = st.columns(min(len(candidates), 4))
                for column, candidate in zip(candidate_columns, candidates):
                    number = candidate["candidate_number"]
                    if column.button(
                        f"Lock Person {number}", key=f"lock_live_candidate_{number}"
                    ):
                        processor.lock_candidate(number)
            if st.button("Reset target lock", key="reset_live_target"):
                processor.reset_target()

    history = st.session_state.live_camera_history
    if history:
        st.markdown("### Measurement history")
        log_df = pd.DataFrame(history)
        st.caption(
            "Only frames with a finite trunk-angle measurement are included. "
            "Review flags are experimental threshold checks, not safety labels."
        )
        valid_count = int(log_df["Trunk inclination (°)"].notna().sum())
        flagged_events = log_df[log_df["review_flag"].fillna(False).astype(bool)]
        summary_columns = st.columns(3)
        summary_columns[0].metric("Valid measured frames", valid_count)
        summary_columns[1].metric("Frames flagged for review", len(flagged_events))
        summary_columns[2].metric(
            "Latest review threshold",
            f"{float(log_df['review_threshold'].iloc[-1]):g}°"
            if "review_threshold" in log_df and not log_df.empty else f"{threshold}°",
        )

        log_df["frame_time"] = pd.to_datetime(log_df["timestamp"], utc=True, errors="coerce")
        log_df["time_seconds"] = (
            log_df["frame_time"] - log_df["frame_time"].dropna().min()
        ).dt.total_seconds()
        chart_columns = ["Trunk inclination (°)"]
        chart_labels = {"Trunk inclination (°)": "Trunk inclination"}
        if "review_threshold" in log_df:
            chart_columns.append("review_threshold")
            chart_labels["review_threshold"] = "Configured review threshold"
        render_glass_chart(
            line_figure(
                log_df.dropna(subset=["time_seconds"]),
                "time_seconds",
                chart_columns,
                labels=chart_labels,
                x_title="Time since first valid measurement (seconds)",
                y_title="Angle (degrees)",
                height=300,
            ),
            key="live_camera_angle_history",
        )

        st.dataframe(log_df.tail(100), width="stretch")

        st.markdown("### Time-stamped review events")
        if flagged_events.empty:
            st.info("No valid camera measurements have met the configured review threshold yet.")
        else:
            event_columns = [
                column for column in (
                    "timestamp", "Trunk inclination (°)", "review_threshold",
                    "landmark_confidence",
                ) if column in flagged_events.columns
            ]
            st.dataframe(flagged_events[event_columns], width="stretch", hide_index=True)

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
