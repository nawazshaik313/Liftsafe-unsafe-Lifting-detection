from pathlib import Path

import tempfile

import subprocess

import cv2

import mediapipe as mp

import numpy as np

import pandas as pd

import streamlit as st

st.set_page_config(

    page_title="LiftSafe | Video Posture Analysis",

    page_icon="🏋️",

    layout="wide",

)

st.title("🎥 Video Posture Analysis")

st.write(

    "Upload a lifting video to estimate trunk inclination and hip/knee "

    "angles using MediaPipe pose landmarks."

)

st.warning(

    "**Experimental 2D analysis:** These image-space estimates are not "

    "clinically validated ergonomic measurements. They do not classify "

    "a lift as safe or unsafe and are not inputs to the INAIL classifier."

)

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils

def angle_from_vertical(a, b):

    dx = float(b[0] - a[0])

    dy = float(b[1] - a[1])

    return float(

        np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-8))

    )

def joint_angle(a, b, c):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b

    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:

        return float("nan")

    cosine = np.dot(ba, bc) / denominator

    return float(

        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    )

def process_video(uploaded_file, progress_bar, status, threshold=45):

    """Process the uploaded video and return an annotated MP4 and CSV data."""

    temp_dir = Path(tempfile.mkdtemp(prefix="liftsafe_"))

    input_path = temp_dir / "input.mp4"

    output_path = temp_dir / "annotated.mp4"

    csv_path = temp_dir / "pose_features.csv"

    input_path.write_bytes(uploaded_file.getvalue())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():

        raise ValueError(

            "OpenCV could not open this video. Please try another MP4."

        )

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

        str(output_path),

        cv2.VideoWriter_fourcc(*"mp4v"),

        fps,

        (width, height),

    )

    if not writer.isOpened():

        cap.release()

        raise ValueError("Could not create the annotated output video.")

    rows = []

    frame_index = 0

    try:

        with mp_pose.Pose(

            static_image_mode=False,

            model_complexity=1,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5,

        ) as pose:

            while True:

                success, frame = cap.read()

                if not success:

                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(rgb)

                row = {

                    "frame": frame_index,

                    "time_seconds": frame_index / fps,

                    "pose_detected": False,

                    "posture_review_flag": False,

                    "trunk_inclination_2d_deg": np.nan,

                    "left_hip_angle_2d_deg": np.nan,

                    "right_hip_angle_2d_deg": np.nan,

                    "left_knee_angle_2d_deg": np.nan,

                    "right_knee_angle_2d_deg": np.nan,

                }

                if results.pose_landmarks:

                    row["pose_detected"] = True

                    landmarks = results.pose_landmarks.landmark

                    def point(name):

                        landmark = landmarks[

                            mp_pose.PoseLandmark[name].value

                        ]

                        return np.array([

                            landmark.x * width,

                            landmark.y * height,

                        ])

                    left_shoulder = point("LEFT_SHOULDER")

                    right_shoulder = point("RIGHT_SHOULDER")

                    left_hip = point("LEFT_HIP")

                    right_hip = point("RIGHT_HIP")

                    left_knee = point("LEFT_KNEE")

                    right_knee = point("RIGHT_KNEE")

                    left_ankle = point("LEFT_ANKLE")

                    right_ankle = point("RIGHT_ANKLE")

                    shoulder = (left_shoulder + right_shoulder) / 2

                    hip = (left_hip + right_hip) / 2

                    row["trunk_inclination_2d_deg"] = (

                        angle_from_vertical(shoulder, hip)

                    )

                    row["left_knee_angle_2d_deg"] = joint_angle(

                        left_hip, left_knee, left_ankle

                    )

                    row["right_knee_angle_2d_deg"] = joint_angle(

                        right_hip, right_knee, right_ankle

                    )

                    row["left_hip_angle_2d_deg"] = joint_angle(

                        left_shoulder, left_hip, left_knee

                    )

                    row["right_hip_angle_2d_deg"] = joint_angle(

                        right_shoulder, right_hip, right_knee

                    )

                    mp_drawing.draw_landmarks(

                        frame,

                        results.pose_landmarks,

                        mp_pose.POSE_CONNECTIONS,

                    )
                    trunk_angle = row["trunk_inclination_2d_deg"]
                    if np.isfinite(trunk_angle) and trunk_angle >= threshold:
                        row["posture_review_flag"] = True
                        # Experimental review flag, not a diagnosis.
                        cv2.rectangle(frame, (0, 0), (width, 72), (0, 140, 255), -1)
                        cv2.putText(frame, "POSTURE REVIEW", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Trunk inclination: {trunk_angle:.1f} deg (threshold {threshold} deg)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


                rows.append(row)

                writer.write(frame)

                frame_index += 1

                if total_frames > 0:

                    progress_bar.progress(

                        min(frame_index / total_frames, 1.0)

                    )

                    status.caption(

                        f"Processed {frame_index} of {total_frames} frames"

                    )

    finally:

        cap.release()

        writer.release()

    if frame_index == 0:

        raise ValueError("No readable frames were found in the video.")

    if not output_path.exists() or output_path.stat().st_size == 0:

        raise ValueError("The annotated video could not be generated.")

    data = pd.DataFrame(rows)

    compatible_path = temp_dir / "annotated_video_h264.mp4"

    try:

        result = subprocess.run(

            [

                "ffmpeg", "-y",

                "-i", str(output_path),

                "-c:v", "libx264",

                "-pix_fmt", "yuv420p",

                "-movflags", "+faststart",

                "-an",

                str(compatible_path),

            ],

            capture_output=True,

            text=True,

            check=True,

        )

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(

            f"FFmpeg conversion failed: {exc.stderr[-1500:]}"

        ) from exc

    output_path = compatible_path

    data.to_csv(csv_path, index=False)

    return output_path, data, fps, temp_dir

uploaded_video = st.file_uploader(

    "Upload a lifting video",

    type=["mp4", "mov", "avi"],

    help="For best results, use a clear side-view video with the person's "

         "shoulders, hips, knees, and ankles visible.",

)


threshold = st.slider(
    "Experimental trunk-inclination review threshold (degrees)",
    min_value=10,
    max_value=90,
    value=45,
    step=5,
    help=(
        "A configurable research threshold, not a validated ergonomic "
        "safety limit. Click Analyze video again after changing it."
    ),
)

if uploaded_video is not None:

    st.caption(f"Selected file: {uploaded_video.name}")


    if st.button("Analyze video", type="primary"):

        progress = st.progress(0.0)

        status = st.empty()

        try:

            with st.spinner("Estimating pose landmarks..."):

                video_path, data, fps, temp_dir = process_video(
                    uploaded_video, progress, status, threshold
                )

            st.session_state["liftsafe_video_path"] = str(video_path)

            st.session_state["liftsafe_video_csv"] = data

            st.session_state["liftsafe_video_fps"] = fps

            st.session_state["liftsafe_video_temp_dir"] = str(temp_dir)

            st.success("Video analysis completed.")

        except Exception as exc:

            st.error(f"Video analysis failed: {exc}")

if "liftsafe_video_csv" in st.session_state:

    data = st.session_state["liftsafe_video_csv"]

    video_path = Path(st.session_state["liftsafe_video_path"])

    if video_path.exists():
        st.subheader("Annotated video")
        st.video(video_path.read_bytes())
    else:
        st.error("The annotated video file could not be found. Please analyze the video again.")

    total_frames = len(data)

    detected = int(data["pose_detected"].sum())

    coverage = detected / total_frames * 100 if total_frames else 0

    st.subheader("Analysis summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Frames processed", total_frames)

    c2.metric("Pose detected", f"{detected} / {total_frames}")

    c3.metric("Detection coverage", f"{coverage:.1f}%")

    st.subheader("Posture angles over time")

    angle_columns = {

        "trunk_inclination_2d_deg": "Trunk inclination",

        "left_hip_angle_2d_deg": "Left hip angle",

        "right_hip_angle_2d_deg": "Right hip angle",

        "left_knee_angle_2d_deg": "Left knee angle",

        "right_knee_angle_2d_deg": "Right knee angle",

    }

    available_columns = list(angle_columns.keys())

    selected_columns = st.multiselect(

        "Choose measurements",

        options=available_columns,

        default=[

            "trunk_inclination_2d_deg",

            "left_knee_angle_2d_deg",

            "right_knee_angle_2d_deg",

        ],

        format_func=lambda value: angle_columns[value],

    )

    if selected_columns:

        chart_data = data.set_index("time_seconds")[selected_columns]

        chart_data = chart_data.rename(columns=angle_columns)

        st.line_chart(chart_data, width="stretch")

    else:

        st.info("Select at least one measurement to display the chart.")

    st.subheader("Angle summary (degrees)")

    summary = data[available_columns].describe().T

    summary = summary.rename(index=angle_columns)

    summary = summary[["mean", "std", "min", "50%", "max"]].round(2)

    summary.columns = ["Mean", "Std. deviation", "Minimum", "Median", "Maximum"]

    st.dataframe(summary, width="stretch")

    st.subheader("Experimental posture screening")

    st.caption(

        "Research-only screening based on approximate 2D trunk inclination. "

        "This is not a validated safe/unsafe classifier or medical assessment."

    )

    valid_trunk = data["trunk_inclination_2d_deg"].dropna()

    if valid_trunk.empty:

        st.warning("No valid trunk-angle measurements are available.")

    else:

        flagged = valid_trunk >= threshold

        flagged_count = int(flagged.sum())

        valid_count = len(valid_trunk)

        flagged_percent = flagged_count / valid_count * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("Frames flagged for review", f"{flagged_count} / {valid_count}")

        c2.metric("Flagged frames", f"{flagged_percent:.1f}%")

        c3.metric("Review threshold", f"{threshold}°")

        if flagged_count:

            st.warning(

                "Posture review recommended: trunk inclination met or exceeded "

                "the selected experimental threshold in some frames."

            )

        else:

            st.success(

                "No frames met the selected experimental trunk-angle threshold. "

                "This does not establish that the lift is safe."

            )

        screening_data = data.loc[

            valid_trunk.index,

            ["frame", "time_seconds", "trunk_inclination_2d_deg"],

        ].copy()

        screening_data["review_flag"] = (

            screening_data["trunk_inclination_2d_deg"] >= threshold

        )

        screening_data["review_flag"] = screening_data["review_flag"].map(

            {True: "Review recommended", False: "Threshold not exceeded"}

        )

        # Timeline based on the flags recorded during video processing
        st.subheader("Posture review timeline")

        timeline_data = data[
            [
                "frame",
                "time_seconds",
                "trunk_inclination_2d_deg",
                "posture_review_flag",
            ]
        ].copy()

        timeline_data["posture_review_flag"] = (
            timeline_data["posture_review_flag"]
            .fillna(False)
            .astype(bool)
        )

        timeline_data["review_value"] = (
            timeline_data["posture_review_flag"].astype(int)
        )

        st.caption(
            "1 = posture review recommended; 0 = threshold not exceeded. "
            "This is an experimental screening indicator, not a safety verdict."
        )

        st.line_chart(
            timeline_data.set_index("time_seconds")[["review_value"]],
            width="stretch",
        )

        # Group consecutive flagged frames into intervals
        timeline_data["interval_id"] = (
            timeline_data["posture_review_flag"]
            .ne(
                timeline_data["posture_review_flag"].shift(
                    fill_value=False
                )
            )
            .cumsum()
        )

        flagged_frames = timeline_data[
            timeline_data["posture_review_flag"]
        ].copy()

        if flagged_frames.empty:
            st.info("No frames were flagged in the analyzed video.")
        else:
            intervals = (
                flagged_frames.groupby("interval_id")
                .agg(
                    start_frame=("frame", "min"),
                    end_frame=("frame", "max"),
                    start_time_seconds=("time_seconds", "min"),
                    end_time_seconds=("time_seconds", "max"),
                    peak_trunk_angle_deg=(
                        "trunk_inclination_2d_deg", "max"
                    ),
                )
                .reset_index(drop=True)
            )

            frame_rate = float(st.session_state.get("liftsafe_video_fps", 30.0))
            intervals["duration_seconds"] = (
                intervals["end_time_seconds"]
                - intervals["start_time_seconds"]
                + 1.0 / frame_rate
            ).round(3)

            for column in [
                "start_time_seconds",
                "end_time_seconds",
                "peak_trunk_angle_deg",
            ]:
                intervals[column] = intervals[column].round(2)


            st.write("Continuous intervals flagged for review")

            intervals["interval_label"] = intervals.apply(
                lambda row: (
                    f"{row['start_time_seconds']:.2f}s–"
                    f"{row['end_time_seconds']:.2f}s | "
                    f"Peak {row['peak_trunk_angle_deg']:.2f}°"
                ),
                axis=1,
            )

            selected_label = st.selectbox(
                "Select a warning interval to inspect",
                options=intervals["interval_label"].tolist(),
            )

            selected_interval = intervals[
                intervals["interval_label"] == selected_label
            ].iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Start time",
                f"{selected_interval['start_time_seconds']:.2f} s",
            )
            c2.metric(
                "End time",
                f"{selected_interval['end_time_seconds']:.2f} s",
            )
            c3.metric(
                "Peak trunk inclination",
                f"{selected_interval['peak_trunk_angle_deg']:.2f}°",
            )

            start_frame = int(selected_interval["start_frame"])
            end_frame = int(selected_interval["end_frame"])

            selected_frames = data[
                data["frame"].between(start_frame, end_frame)
            ].copy()

            st.write(
                f"Inspecting frames {start_frame}–{end_frame} "
                f"({len(selected_frames)} frames)."
            )

            st.dataframe(
                selected_frames[
                    [
                        "frame",
                        "time_seconds",
                        "trunk_inclination_2d_deg",
                        "posture_review_flag",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download selected interval (CSV)",
                key="download_selected_interval",
                data=selected_frames.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"liftsafe_interval_{start_frame}_{end_frame}.csv"
                ),
                mime="text/csv",
            )

            st.download_button(
                "Download warning intervals (CSV)",
                key="download_warning_intervals",
                data=intervals.drop(
                    columns=["interval_label"]
                ).to_csv(index=False).encode("utf-8"),
                file_name="liftsafe_warning_intervals.csv",
                mime="text/csv",
            )
            st.dataframe(
                intervals,
                width="stretch",
                hide_index=True,
            )
        st.line_chart(

            screening_data.set_index("time_seconds")[

                ["trunk_inclination_2d_deg"]

            ],

            width="stretch",

        )

        with st.expander("View flagged frames"):

            flagged_rows = screening_data[

                screening_data["review_flag"] == "Review recommended"

            ]

            if flagged_rows.empty:

                st.info("No frames exceeded the selected threshold.")

            else:

                st.dataframe(flagged_rows, width="stretch", hide_index=True)

        st.download_button(

            "Download posture screening report (CSV)",
                key="download_screening_report",

            data=screening_data.to_csv(index=False).encode("utf-8"),

            file_name="liftsafe_posture_screening.csv",

            mime="text/csv",

        )

    with st.expander("View frame-by-frame measurements"):

        st.dataframe(data, width="stretch", hide_index=True)

    st.download_button(

        "Download pose measurements (CSV)",
                key="download_pose_measurements",

        data=data.to_csv(index=False).encode("utf-8"),

        file_name="liftsafe_pose_features.csv",

        mime="text/csv",

    )

    st.info(

        "Angles are approximate 2D image-space measurements. Camera angle, "

        "occlusion, clothing, and landmark errors can affect them. A high "

        "pose-detection rate does not guarantee accurate measurements."

    )

else:

    st.info(

        "Upload a video and select **Analyze video** to view pose landmarks, "

        "angle charts, summary statistics, and a CSV report."

    )
