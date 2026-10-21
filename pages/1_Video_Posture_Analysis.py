import numpy as np
import pandas as pd
import streamlit as st

from src.charts import line_figure, render_glass_chart
from src.ui import (
    inject_app_styles,
    render_empty_state,
    render_page_header,
    render_section_heading,
    render_sidebar_brand,
    render_theme_toggle,
)
from src.video_processing import process_video

st.set_page_config(

    page_title="LiftSafe | Video Posture Analysis",

    page_icon="🏋️",

    layout="wide",

)
render_sidebar_brand()
render_theme_toggle()
inject_app_styles()

render_page_header(
    "Video posture analysis",
    "Estimate image-space joint angles from a clear side-view lifting clip.",
    eyebrow="LIFTSAFE  /  VIDEO REVIEW",
)

st.warning(

    "**Experimental 2D analysis:** These image-space estimates are not "

    "clinically validated ergonomic measurements. They do not classify "

    "a lift as safe or unsafe and are not inputs to the INAIL classifier."

)


with st.container(border=True):
    render_section_heading("Upload and configure", "MP4, MOV, or AVI · maximum 500 MB")
    uploaded_video = st.file_uploader(
        "Upload a lifting video",
        type=["mp4", "mov", "avi"],
        help=(
            "For best results, use a clear side-view video with the person's "
            "shoulders, hips, knees, and ankles visible."
        ),
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

                video_bytes, data, fps = process_video(
                    uploaded_video, progress, status, threshold
                )

            st.session_state["liftsafe_video_bytes"] = video_bytes

            st.session_state["liftsafe_video_csv"] = data

            st.session_state["liftsafe_video_fps"] = fps

            st.success("Video analysis completed.")

        except Exception as exc:

            st.error(f"Video analysis failed: {exc}")

if "liftsafe_video_csv" in st.session_state:

    data = st.session_state["liftsafe_video_csv"]

    video_bytes = st.session_state.get("liftsafe_video_bytes")

    if video_bytes:
        render_section_heading("Annotated video", "Pose overlay and review markers")
        with st.container(border=True):
            st.video(video_bytes)
    else:
        st.error("The annotated video is unavailable. Please analyze the video again.")

    total_frames = len(data)

    detected = int(data["pose_detected"].sum())

    coverage = detected / total_frames * 100 if total_frames else 0

    render_section_heading("Analysis summary", "Frame coverage and landmark visibility")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Frames processed", total_frames)

    c2.metric("Pose detected", f"{detected} / {total_frames}")

    c3.metric("Detection coverage", f"{coverage:.1f}%")

    if "landmark_confidence" in data.columns:
        confidence = data["landmark_confidence"].dropna()
        confidence_label = f"{confidence.mean() * 100:.0f}%" if not confidence.empty else "—"
        c4.metric("Mean keypoint visibility", confidence_label)

    render_section_heading("Posture angles over time", "Approximate 2D geometry in degrees")

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

        render_glass_chart(
            line_figure(
                chart_data,
                "time_seconds",
                list(chart_data.columns),
                x_title="Time (seconds)",
                y_title="Angle (degrees)",
                height=360,
            ),
            key="video_joint_angles",
        )

    else:

        st.info("Select at least one measurement to display the chart.")

    render_section_heading("Angle summary", "Degrees across frames with visible landmarks")

    summary = data[available_columns].describe().T

    summary = summary.rename(index=angle_columns)

    summary = summary[["mean", "std", "min", "50%", "max"]].round(2)

    summary.columns = ["Mean", "Std. deviation", "Minimum", "Median", "Maximum"]

    st.dataframe(summary, width="stretch")

    render_section_heading("Experimental posture screening", "Threshold-based review only")

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
        render_section_heading("Posture review timeline", "Frames that met the selected experimental threshold")

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

        render_glass_chart(
            line_figure(
                timeline_data,
                "time_seconds",
                ["review_value"],
                labels={"review_value": "Review flag"},
                x_title="Time (seconds)",
                y_title="Experimental review flag (0–1)",
                height=260,
                step=True,
            ),
            key="video_review_timeline",
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
        render_glass_chart(
            line_figure(
                screening_data,
                "time_seconds",
                ["trunk_inclination_2d_deg"],
                labels={"trunk_inclination_2d_deg": "Trunk inclination"},
                x_title="Time (seconds)",
                y_title="Angle (degrees)",
                threshold=float(threshold),
                height=300,
            ),
            key="video_trunk_screening",
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
    with st.container(border=True):
        render_empty_state(
            "↥",
            "No video analyzed yet",
            "Upload a lifting clip above and select Analyze video to view landmarks, "
            "angle charts, summary statistics, and downloadable measurements.",
        )
