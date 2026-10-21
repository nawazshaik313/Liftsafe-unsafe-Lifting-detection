"""Sequential uploaded-video decoding and shared MediaPipe pose analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import BinaryIO, Iterator

import cv2
import numpy as np
import pandas as pd

from src.pose_estimation import (
    PoseTargetTracker,
    analyze_pose_frame,
    create_pose_estimator,
    detect_pose_candidates,
    draw_candidate_boxes,
    draw_pose_landmarks,
)


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_VIDEO_PROCESSING_SECONDS = 600
SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi"}


class VideoProcessingError(RuntimeError):
    """A user-safe video failure with its processing stage and diagnostic detail."""

    def __init__(self, stage: str, user_message: str, details: str = ""):
        super().__init__(user_message)
        self.stage = stage
        self.user_message = user_message
        self.details = details


def _safe_suffix(name: str, payload: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return suffix
    if len(payload) >= 12 and payload[4:8] == b"ftyp":
        return ".mp4"
    if payload.startswith(b"RIFF") and payload[8:12] == b"AVI ":
        return ".avi"
    return ".mp4"


def _probe_video(path: Path) -> dict:
    """Read stream metadata only when OpenCV cannot provide usable dimensions."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise VideoProcessingError(
            "decoder_open",
            "The video could not be opened by the installed decoder.",
            "OpenCV could not open the input, and ffprobe is unavailable for a fallback check.",
        )
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=30
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise ValueError("No video stream was found.")
        return streams[0]
    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise VideoProcessingError(
            "decoder_open",
            "This file does not contain a readable video stream. Try exporting it as MP4, MOV, or AVI.",
            str(detail),
        ) from exc


def _parse_rate(value) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        numerator, denominator = str(value).split("/", maxsplit=1)
        rate = float(numerator) / float(denominator)
        return rate if np.isfinite(rate) and rate > 0 else None
    except (ValueError, ZeroDivisionError):
        try:
            rate = float(value)
            return rate if np.isfinite(rate) and rate > 0 else None
        except (TypeError, ValueError):
            return None


class SequentialVideoReader:
    """Read frames in order, falling back to FFmpeg only if OpenCV yields none."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.decoder = "opencv"
        self._cap = None
        self._first_frame = None
        self._ffmpeg_process = None
        self._ffmpeg_stderr = None
        self._ffmpeg_error = ""

    def __enter__(self):
        cap = cv2.VideoCapture(str(self.path))
        try:
            opened = cap.isOpened()
            first_ok, first_frame = cap.read() if opened else (False, None)
        except Exception:
            cap.release()
            raise
        if first_ok and self._valid_frame(first_frame):
            self._cap = cap
            self._first_frame = first_frame
            self.width = int(first_frame.shape[1])
            self.height = int(first_frame.shape[0])
            self.total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.fps = self._valid_fps(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            return self

        cap.release()
        open_reason = "OpenCV opened the file but could not decode its first frame."
        if not opened:
            open_reason = "OpenCV could not open the uploaded file."
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            stage = "decoder_open" if not opened else "decode_zero_frames"
            raise VideoProcessingError(
                stage,
                "No video frames could be decoded. The file may use an unsupported codec or be damaged.",
                open_reason + " FFmpeg is not installed, so no fallback decoder is available.",
            )

        probe = _probe_video(self.path)
        try:
            self.width = int(probe.get("width", 0))
            self.height = int(probe.get("height", 0))
            self.fps = _parse_rate(probe.get("avg_frame_rate")) or _parse_rate(
                probe.get("r_frame_rate")
            ) or 30.0
            try:
                self.total_frames = max(0, int(probe.get("nb_frames", 0)))
            except (TypeError, ValueError):
                self.total_frames = 0
            if self.width <= 0 or self.height <= 0:
                raise ValueError("ffprobe returned invalid video dimensions")

            self._ffmpeg_stderr = tempfile.TemporaryFile(mode="w+b")
            self._ffmpeg_process = subprocess.Popen(
                [
                    ffmpeg, "-nostdin", "-v", "error", "-i", str(self.path),
                    "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo",
                    "-pix_fmt", "bgr24", "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=self._ffmpeg_stderr,
            )
            self.decoder = "ffmpeg"
            return self
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.close()
            raise VideoProcessingError(
                "decoder_open",
                "This video could not be opened by either available decoder. Try re-exporting it as MP4, MOV, or AVI.",
                f"{open_reason} FFmpeg fallback setup failed: {exc}",
            ) from exc

    @staticmethod
    def _valid_frame(frame) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.size > 0
        )

    @staticmethod
    def _valid_fps(value) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is not None:
            frame = self._first_frame
            while frame is not None:
                if not self._valid_frame(frame):
                    raise VideoProcessingError(
                        "decode_frame", "A damaged frame stopped video decoding.",
                        "OpenCV returned an empty or malformed frame.",
                    )
                yield frame
                ok, frame = self._cap.read()
                if not ok:
                    break
            return

        if self._ffmpeg_process is None or self._ffmpeg_process.stdout is None:
            raise VideoProcessingError("decoder_open", "The video decoder was not initialized.")
        frame_bytes = self.width * self.height * 3
        decoded = 0
        while True:
            chunk = self._ffmpeg_process.stdout.read(frame_bytes)
            if not chunk:
                break
            if len(chunk) != frame_bytes:
                raise VideoProcessingError(
                    "decode_frame",
                    "A damaged or incomplete frame stopped video decoding.",
                    f"FFmpeg returned a partial frame ({len(chunk)} of {frame_bytes} bytes).",
                )
            decoded += 1
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )

        return_code = self._ffmpeg_process.wait(timeout=30)
        if self._ffmpeg_stderr is not None:
            self._ffmpeg_stderr.seek(0)
            self._ffmpeg_error = self._ffmpeg_stderr.read().decode(
                "utf-8", errors="replace"
            ).strip()
        if return_code != 0:
            raise VideoProcessingError(
                "decode_frame",
                "The video opened, but its frames could not be decoded completely.",
                self._ffmpeg_error or f"FFmpeg exited with status {return_code}.",
            )
        if decoded == 0:
            raise VideoProcessingError(
                "decode_zero_frames",
                "The video opened, but it contained no decodable frames.",
                self._ffmpeg_error or "FFmpeg returned an empty video stream.",
            )

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._ffmpeg_process is not None:
            if self._ffmpeg_process.poll() is None:
                self._ffmpeg_process.terminate()
                try:
                    self._ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._ffmpeg_process.kill()
                    self._ffmpeg_process.wait()
            if self._ffmpeg_process.stdout is not None:
                self._ffmpeg_process.stdout.close()
            self._ffmpeg_process = None
        if self._ffmpeg_stderr is not None:
            self._ffmpeg_stderr.close()
            self._ffmpeg_stderr = None

    def __exit__(self, *_):
        self.close()


def _source_path(source, temp_dir: Path) -> Path:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file() or path.stat().st_size == 0:
            raise VideoProcessingError(
                "input_file", "The selected video file is missing or empty.", str(path)
            )
        return path
    try:
        payload = source if isinstance(source, (bytes, bytearray, memoryview)) else source.getvalue()
    except (AttributeError, OSError) as exc:
        raise VideoProcessingError(
            "input_file", "The uploaded video could not be read.", str(exc)
        ) from exc
    payload = bytes(payload)
    if not payload:
        raise VideoProcessingError("input_file", "The uploaded video is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise VideoProcessingError(
            "input_file", "The video exceeds the 500 MB analysis limit."
        )
    name = getattr(source, "name", "")
    path = temp_dir / f"uploaded_video{_safe_suffix(name, payload)}"
    try:
        path.write_bytes(payload)
    except OSError as exc:
        raise VideoProcessingError(
            "input_file", "The uploaded video could not be staged for analysis.", str(exc)
        ) from exc
    return path


def inspect_video_first_frame(uploaded_file):
    """Return a numbered, all-candidate preview for explicit target selection."""
    with tempfile.TemporaryDirectory(prefix="liftsafe_preview_") as temp_name:
        source = _source_path(uploaded_file, Path(temp_name))
        try:
            with SequentialVideoReader(source) as reader:
                first_frame = next(iter(reader), None)
        except VideoProcessingError:
            raise
        if first_frame is None:
            raise VideoProcessingError(
                "decode_zero_frames", "No video frame was available for target selection."
            )

        pose = None
        try:
            pose = create_pose_estimator(static_image_mode=True)
            candidates = detect_pose_candidates(pose, first_frame)
        except Exception as exc:
            raise VideoProcessingError(
                "pose_initialization",
                "People could not be detected in the preview frame on this server.",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if pose is not None:
                pose.close()

        ok, encoded = cv2.imencode(".jpg", first_frame)
        if not ok:
            raise VideoProcessingError(
                "preview_encode", "The first video frame could not be displayed."
            )
        regions = [
            {
                "candidate_number": item["candidate_number"],
                "bbox": list(item["bbox"]),
                "center": list(item["center"]),
                "visibility": item["visibility"],
            }
            for item in candidates
        ]
        return encoded.tobytes(), regions


def render_target_selection_preview(frame_jpeg, candidates, selected_candidate=None):
    """Draw candidate boxes and highlight the user's chosen person."""
    image = cv2.imdecode(np.frombuffer(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return frame_jpeg
    draw_candidate_boxes(image, candidates, selected_candidate)
    ok, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if ok else frame_jpeg


def process_video(
    uploaded_file, progress_bar, status, threshold=45, target_region="auto",
    manual_target_center=None,
):
    """Decode a video sequentially, analyze frames, and return an H.264 MP4."""
    with tempfile.TemporaryDirectory(prefix="liftsafe_") as temp_name:
        temp_dir = Path(temp_name)
        input_path = _source_path(uploaded_file, temp_dir)
        output_path = temp_dir / "annotated.mp4"
        compatible_path = temp_dir / "annotated_video_h264.mp4"

        rows = []
        frame_index = 0
        target_tracker = PoseTargetTracker(
            preferred_region=target_region,
            manual_target_center=manual_target_center,
        )
        writer = None
        pose = None
        started = time.monotonic()
        try:
            try:
                with SequentialVideoReader(input_path) as reader:
                    fps = reader.fps
                    total_frames = reader.total_frames
                    output_size = None
                    for frame in reader:
                        if time.monotonic() - started >= MAX_VIDEO_PROCESSING_SECONDS:
                            raise VideoProcessingError(
                                "processing_timeout",
                                "Video analysis took too long. Try a shorter or lower-resolution clip.",
                                f"Processing exceeded {MAX_VIDEO_PROCESSING_SECONDS} seconds.",
                            )
                        height, width = frame.shape[:2]
                        if writer is None:
                            output_size = (width, height)
                            writer = cv2.VideoWriter(
                                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (width, height),
                            )
                            if not writer.isOpened():
                                raise VideoProcessingError(
                                    "output_writer",
                                    "Video frames decoded, but an annotated output video could not be created.",
                                    "OpenCV could not open the MP4 output writer.",
                                )
                        elif (width, height) != output_size:
                            frame = cv2.resize(frame, output_size)
                            width, height = output_size

                        if pose is None:
                            try:
                                pose = create_pose_estimator()
                            except Exception as exc:
                                raise VideoProcessingError(
                                    "pose_initialization",
                                    "Video frames decoded, but pose detection could not start on this server.",
                                    f"{type(exc).__name__}: {exc}",
                                ) from exc
                        try:
                            pose_result = analyze_pose_frame(pose, frame, target_tracker)
                        except Exception as exc:
                            logger.exception("Pose inference failed on decoded video frame %s", frame_index)
                            raise VideoProcessingError(
                                "pose_inference",
                                "Video frames decoded, but pose estimation failed while analyzing them.",
                                f"Frame {frame_index}: {type(exc).__name__}: {exc}",
                            ) from exc

                        measurements = pose_result.measurements
                        trunk = measurements.get("Trunk inclination (°)")
                        review = trunk is not None and trunk >= threshold
                        row = {
                            "frame": frame_index,
                            "time_seconds": frame_index / fps,
                            "pose_detected": pose_result.pose_detected,
                            "target_state": pose_result.state,
                            "target_id": pose_result.target_id,
                            "target_score": pose_result.target_score,
                            "people_detected": pose_result.person_count,
                            "candidate_number": pose_result.candidate_number,
                            "selection_reason": pose_result.selection_reason,
                            "candidate_diagnostics": json.dumps([
                                {
                                    **candidate,
                                    "bbox": list(candidate["bbox"]),
                                    "center": list(candidate["center"]),
                                }
                                for candidate in pose_result.candidates
                            ]),
                            "pose_crop_used": False,
                            "overlay_matches_selected_pose": bool(
                                pose_result.landmarks is not None
                            ),
                            "landmark_confidence": (
                                pose_result.visibility
                                if pose_result.visibility is not None else np.nan
                            ),
                            "posture_review_flag": bool(review),
                            "trunk_inclination_2d_deg": measurements.get(
                                "Trunk inclination (°)", np.nan
                            ),
                            "left_hip_angle_2d_deg": measurements.get(
                                "Left hip angle (°)", np.nan
                            ),
                            "right_hip_angle_2d_deg": measurements.get(
                                "Right hip angle (°)", np.nan
                            ),
                            "left_knee_angle_2d_deg": measurements.get(
                                "Left knee angle (°)", np.nan
                            ),
                            "right_knee_angle_2d_deg": measurements.get(
                                "Right knee angle (°)", np.nan
                            ),
                        }
                        if pose_result.candidates:
                            draw_candidate_boxes(
                                frame, pose_result.candidates,
                                pose_result.candidate_number,
                            )
                        if pose_result.landmarks is not None:
                            draw_pose_landmarks(frame, pose_result.landmarks.landmark)
                        logger.debug(
                            "video frame=%s candidates=%s selected_track=%s selected_candidate=%s reason=%s crop_used=false overlay_same_pose=%s",
                            frame_index, row["candidate_diagnostics"],
                            pose_result.target_id, pose_result.candidate_number,
                            pose_result.selection_reason,
                            row["overlay_matches_selected_pose"],
                        )
                        if review:
                            cv2.rectangle(frame, (0, 0), (width, 72), (0, 140, 255), -1)
                            cv2.putText(
                                frame, "POSTURE REVIEW", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2, cv2.LINE_AA,
                            )
                            cv2.putText(
                                frame,
                                f"Trunk inclination: {trunk:.1f} deg (threshold {threshold} deg)",
                                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA,
                            )
                        rows.append(row)
                        writer.write(frame)
                        frame_index += 1
                        if total_frames > 0:
                            progress_bar.progress(min(frame_index / total_frames, 0.99))
                        status.caption(f"Decoded and analyzed {frame_index} frames")
            except VideoProcessingError:
                raise
            except Exception as exc:
                logger.exception("Unexpected video decoding failure")
                raise VideoProcessingError(
                    "video_processing",
                    "An unexpected error interrupted video processing.",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            finally:
                if writer is not None:
                    writer.release()

            if frame_index == 0:
                raise VideoProcessingError(
                    "decode_zero_frames",
                    "The file was opened, but no video frames could be decoded.",
                    "The decoder reached the end of the stream before producing a frame.",
                )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise VideoProcessingError(
                    "output_writer", "Frames were analyzed, but the annotated video could not be created."
                )

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise VideoProcessingError(
                    "output_encode",
                    "Frames were analyzed, but FFmpeg is unavailable to create a browser-compatible download.",
                    "Install FFmpeg or add it to the deployment packages.",
                )
            try:
                result = subprocess.run(
                    [
                        ffmpeg, "-y", "-v", "error", "-i", str(output_path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(compatible_path),
                    ],
                    capture_output=True, text=True, check=True, timeout=300,
                )
            except subprocess.TimeoutExpired as exc:
                raise VideoProcessingError(
                    "output_encode", "Video analysis finished, but output encoding took too long.", str(exc)
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise VideoProcessingError(
                    "output_encode",
                    "Frames were analyzed, but the annotated video could not be encoded.",
                    (exc.stderr or "FFmpeg exited with an error.")[-1500:],
                ) from exc
            if not compatible_path.exists() or compatible_path.stat().st_size == 0:
                raise VideoProcessingError(
                    "output_encode", "Frames were analyzed, but the browser-ready video is empty."
                )
            progress_bar.progress(1.0)
            pose_count = int(pd.DataFrame(rows)["pose_detected"].sum())
            if pose_count == 0:
                status.caption(
                    f"Processed {frame_index} frames; no usable person landmarks were detected."
                )
            elif int(pd.DataFrame(rows)["people_detected"].max()) > 1:
                status.caption(
                    f"Tracked a likely lifting person across {frame_index} frames; "
                    "ambiguous and lost-target frames have no measurements."
                )
            return compatible_path.read_bytes(), pd.DataFrame(rows), fps
        finally:
            # The pose object is created before the decoder; ensure it is closed
            # even if decoder setup fails before the nested processing block.
            if pose is not None:
                try:
                    pose.close()
                except Exception:
                    logger.exception("Failed to close MediaPipe pose estimator")
