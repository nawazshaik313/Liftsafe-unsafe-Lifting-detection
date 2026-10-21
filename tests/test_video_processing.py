import io
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src import video_processing


class FakePose:
    def close(self):
        pass

    def process(self, _frame):
        return SimpleNamespace(pose_landmarks=None)


class FailingPose(FakePose):
    def process(self, _frame):
        raise RuntimeError("inference broke")


class NoOpProgress:
    def progress(self, _value):
        pass


class NoOpStatus:
    def caption(self, _message):
        pass


class VideoProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tempdirs = []
        self.real_tempdir = tempfile.TemporaryDirectory

    def _track_tempdir(self, *args, **kwargs):
        directory = self.real_tempdir(*args, **kwargs)
        self.tempdirs.append(Path(directory.name))
        return directory

    def _run(self, payload, name="sample.mp4"):
        upload = io.BytesIO(payload)
        upload.name = name
        return video_processing.process_video(upload, NoOpProgress(), NoOpStatus())

    def _short_video(self, frame_count=3):
        with tempfile.TemporaryDirectory(prefix="liftsafe_test_input_") as source_dir:
            source = Path(source_dir) / "short.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV mp4v encoder is unavailable")
            for index in range(frame_count):
                frame = np.full((64, 64, 3), index * 35, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            return source.read_bytes()

    def test_empty_and_oversized_uploads_are_rejected(self):
        with self.assertRaisesRegex(video_processing.VideoProcessingError, "empty"):
            self._run(b"")

        with patch.object(video_processing, "MAX_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(video_processing.VideoProcessingError, "500 MB"):
                self._run(b"1234")

    def test_corrupt_video_reports_decoder_stage_and_cleans_temp_files(self):
        with (
            patch.object(video_processing.tempfile, "TemporaryDirectory", self._track_tempdir),
            self.assertRaises(video_processing.VideoProcessingError) as raised,
        ):
            self._run(b"not a video")

        self.assertEqual(raised.exception.stage, "decoder_open")
        self.assertTrue(raised.exception.details)
        self.assertEqual(len(self.tempdirs), 1)
        self.assertFalse(self.tempdirs[0].exists())

    def test_missing_ffmpeg_does_not_block_decoding_but_reports_output_stage(self):
        payload = self._short_video()
        with (
            patch.object(video_processing, "create_pose_estimator", return_value=FakePose()),
            patch.object(video_processing.shutil, "which", return_value=None),
            self.assertRaises(video_processing.VideoProcessingError) as raised,
        ):
            self._run(payload)
        self.assertEqual(raised.exception.stage, "output_encode")
        self.assertIn("Frames were analyzed", raised.exception.user_message)

    def test_open_failure_is_distinguished_from_zero_decoded_frames(self):
        class ClosedCapture:
            def isOpened(self):
                return False

            def read(self):
                return False, None

            def release(self):
                self.released = True

        with (
            patch.object(video_processing.cv2, "VideoCapture", return_value=ClosedCapture()),
            patch.object(video_processing.shutil, "which", return_value=None),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"invalid")
            with self.assertRaises(video_processing.VideoProcessingError) as raised:
                with video_processing.SequentialVideoReader(source):
                    pass
        self.assertEqual(raised.exception.stage, "decoder_open")

    def test_open_but_empty_decoder_is_reported_separately_and_released(self):
        class EmptyCapture:
            released = False

            def isOpened(self):
                return True

            def read(self):
                return False, None

            def release(self):
                self.released = True

        capture = EmptyCapture()
        with (
            patch.object(video_processing.cv2, "VideoCapture", return_value=capture),
            patch.object(video_processing.shutil, "which", return_value=None),
            tempfile.TemporaryDirectory() as directory,
        ):
            source = Path(directory) / "source.mp4"
            source.write_bytes(b"container bytes")
            with self.assertRaises(video_processing.VideoProcessingError) as raised:
                with video_processing.SequentialVideoReader(source):
                    pass
        self.assertEqual(raised.exception.stage, "decode_zero_frames")
        self.assertTrue(capture.released)

    def test_sequential_decode_does_not_trust_frame_count_metadata(self):
        payload = self._short_video(frame_count=4)
        real_capture = cv2.VideoCapture

        class MisleadingCountCapture:
            def __init__(self, path):
                self.capture = real_capture(path)

            def isOpened(self):
                return self.capture.isOpened()

            def read(self):
                return self.capture.read()

            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 1
                return self.capture.get(prop)

            def release(self):
                self.capture.release()

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(payload)
            with patch.object(
                video_processing.cv2, "VideoCapture", MisleadingCountCapture
            ):
                with video_processing.SequentialVideoReader(source) as reader:
                    frames = list(reader)
                    self.assertEqual(reader.total_frames, 1)
        self.assertEqual(len(frames), 4)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are not installed")
    def test_ffmpeg_fallback_decodes_when_opencv_cannot_open(self):
        payload = self._short_video(frame_count=2)

        class ClosedCapture:
            def isOpened(self):
                return False

            def read(self):
                return False, None

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            source.write_bytes(payload)
            with patch.object(video_processing.cv2, "VideoCapture", return_value=ClosedCapture()):
                with video_processing.SequentialVideoReader(source) as reader:
                    frames = list(reader)
                    self.assertEqual(reader.decoder, "ffmpeg")
        self.assertEqual(len(frames), 2)

    def test_first_frame_target_preview_returns_numbered_boxes_and_manual_highlight(self):
        payload = self._short_video(frame_count=1)
        upload = io.BytesIO(payload)
        upload.name = "preview.mp4"
        candidates = (
            {
                "candidate_number": 1,
                "bbox": (0.1, 0.1, 0.35, 0.9),
                "center": (0.225, 0.5),
                "visibility": 0.8,
                "landmarks": [],
            },
            {
                "candidate_number": 2,
                "bbox": (0.6, 0.1, 0.9, 0.9),
                "center": (0.75, 0.5),
                "visibility": 0.9,
                "landmarks": [],
            },
        )
        with (
            patch.object(video_processing, "create_pose_estimator", return_value=FakePose()),
            patch.object(video_processing, "detect_pose_candidates", return_value=candidates),
        ):
            frame_jpeg, regions = video_processing.inspect_video_first_frame(upload)

        self.assertEqual([item["candidate_number"] for item in regions], [1, 2])
        self.assertEqual(regions[1]["center"], [0.75, 0.5])
        highlighted = video_processing.render_target_selection_preview(frame_jpeg, regions, 2)
        image = cv2.imdecode(np.frombuffer(highlighted, dtype=np.uint8), cv2.IMREAD_COLOR)
        selected_green = (
            (image[:, :, 1] > 130)
            & (image[:, :, 1] > image[:, :, 2] * 1.5)
            & (image[:, :, 1] > image[:, :, 0] * 1.5)
        )
        self.assertTrue(np.any(selected_green))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_pose_failure_is_not_mislabeled_as_decode_failure(self):
        payload = self._short_video(frame_count=1)
        with (
            patch.object(video_processing, "create_pose_estimator", return_value=FailingPose()),
            self.assertRaises(video_processing.VideoProcessingError) as raised,
        ):
            self._run(payload)
        self.assertEqual(raised.exception.stage, "pose_inference")
        self.assertIn("Frame 0", raised.exception.details)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_short_video_processes_frames_and_cleans_temporary_files(self):
        payload = self._short_video(frame_count=3)
        with (
            patch.object(video_processing, "create_pose_estimator", return_value=FakePose()),
            patch.object(video_processing.tempfile, "TemporaryDirectory", self._track_tempdir),
        ):
            output, measurements, fps = self._run(payload)

        self.assertTrue(output)
        self.assertIn(b"ftyp", output[:32])
        self.assertEqual(len(measurements), 3)
        self.assertFalse(measurements["pose_detected"].any())
        self.assertEqual(fps, 10.0)
        self.assertEqual(len(self.tempdirs), 1)
        self.assertFalse(self.tempdirs[0].exists())

        with tempfile.TemporaryDirectory(prefix="liftsafe_test_decode_") as check_dir:
            result_path = Path(check_dir) / "result.mp4"
            result_path.write_bytes(output)
            cap = cv2.VideoCapture(str(result_path))
            try:
                decoded = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    decoded += 1
            finally:
                cap.release()
        self.assertEqual(decoded, 3)


if __name__ == "__main__":
    unittest.main()
