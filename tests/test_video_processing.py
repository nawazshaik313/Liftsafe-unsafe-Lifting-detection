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
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def process(self, _frame):
        return SimpleNamespace(pose_landmarks=None)


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

    def _run(self, payload):
        return video_processing.process_video(
            io.BytesIO(payload), NoOpProgress(), NoOpStatus()
        )

    def test_empty_and_oversized_uploads_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self._run(b"")

        with patch.object(video_processing, "MAX_UPLOAD_BYTES", 3):
            with self.assertRaisesRegex(ValueError, "500 MB"):
                self._run(b"1234")

    def test_corrupt_video_is_rejected_and_temporary_files_are_removed(self):
        with (
            patch.object(video_processing.shutil, "which", return_value="ffmpeg"),
            patch.object(video_processing.tempfile, "TemporaryDirectory", self._track_tempdir),
        ):
            with self.assertRaisesRegex(ValueError, "could not read"):
                self._run(b"not a video")

        self.assertEqual(len(self.tempdirs), 1)
        self.assertFalse(self.tempdirs[0].exists())

    def test_missing_ffmpeg_has_actionable_error(self):
        with patch.object(video_processing.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg is required"):
                self._run(b"non-empty upload")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_processing_timeout_releases_handles_and_temporary_files(self):
        with tempfile.TemporaryDirectory(prefix="liftsafe_test_input_") as source_dir:
            source = Path(source_dir) / "short.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV mp4v encoder is unavailable")
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            payload = source.read_bytes()

        with (
            patch.object(video_processing.mp_pose, "Pose", return_value=FakePose()),
            patch.object(video_processing, "MAX_VIDEO_PROCESSING_SECONDS", -1),
            patch.object(video_processing.tempfile, "TemporaryDirectory", self._track_tempdir),
        ):
            with self.assertRaisesRegex(RuntimeError, "10 minute processing limit"):
                self._run(payload)

        self.assertEqual(len(self.tempdirs), 1)
        self.assertFalse(self.tempdirs[0].exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_short_video_returns_playable_output_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory(prefix="liftsafe_test_input_") as source_dir:
            source = Path(source_dir) / "short.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
            )
            if not writer.isOpened():
                self.skipTest("OpenCV mp4v encoder is unavailable")
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            payload = source.read_bytes()

        with (
            patch.object(video_processing.mp_pose, "Pose", return_value=FakePose()),
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
