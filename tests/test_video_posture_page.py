import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


class VideoPosturePageTests(unittest.TestCase):
    def test_cached_analysis_results_render_in_streamlit_page(self):
        video_path = Path("outputs/posture_risk_demo.mp4")
        if not video_path.is_file():
            self.skipTest("Repository demo video is unavailable for the page smoke test")

        app = AppTest.from_file("pages/1_Video_Posture_Analysis.py", default_timeout=30)
        app.session_state["liftsafe_video_upload_key"] = None
        app.session_state["liftsafe_video_bytes"] = video_path.read_bytes()
        app.session_state["liftsafe_video_fps"] = 30.0
        app.session_state["liftsafe_video_csv"] = pd.DataFrame([{
            "frame": 0,
            "time_seconds": 0.0,
            "pose_detected": True,
            "target_state": "measurements_available",
            "target_id": 1,
            "people_detected": 1,
            "candidate_number": 1,
            "selection_reason": "deterministic UI smoke test",
            "candidate_diagnostics": "[]",
            "pose_crop_used": False,
            "overlay_matches_selected_pose": True,
            "landmark_confidence": 0.9,
            "measurement_status": "Review threshold exceeded",
            "posture_review_flag": True,
            "trunk_inclination_2d_deg": 60.0,
            "left_hip_angle_2d_deg": 90.0,
            "right_hip_angle_2d_deg": 90.0,
            "left_knee_angle_2d_deg": 120.0,
            "right_knee_angle_2d_deg": 120.0,
        }])

        app.run()
        self.assertFalse(app.exception)
        metric_values = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metric_values["Frames processed"], "1")
        self.assertEqual(metric_values["Selected target"], "Person 1")
        self.assertTrue(any("Current status: Review threshold exceeded" in item.value for item in app.warning))
        # Streamlit stores results in session state, so the page can render
        # them again on subsequent reruns without recomputing the video.
        self.assertIn("liftsafe_video_csv", app.session_state)


if __name__ == "__main__":
    unittest.main()
