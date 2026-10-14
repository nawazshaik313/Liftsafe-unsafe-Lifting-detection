
import cv2
from pathlib import Path

video_dir = Path("data/raw/video_env_1_anonymized")
videos = sorted(video_dir.glob("*.mp4"))

print("Videos found:", len(videos))

if not videos:
    raise SystemExit("No MP4 files found. Check the extraction path.")

video_path = videos[0]
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise SystemExit(f"Could not open {video_path.name}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video:", video_path.name)
print("Resolution:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("FPS:", fps)
print("Frames:", frames)
print("Duration:", round(frames / fps, 2) if fps > 0 else "unknown")

cap.release()

