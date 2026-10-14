
import json
import pandas as pd
from pathlib import Path

POSE_PATH = Path("data/processed/all_pose_angles.csv")
ANNOTATION_PATH = Path("data/raw/annotations.json")
OUTPUT_PATH = Path("data/processed/video_feature_dataset.csv")

# Load pose measurements
pose = pd.read_csv(POSE_PATH)

# Load annotations
with ANNOTATION_PATH.open("r", encoding="utf-8") as f:
    annotations = json.load(f)

ann = pd.json_normalize(annotations)

# Calculate per-video posture statistics
pose_features = pose.groupby("video").agg(
    usable_frames=("frame", "count"),
    mean_trunk_inclination=("trunk_inclination_deg", "mean"),
    max_trunk_inclination=("trunk_inclination_deg", "max"),
    std_trunk_inclination=("trunk_inclination_deg", "std"),
    mean_hip_angle=("hip_angle_deg", "mean"),
    min_hip_angle=("hip_angle_deg", "min"),
    mean_knee_angle=("knee_angle_deg", "mean"),
    min_knee_angle=("knee_angle_deg", "min"),
    mean_landmark_visibility=("min_visibility", "mean"),
).reset_index()

# Merge annotation information
dataset = pose_features.merge(
    ann,
    on="video",
    how="left",
    validate="one_to_one"
)

# Calculate duration of the annotated lifting action
dataset["action_duration_seconds"] = (
    dataset["action_end.video_second"]
    - dataset["action_start.video_second"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_PATH, index=False)

print("Videos with pose features:", len(pose_features))
print("Videos matched with annotations:",
      dataset["subjective_judgment"].notna().sum())
print("Videos with intermediate judgment:",
      (dataset["subjective_judgment"] == "intermediate").sum())
print("\nFeature columns:")
print(pose_features.columns.tolist())
print("\nSample rows:")
print(dataset[[
    "video",
    "usable_frames",
    "mean_trunk_inclination",
    "mean_hip_angle",
    "mean_knee_angle",
    "subjective_judgment"
]].head().round(2).to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
