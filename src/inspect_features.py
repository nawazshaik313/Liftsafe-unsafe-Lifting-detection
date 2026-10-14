
import pandas as pd
from pathlib import Path

path = Path("data/processed/video_feature_dataset.csv")
df = pd.read_csv(path)

features = [
    "mean_trunk_inclination",
    "max_trunk_inclination",
    "std_trunk_inclination",
    "mean_hip_angle",
    "min_hip_angle",
    "mean_knee_angle",
    "min_knee_angle",
    "mean_landmark_visibility",
]

print("Dataset shape:", df.shape)
print("Unique videos:", df["video"].nunique())

print("\nMissing values:")
print(df[features].isna().sum())

print("\nFeature statistics:")
print(df[features].describe().round(2).to_string())

print("\nJudgment distribution:")
print(df["subjective_judgment"].value_counts(dropna=False))

print("\nLowest average trunk inclination:")
print(
    df.nsmallest(5, "mean_trunk_inclination")[
        ["video", "mean_trunk_inclination", "max_trunk_inclination"]
    ].round(2).to_string(index=False)
)

print("\nHighest average trunk inclination:")
print(
    df.nlargest(5, "mean_trunk_inclination")[
        ["video", "mean_trunk_inclination", "max_trunk_inclination"]
    ].round(2).to_string(index=False)
)
