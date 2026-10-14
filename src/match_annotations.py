
import json
import pandas as pd
from pathlib import Path

annotation_path = Path("data/raw/annotations.json")
features_path = Path("data/processed/all_pose_angles.csv")
output_path = Path("data/processed/merged_features.csv")

with annotation_path.open("r", encoding="utf-8") as file:
    annotations = json.load(file)

features = pd.read_csv(features_path)

annotation_df = pd.json_normalize(annotations)

merged = features.merge(
    annotation_df,
    on="video",
    how="left",
    validate="many_to_one"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_path, index=False)

print("Videos in features:", features["video"].nunique())
print("Annotation records:", len(annotation_df))
print("Feature rows:", len(features))
print("Merged rows:", len(merged))
print("\nJudgment counts:")
print(merged.groupby("video")["subjective_judgment"].first()
      .value_counts(dropna=False))

print("\nAvailable annotation columns:")
print(annotation_df.columns.tolist())

print("\nMerged dataset saved to:", output_path)
