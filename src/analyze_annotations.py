
import json
import pandas as pd
from pathlib import Path

annotation_path = Path("data/raw/annotations.json")

with annotation_path.open("r", encoding="utf-8") as f:
    annotations = json.load(f)

df = pd.json_normalize(annotations)

print("Total annotation records:", len(df))
print("Unique video names:", df["video"].nunique())

# Inspect which annotation fields contain usable values.
columns = [
    "subjective_judgment",
    "action_start.video_second",
    "action_end.video_second",
    "action_start.box_height_cm",
    "action_end.box_height_cm",
    "action_start.box_distance_horizontal_from_mid_ankles_cm",
    "action_end.box_distance_horizontal_from_mid_ankles_cm",
    "action_start.hand_distance_horizontal_from_mid_ankles_cm",
    "action_end.hand_distance_horizontal_from_mid_ankles_cm",
    "action_start.angle_of_asymmetry_degree",
    "action_end.angle_of_asymmetry_degree",
]

for col in columns:
    if col not in df.columns:
        continue

    values = df[col]
    usable = values.notna() & (values.astype(str).str.lower() != "unknown")

    print(f"\n{col}")
    print("  Usable values:", int(usable.sum()))
    print("  Missing/unknown:", int((~usable).sum()))
    print("  Examples:", values[usable].head(5).tolist())

# Show judgment distribution across all annotation records.
print("\nJudgment distribution:")
print(df["subjective_judgment"].value_counts(dropna=False))

# Save the normalized annotation table.
output_path = Path("data/processed/annotation_summary.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nSaved annotation table:", output_path)
