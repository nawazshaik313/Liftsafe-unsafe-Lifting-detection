
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/processed/merged_features.csv")
OUTPUT_PATH = Path("data/processed/temporal_features.csv")

TRUNK_LIMIT = 30.0
HIP_LIMIT = 90.0
KNEE_LIMIT = 90.0


def calculate_features(group):
    group = group.sort_values("frame").copy()

    trunk = group["trunk_inclination_deg"].dropna()
    hip = group["hip_angle_deg"].dropna()
    knee = group["knee_angle_deg"].dropna()

    if len(group) < 2 or trunk.empty or hip.empty or knee.empty:
        return None

    # Threshold percentages are experimental descriptive features.
    valid = group.dropna(
        subset=[
            "trunk_inclination_deg",
            "hip_angle_deg",
            "knee_angle_deg",
        ]
    )

    frame_gap = (
        valid["frame"].max() - valid["frame"].min()
    )

    trunk_values = valid["trunk_inclination_deg"].to_numpy()
    hip_values = valid["hip_angle_deg"].to_numpy()
    knee_values = valid["knee_angle_deg"].to_numpy()

    return {
        "video": group["video"].iloc[0],
        "frame_count": len(group),
        "mean_trunk_deg": trunk.mean(),
        "max_trunk_deg": trunk.max(),
        "std_trunk_deg": trunk.std(),
        "min_hip_deg": hip.min(),
        "mean_hip_deg": hip.mean(),
        "min_knee_deg": knee.min(),
        "mean_knee_deg": knee.mean(),
        "trunk_change_deg": (
            trunk_values[-1] - trunk_values[0]
        ),
        "trunk_over_threshold_pct": (
            (valid["trunk_inclination_deg"] > TRUNK_LIMIT).mean()
            * 100
        ),
        "hip_under_threshold_pct": (
            (valid["hip_angle_deg"] < HIP_LIMIT).mean()
            * 100
        ),
        "knee_under_threshold_pct": (
            (valid["knee_angle_deg"] < KNEE_LIMIT).mean()
            * 100
        ),
        "first_frame": int(valid["frame"].min()),
        "last_frame": int(valid["frame"].max()),
        "frame_span": int(frame_gap),
        "subjective_judgment": group[
            "subjective_judgment"
        ].iloc[0],
    }


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing input: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = [
        "video",
        "frame",
        "trunk_inclination_deg",
        "hip_angle_deg",
        "knee_angle_deg",
        "subjective_judgment",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    records = []

    for _, group in df.groupby("video"):
        result = calculate_features(group)
        if result is not None:
            records.append(result)

    output = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Videos processed:", len(output))
    print("Feature columns:", len(output.columns))
    print("Saved:", OUTPUT_PATH)

    feature_columns = [
        "mean_trunk_deg",
        "max_trunk_deg",
        "std_trunk_deg",
        "min_hip_deg",
        "mean_hip_deg",
        "min_knee_deg",
        "trunk_over_threshold_pct",
        "hip_under_threshold_pct",
        "knee_under_threshold_pct",
    ]

    print("\nFeature summary:")
    print(output[feature_columns].describe().round(2).to_string())

    print("\nJudgment distribution:")
    print(output["subjective_judgment"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
