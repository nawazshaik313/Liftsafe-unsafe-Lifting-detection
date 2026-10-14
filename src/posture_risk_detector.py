
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/pose_angles.csv")
OUTPUT_PATH = Path("data/processed/posture_risk_results.csv")

# Experimental thresholds for prototype testing only.
TRUNK_LIMIT = 30.0
HIP_LIMIT = 90.0
KNEE_LIMIT = 90.0


def classify_frame(row):
    warnings = []

    if row["trunk_inclination_deg"] > TRUNK_LIMIT:
        warnings.append("High trunk inclination")

    if row["hip_angle_deg"] < HIP_LIMIT:
        warnings.append("Deep hip flexion")

    if row["knee_angle_deg"] < KNEE_LIMIT:
        warnings.append("Deep knee flexion")

    if warnings:
        return "REVIEW POSTURE", "; ".join(warnings)

    return "NO THRESHOLD TRIGGERED", "None"


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Input not found: {INPUT_PATH}. "
            "Run calculate_angles.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "frame",
        "min_visibility",
        "trunk_inclination_deg",
        "hip_angle_deg",
        "knee_angle_deg",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()

    results = df.apply(classify_frame, axis=1, result_type="expand")
    results.columns = ["status", "warnings"]

    df[["status", "warnings"]] = results

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Experimental posture screening ---")
    print("Frames analyzed:", len(df))
    print("Frames flagged for review:",
          (df["status"] == "REVIEW POSTURE").sum())
    print("Frames without threshold triggers:",
          (df["status"] == "NO THRESHOLD TRIGGERED").sum())
    print("Saved results:", OUTPUT_PATH)

    print("\nWarning counts:")
    print(
        df["warnings"]
        .str.split("; ")
        .explode()
        .loc[lambda values: values != "None"]
        .value_counts()
        .to_string()
        if (df["warnings"] != "None").any()
        else "No warnings"
    )

    print("\nFirst 10 results:")
    print(
        df[["frame", "status", "warnings"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
    