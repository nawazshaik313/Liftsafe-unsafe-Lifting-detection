import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = "data/external/inail_dataset1/Dataset1/DATA"
OUTPUT_FILE = "data/processed/inail_trunk_features.csv"


def calculate_trunk_inclination(markers):
    """Calculate trunk angle relative to the global vertical Y-axis."""
    c7 = np.asarray(markers["c7"], dtype=float)
    sacrum = np.asarray(markers["Sacrum"], dtype=float)

    trunk = c7 - sacrum
    lengths = np.linalg.norm(trunk, axis=1)

    angles = np.full(len(lengths), np.nan)
    valid = np.isfinite(trunk).all(axis=1) & (lengths > 1e-8)

    cos_angle = trunk[valid, 1] / lengths[valid]
    angles[valid] = np.degrees(
        np.arccos(np.clip(cos_angle, -1.0, 1.0))
    )

    return angles


def main():
    files = sorted(
        glob.glob(
            os.path.join(
                DATA_DIR, "*", "Exported", "mat_format",
                "*_all_trials.mat"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No INAIL MAT files found under {DATA_DIR}"
        )

    rows = []

    for file_path in files:
        mat = sio.loadmat(file_path, simplify_cells=True)
        variable_name = os.path.basename(file_path).replace(
            "_all_trials.mat", "_all_trials"
        )
        subject_data = mat[variable_name]

        trials = subject_data["Kinematic_data"]
        if isinstance(trials, dict):
            trials = [trials]

        for trial in trials:
            task = str(trial["task"])
            subject_id = task.split("_")[0]

            # Example: S01_Lift_LI1_1.TDF -> LI1
            li_condition = np.nan
            for part in task.replace(".TDF", "").split("_"):
                if part.startswith("LI") and part[2:].isdigit():
                    li_condition = int(part[2:])
                    break

            markers = trial["data_3D"]
            angles = calculate_trunk_inclination(markers)
            valid_angles = angles[np.isfinite(angles)]

            events = trial.get("events", {})
            if not isinstance(events, dict):
                events = {}

            rows.append({
                "subject_id": subject_id,
                "trial": task,
                "li_condition": li_condition,
                "sampling_rate_hz": float(trial["fs_3D"]),
                "frame_count": len(angles),
                "duration_seconds": len(angles) / float(trial["fs_3D"]),
                "start_frame": events.get("start", np.nan),
                "stop_frame": events.get("stop", np.nan),
                "stop_lowering_frame": events.get(
                    "stop_lowering", np.nan
                ),
                "valid_angle_frames": len(valid_angles),
                "trunk_mean_deg": (
                    float(valid_angles.mean()) if len(valid_angles) else np.nan
                ),
                "trunk_median_deg": (
                    float(np.median(valid_angles))
                    if len(valid_angles) else np.nan
                ),
                "trunk_min_deg": (
                    float(valid_angles.min()) if len(valid_angles) else np.nan
                ),
                "trunk_max_deg": (
                    float(valid_angles.max()) if len(valid_angles) else np.nan
                ),
                "trunk_std_deg": (
                    float(valid_angles.std()) if len(valid_angles) else np.nan
                ),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("INAIL feature extraction complete")
    print("MAT files:", len(files))
    print("Trials extracted:", len(df))
    print("Output:", OUTPUT_FILE)
    print("\nTrials per LI condition:")
    print(df["li_condition"].value_counts(dropna=False).sort_index())
    print("\nTrunk inclination summary:")
    print(df[
        ["trunk_mean_deg", "trunk_median_deg",
         "trunk_min_deg", "trunk_max_deg"]
    ].describe().round(2).to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
