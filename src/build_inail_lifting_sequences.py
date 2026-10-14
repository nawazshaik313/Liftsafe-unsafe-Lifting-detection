import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = "data/external/inail_dataset1/Dataset1/DATA"
OUTPUT_DIR = "data/processed/inail_lifting_sequences"
TARGET_FRAMES = 64
FS = 340.0


def angle_between(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(len(denom), np.nan)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    valid &= denom > 1e-8
    cosine = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    out[valid] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return out


def extract_features(markers):
    c7 = np.asarray(markers["c7"], dtype=float)
    sacrum = np.asarray(markers["Sacrum"], dtype=float)
    trunk = c7 - sacrum

    lengths = np.linalg.norm(trunk, axis=1)
    trunk_angle = np.full(len(lengths), np.nan)
    valid = np.isfinite(trunk).all(axis=1) & (lengths > 1e-8)
    trunk_angle[valid] = np.degrees(
        np.arccos(np.clip(trunk[valid, 1] / lengths[valid], -1, 1))
    )

    hip = (np.asarray(markers["RAsis"]) +
           np.asarray(markers["LAsis"])) / 2
    knee = (np.asarray(markers["RKnee1"]) +
            np.asarray(markers["LKnee1"])) / 2
    shoulder = (np.asarray(markers["RShould"]) +
                np.asarray(markers["LShould"])) / 2
    ankle = (np.asarray(markers["RMall"]) +
             np.asarray(markers["LMall"])) / 2

    hip_included = angle_between(shoulder - hip, knee - hip)
    knee_included = angle_between(hip - knee, ankle - knee)

    # Supplementary geometric angles; not validated anatomical joint angles.
    hip_bending = 180.0 - hip_included
    knee_flexion = 180.0 - knee_included

    angular_change = np.full(len(trunk_angle), np.nan)
    angular_change[1:] = np.diff(trunk_angle) * FS

    return np.column_stack([
        trunk_angle, hip_bending, knee_flexion, angular_change
    ])


def resample_sequence(features):
    old_t = np.linspace(0, 1, len(features))
    new_t = np.linspace(0, 1, TARGET_FRAMES)
    out = np.empty((TARGET_FRAMES, features.shape[1]))

    for j in range(features.shape[1]):
        values = features[:, j]
        valid = np.isfinite(values)
        if valid.sum() < 2:
            out[:, j] = 0.0
        else:
            out[:, j] = np.interp(new_t, old_t[valid], values[valid])
    return out


def main():
    files = sorted(glob.glob(os.path.join(
        DATA_DIR, "*", "Exported", "mat_format", "*_all_trials.mat"
    )))
    if not files:
        raise FileNotFoundError(f"No MAT files found in {DATA_DIR}")

    sequences, rows = [], []

    for path in files:
        mat = sio.loadmat(path, simplify_cells=True)
        key = os.path.basename(path).replace("_all_trials.mat", "_all_trials")
        trials = mat[key]["Kinematic_data"]
        if isinstance(trials, dict):
            trials = [trials]

        for trial in trials:
            task = str(trial["task"])
            events = trial["events"]
            start = int(events["start"])
            stop = int(events["stop"])
            markers = trial["data_3D"]
            frame_count = len(np.asarray(markers["c7"]))

            if start < 0 or stop < start or stop >= frame_count:
                print("Skipping invalid event range:", task)
                continue

            features = extract_features(markers)[start:stop + 1]
            sequence = resample_sequence(features)

            li = np.nan
            for part in task.replace(".TDF", "").split("_"):
                if part.startswith("LI") and part[2:].isdigit():
                    li = int(part[2:])
                    break

            sequences.append(sequence)
            rows.append({
                "subject_id": task.split("_")[0],
                "trial": task,
                "li_condition": li,
                "start_frame": start,
                "stop_frame": stop,
                "lifting_frames": stop - start + 1,
                "lifting_duration_seconds":
                    (stop - start) / float(trial["fs_3D"]),
                "sampling_rate_hz": float(trial["fs_3D"]),
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    array = np.asarray(sequences, dtype=np.float32)
    metadata = pd.DataFrame(rows)
    np.save(os.path.join(OUTPUT_DIR, "sequences.npy"), array)
    metadata.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print("Lifting-phase extraction complete")
    print("Sequences:", len(metadata))
    print("Array shape:", array.shape)
    print("NaN values:", int(np.isnan(array).sum()))
    print("Infinite values:", int(np.isinf(array).sum()))
    print("Features: trunk inclination, supplementary hip angle, "
          "knee flexion angle, trunk angular change (deg/s)")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
