
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/merged_features.csv")
OUTPUT_DIR = Path("data/processed")
SEQUENCE_LENGTH = 64

FEATURE_COLUMNS = [
    "trunk_inclination_deg",
    "hip_angle_deg",
    "knee_angle_deg",
    "min_visibility",
]


def resample_sequence(values, length):
    """Resample one video's ordered frame features to a fixed length."""
    if len(values) == 0:
        return None

    old_positions = np.linspace(0, 1, num=len(values))
    new_positions = np.linspace(0, 1, num=length)

    result = np.column_stack([
        np.interp(
            new_positions,
            old_positions,
            values[:, column]
        )
        for column in range(values.shape[1])
    ])

    return result.astype(np.float32)


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing input: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = ["video", "frame", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    sequences = []
    video_names = []
    skipped = []

    for video, group in df.groupby("video"):
        group = group.sort_values("frame")
        values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            skipped.append(video)
            continue

        sequence = resample_sequence(values, SEQUENCE_LENGTH)
        if sequence is None:
            skipped.append(video)
            continue

        sequences.append(sequence)
        video_names.append(video)

    if not sequences:
        raise SystemExit("No valid sequences were produced.")

    X = np.stack(sequences)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "cnn_sequences.npy", X)
    pd.DataFrame({"video": video_names}).to_csv(
        OUTPUT_DIR / "cnn_sequence_videos.csv",
        index=False
    )

    print("Sequence array shape:", X.shape)
    print("Videos included:", len(video_names))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Features per timestep:", len(FEATURE_COLUMNS))
    print("Feature order:", FEATURE_COLUMNS)
    print("Skipped videos:", skipped)
    print("Saved:", OUTPUT_DIR / "cnn_sequences.npy")
    print("Saved:", OUTPUT_DIR / "cnn_sequence_videos.csv")
    print("\nNo labels were assigned.")


if __name__ == "__main__":
    main()
