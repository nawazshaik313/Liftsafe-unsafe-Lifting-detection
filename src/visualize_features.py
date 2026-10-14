
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("data/processed/video_feature_dataset.csv")
OUTPUT = Path("outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Chart 1: Average and maximum trunk inclination
df = df.sort_values("mean_trunk_inclination")

plt.figure(figsize=(12, 7))
plt.barh(
    df["video"],
    df["mean_trunk_inclination"],
    label="Mean trunk inclination"
)
plt.xlabel("Trunk inclination (degrees)")
plt.ylabel("Video")
plt.title("Average Trunk Inclination by Video")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "trunk_inclination.png", dpi=150)
plt.close()

# Chart 2: Relationship between trunk inclination and minimum hip angle
plt.figure(figsize=(9, 6))
plt.scatter(
    df["mean_trunk_inclination"],
    df["min_hip_angle"]
)
plt.xlabel("Mean trunk inclination (degrees)")
plt.ylabel("Minimum hip angle (degrees)")
plt.title("Trunk Inclination vs Minimum Hip Angle")
plt.tight_layout()
plt.savefig(OUTPUT / "posture_feature_scatter.png", dpi=150)
plt.close()

print("Charts saved:")
print(OUTPUT / "trunk_inclination.png")
print(OUTPUT / "posture_feature_scatter.png")
