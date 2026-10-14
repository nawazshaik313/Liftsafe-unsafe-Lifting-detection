
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("data/processed/pose_angles.csv")
output_path = Path("outputs/angle_plot.png")

df = pd.read_csv(csv_path)

if df.empty:
    raise SystemExit("CSV is empty. Run calculate_angles.py first.")

columns = [
    "trunk_inclination_deg",
    "hip_angle_deg",
    "knee_angle_deg"
]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for ax, column in zip(axes, columns):
    ax.plot(df["frame"], df[column])
    ax.set_ylabel(column.replace("_", " ").title())
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Video frame")
fig.suptitle("Lifting Posture Angles Over Time")
fig.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
plt.close(fig)

print("Plot saved to:", output_path)
print("Frames plotted:", len(df))
