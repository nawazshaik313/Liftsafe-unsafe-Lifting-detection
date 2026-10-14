
import json
from collections import Counter
from pathlib import Path

path = Path("data/raw/annotations.json")

with path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print("Total annotation records:", len(records))

print("\nSubjective judgment counts:")
print(Counter(str(r.get("subjective_judgment", "missing")) for r in records))

print("\nOne-hand lifting counts:")
print(Counter(str(r.get("one_hand_lifting", "missing")) for r in records))

print("\nMultiple-subject lifting counts:")
print(Counter(str(r.get("multiple_subjects_lifting", "missing")) for r in records))

print("\nVideos per environment archive:")
for prefix in ["gd_", "ap_", "em_"]:
    matching = [r.get("video", "") for r in records
                if r.get("video", "").startswith(prefix)]
    print(f"{prefix}: {len(matching)} records")

print("\nFirst 10 video records:")
for r in records[:10]:
    print(
        r.get("video"),
        "| judgment:", r.get("subjective_judgment"),
        "| one hand:", r.get("one_hand_lifting"),
        "| multiple subjects:", r.get("multiple_subjects_lifting")
    )
