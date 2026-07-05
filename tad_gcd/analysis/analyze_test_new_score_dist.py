import json
import numpy as np
from collections import Counter

PRED_JSON = "/data/lcl/checkpoints/opentad_tadgcd/test_old_detector_on_test_new_i3d/gpu1_id0/result_detection.json"

with open(PRED_JSON, "r") as f:
    data = json.load(f)

results = data["results"]

scores = []
labels = []
lengths = []
num_per_video = []

for vid, preds in results.items():
    num_per_video.append(len(preds))
    for p in preds:
        s, e = p["segment"]
        scores.append(float(p.get("score", 0.0)))
        labels.append(p.get("label", "unknown"))
        lengths.append(float(e) - float(s))

scores = np.array(scores)
lengths = np.array(lengths)

print("num videos:", len(results))
print("total preds:", len(scores))
print("preds per video: mean", np.mean(num_per_video), "min", np.min(num_per_video), "max", np.max(num_per_video))

print("\nscore stats:")
for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"p{q:>3}: {np.percentile(scores, q):.6f}")

print("\nlength stats:")
for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"p{q:>3}: {np.percentile(lengths, q):.3f}")

print("\nlabel counts top 20:")
for k, v in Counter(labels).most_common(20):
    print(k, v)