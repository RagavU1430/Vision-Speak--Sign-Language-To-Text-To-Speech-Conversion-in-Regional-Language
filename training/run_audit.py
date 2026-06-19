import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import json
import joblib
import numpy as np
import pandas as pd

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv(os.path.join("dataset", "landmarks_v1.csv"))
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values.astype(np.float32)
y = df["label"].values

model = joblib.load("models/mlp_model.pkl")
le = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

y_encoded = le.transform(y)
_, X_test_raw, _, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_test = scaler.transform(X_test_raw)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) * 100
class_names = le.classes_

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(f"\nOverall Accuracy: {accuracy:.2f}%")
print(f"\nPer-Class Accuracy:")
for name in class_names:
    acc = report[name]["recall"] * 100
    bars = "#" * int(acc / 2)
    print(f"  {name}: {acc:5.1f}% |{bars:<50}")

n = len(class_names)
merged = {}
for i in range(n):
    for j in range(i + 1, n):
        ab = cm[i, j]
        ba = cm[j, i]
        total = ab + ba
        if total > 0:
            total_i = cm[i, :].sum() or 1
            total_j = cm[j, :].sum() or 1
            pct = (ab / total_i + ba / total_j) * 50
            merged[f"{class_names[i]}<->{class_names[j]}"] = {
                "pair": f"{class_names[i]}<->{class_names[j]}",
                "total_pct": round(pct, 2),
                "samples": total,
            }

ranked = sorted(merged.values(), key=lambda x: x["total_pct"], reverse=True)
print(f"\n\nRanked Collision Pairs ({len(ranked)} total):")
for i, r in enumerate(ranked):
    print(f"  {i+1:2d}. {r['pair']}: {r['total_pct']:5.2f}% ({r['samples']} samples)")

# Save report
os.makedirs("models/collision_report", exist_ok=True)
with open("models/collision_report/collision_report.json", "w") as f:
    json.dump({
        "accuracy": round(accuracy, 2),
        "total_pairs": len(ranked),
        "top_25": [
            {"rank": i + 1, "pair": r["pair"], "pct": r["total_pct"], "samples": r["samples"]}
            for i, r in enumerate(ranked)
        ],
        "per_class": {n: round(report[n]["recall"] * 100, 2) for n in class_names},
    }, f, indent=2)
print("\nReport saved to models/collision_report/collision_report.json")
