# analyze_thresholds.py

import json
import pandas as pd

# 1) Charger le CSV de ratios (Dataset 1)
# Colonnes attendues :
#   stage (down/up), shoulder_width, feet_width, knee_width,
#   ratio_feet_shoulder, ratio_knee_feet
df = pd.read_csv("./dataset/analyze_pose.csv")

# 2) Ne garder que les deux phases
df = df[df["stage"].isin(["down", "up"])]

# 3) Pour chaque phase, calculer min et max des ratios
thresholds = {}
for phase in ["down", "up"]:
    sub = df[df["stage"] == phase]
    thresholds[phase] = {
        "foot_shoulder_ratio": [
            float(sub["ratio_feet_shoulder"].min()),
            float(sub["ratio_feet_shoulder"].max())
        ],
        "knee_feet_ratio": [
            float(sub["ratio_knee_feet"].min()),
            float(sub["ratio_knee_feet"].max())
        ]
    }

# 4) Sauvegarder les seuils dans JSON
with open("thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("✅ Seuils pour down/up enregistrés dans thresholds.json")
print(json.dumps(thresholds, indent=2))
