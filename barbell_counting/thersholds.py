import json
import pandas as pd

# 1) Charger le CSV de ratios (Dataset 1)
# Colonnes attendues :
#   stage (down/up), shoulder_width, feet_width, knee_width,
#   ratio_feet_shoulder, ratio_knee_feet
df = pd.read_csv("biceps_curl_dataset.csv")

# 2) Ne garder que les deux phases
df = df[df["label"].isin(["haut", "bas"])]

# 3) Pour chaque phase, calculer min et max des ratios
thresholds = {}
for phase in ["haut", "bas"]:
    sub = df[df["label"] == phase]
    thresholds[phase] = {
        "shoulder_wrist_ratio": [
            float(sub["wrist"].min()),
            float(sub["wrist"].max())
        ],
        "shoulder_elbow_ratio": [
            float(sub["elbow"].min()),
            float(sub["elbow"].max())
        ],
        "left_shoulder_elobow_wrist_angle_ratio": [
            float(sub["left_elbow_angle"].min()),
            float(sub["left_elbow_angle"].max())
        ],
        "right_shoulder_elobow_wrist_angle_ratio": [
            float(sub["right_elbow_angle"].min()),
            float(sub["right_elbow_angle"].max())
        ],
        "left_shoulder_elbow_hip_angle_ratio": [
            float(sub["left_elbow_shoulder_hip"].min()),
            float(sub["left_elbow_shoulder_hip"].max())
        ],
        "right_shoulder_elbow_hip_angle_ratio": [
            float(sub["right_elbow_shoulder_hip"].min()),
            float(sub["right_elbow_shoulder_hip"].max())
        ]
    }

# 4) Sauvegarder les seuils dans JSON
with open("thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("✅ Seuils pour down/up enregistrés dans thresholds.json")
print(json.dumps(thresholds, indent=2))
