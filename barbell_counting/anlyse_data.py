import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")

# Chargement des données
df = pd.read_csv('biceps_curl_dataset.csv')

# Affich
for label in df['label'].unique():
    left_elbow_shoulder_hip = df[df['label'] == label]['left_elbow_shoulder_hip']
    right_elbow_shoulder_hip = df[df['label'] == label]['right_elbow_shoulder_hip']
    right_elbow_angle = df[df['label'] == label]['right_elbow_angle']
    left_elbow_angle = df[df['label'] == label]['left_elbow_angle']

    print(f"\n{label.upper()} ➤ left_elbow_shoulder_hip:")
    print(f"  ▸ min = {left_elbow_shoulder_hip.min():.2f}, max = {left_elbow_shoulder_hip.max():.2f}, mean = {left_elbow_shoulder_hip.mean():.2f}")

    print(f"\n{label.upper()} ➤ right_elbow_shoulder_hip:")
    print(f"  ▸ min = {right_elbow_shoulder_hip.min():.2f}, max = {right_elbow_shoulder_hip.max():.2f}, mean = {right_elbow_shoulder_hip.mean():.2f}")

    print(f"\n{label.upper()} ➤ right_elbow_angle:")
    print(f"  ▸ min = {right_elbow_angle.min():.2f}, max = {right_elbow_angle.max():.2f}, mean = {right_elbow_angle.mean():.2f}")

    print(f"\n{label.upper()} ➤ left_elbow_angle:")
    print(f"  ▸ min = {left_elbow_angle.min():.2f}, max = {left_elbow_angle.max():.2f}, mean = {left_elbow_angle.mean():.2f}")

## Crée une figure 2×2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1er subplot (ligne 0, colonne 0)
sns.boxplot(
    data=df,
    x='label', y='left_elbow_shoulder_hip',
    hue="label", legend=False,
    ax=axes[0, 0], palette="Set3"
)
axes[0, 0].set_title("left_elbow_shoulder_hip")
axes[0, 0].set_ylabel("Angle (°)")
axes[0, 0].set_xlabel("Phase")

# 2e subplot (ligne 0, colonne 1)
sns.boxplot(
    data=df,
    x='label', y='right_elbow_shoulder_hip',
    hue="label", legend=False,
    ax=axes[0, 1], palette="Set2"
)
axes[0, 1].set_title("right_elbow_shoulder_hip")
axes[0, 1].set_ylabel("Angle (°)")
axes[0, 1].set_xlabel("Phase")

# 3e subplot (ligne 1, colonne 0)
sns.boxplot(
    data=df,
    x='label', y='right_elbow_angle',
    hue="label", legend=False,
    ax=axes[1, 0], palette="Set3"
)
axes[1, 0].set_title("right_elbow_angle")
axes[1, 0].set_ylabel("Angle (°)")
axes[1, 0].set_xlabel("Phase")

# 4e subplot (ligne 1, colonne 1)
sns.boxplot(
    data=df,
    x='label', y='left_elbow_angle',
    hue="label", legend=False,
    ax=axes[1, 1], palette="Set2"
)
axes[1, 1].set_title("left_elbow_angle")
axes[1, 1].set_ylabel("Angle (°)")
axes[1, 1].set_xlabel("Phase")

plt.tight_layout()
plt.show()
