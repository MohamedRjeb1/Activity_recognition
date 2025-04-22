import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('push_up_counting/data.csv')

# Affichage des statistiques par label pour les angles du coude et hanche-genou
for label in df['label'].unique():
    right_elbow = df[df['label'] == label]['right_elbow_angle']
    hip_knee = df[df['label'] == label]['right_hip_knee_angle']

    print(f"\n{label.upper()} ➤ Right elbow angle:")
    print(f"  ▸ min = {right_elbow.min():.2f}, max = {right_elbow.max():.2f}, mean = {right_elbow.mean():.2f}")

    print(f"{label.upper()} ➤ Right hip-knee angle:")
    print(f"  ▸ min = {hip_knee.min():.2f}, max = {hip_knee.max():.2f}, mean = {hip_knee.mean():.2f}")

# Visualisation avec seaborn
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='label', hue='label', y='right_elbow_angle', palette="Set2")
plt.title("Distribution de l'angle du coude droit par phase")
plt.ylabel("Angle (degrés)")
plt.xlabel("Phase")

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='label', hue='label', y='right_hip_knee_angle', palette="Set3")
plt.title("Distribution de l'angle hanche-genou droit par phase")
plt.ylabel("Angle (degrés)")
plt.xlabel("Phase")

plt.tight_layout()
plt.show()
