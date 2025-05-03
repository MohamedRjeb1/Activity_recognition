import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Chargement des données
df = pd.read_csv('shoulder_press_counting/analyse.csv')
# Remplacer les NaN par 0
#df.fillna(0, inplace=True)


# Affichage des statistiques par label pour les angles du coude et hanche-genou
def afficher_statistique():
 for label in df['label'].unique():
    right_elbow = df[df['label'] == label]['right_shoulder']
    hip_knee = df[df['label'] == label]['left_shoulder']

    print(f"\n{label.upper()} ➤ Right elbow angle:")
    print(f"  ▸ min = {right_elbow.min():.2f}, max = {right_elbow.max():.2f}, mean = {right_elbow.mean():.2f}")

    print(f"{label.upper()} ➤ Right hip-knee angle:")
    print(f"  ▸ min = {hip_knee.min():.2f}, max = {hip_knee.max():.2f}, mean = {hip_knee.mean():.2f}")

# Visualisation avec seaborn
 plt.figure(figsize=(14, 6))

 plt.subplot(1, 2, 1)
 sns.boxplot(data=df, x='label', hue='label', y='right_shoulder', palette="Set2")
 plt.title("Distribution de l'angle du coude droit par phase")
 plt.ylabel("Angle (degrés)")
 plt.xlabel("Phase")

 plt.subplot(1, 2, 2)
 sns.boxplot(data=df, x='label', hue='label', y='left_shoulder', palette="Set3")
 plt.title("Distribution de l'angle hanche-genou droit par phase")
 plt.ylabel("Angle (degrés)")
 plt.xlabel("Phase")

 plt.tight_layout()
 plt.show()
# Afficher les colonnes
 print("Colonnes détectées :", df.columns.tolist())
def calcuate_thresholds():
# Remplacer les éventuelles valeurs manquantes

# Liste des colonnes à analyser
 features = ['left_elbow_shoulder_hip', 'right_elbow_shoulder_hip', 
            'right_hip_knee_angle', 'left_hip_knee_angle', 
            'right_elbow_angle', 'left_elbow_angle', 
            'wrist', 'elbow']

# Calcul des statistiques globales
 print("\n📊 Seuils globaux :")
 stats = df[features].describe().transpose()
 print(stats[['min', 'max', 'mean', 'std']])

# ✅ Si tu veux calculer les seuils par phase (label) :
 print("\n📊 Seuils par phase (label) :")
 grouped_stats = df.groupby('label')[features].agg(['min', 'max', 'mean', 'std'])

# Affiche les résultats
 print(grouped_stats)
def sauvgarder_thresholds():
# Chargement du fichier CSV
 features = ['left_shoulder', 'right_shoulder',
            'wrist', 'elbow']

# Création du dictionnaire thresholds
 thresholds = {}
 for phase in df["label"].unique():  # ou ['descente', 'montée'] si tu veux forcer l'ordre
    sub = df[df["label"] == phase]
    thresholds[phase] = {}
    for feature in features:
        # Supprimer les NaN dans la colonne feature
        valid_values = sub[feature].dropna()
        thresholds[phase][feature] = [
            float(valid_values.min()),
            float(valid_values.max()),
            float(valid_values.mean())
        ]

# Sauvegarde dans un fichier JSON
 with open('shoulder_press_counting/thresholds2.json', 'w') as f:
    json.dump(thresholds, f, indent=4)
sauvgarder_thresholds()



