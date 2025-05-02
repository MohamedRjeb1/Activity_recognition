import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


df = pd.read_csv('Model _using_mediapipe/keypoints6 (1).csv')


# Suppression des lignes où la personne est debout : angle trop ouvert
#df = df[~((df['label'] == 'barbell biceps curl') & (df['LEFT_SHOULDER_LEFT_ELBOW_LEFT_WRIST_angle'] > 175))]
#df=df[df['label']!="hammer curl"]
# Sauvegarde du dataset nettoyé
data=df[['label','LEFT_SHOULDER_LEFT_ELBOW_LEFT_WRIST_angle',
       'RIGHT_SHOULDER_RIGHT_ELBOW_RIGHT_WRIST_angle',
       'LEFT_HIP_LEFT_KNEE_LEFT_ANKLE_angle',
       'RIGHT_HIP_RIGHT_KNEE_RIGHT_ANKLE_angle']]
def normalize_angles(df):
    """
    Normalise les angles articulaires entre -1 et 1 en conservant la relation cyclique des angles
    """
    # Création d'une copie pour ne pas modifier l'original
    normalized_df = df.copy()

    # Normalisation sinus/cosinus pour préserver la circularité des angles
    for col in ['LEFT_SHOULDER_LEFT_ELBOW_LEFT_WRIST_angle',
       'RIGHT_SHOULDER_RIGHT_ELBOW_RIGHT_WRIST_angle',
       'LEFT_HIP_LEFT_KNEE_LEFT_ANKLE_angle',
       'RIGHT_HIP_RIGHT_KNEE_RIGHT_ANKLE_angle']:
        # Conversion en radians
        # Normalisation circulaire
        normalized_df[col]= df[col]/180

        

    return normalized_df
df=normalize_angles(data)
df.to_csv('Model _using_mediapipe/keypoints6 (1).csv', index=False)
