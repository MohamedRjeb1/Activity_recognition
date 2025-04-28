# This script runs once to generate a dataset CSV from a video for model training
import cv2
import csv
import mediapipe as mp
import math
from annotate import calculate_angle
import numpy as np
import joblib
import os

# Chargement du modèle et de l'encodeur
model2 = joblib.load('push_up_counting/model2_push_up.pkl')
label_encoder = joblib.load('push_up_counting/label_encoder_push_up.pkl')

# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)
Landmark = mp.solutions.pose.PoseLandmark

import numpy as np

VISIBILITY_THRESHOLD = 0.5
def smooth_point(prev_point, new_point, alpha=0.75):
    return alpha * prev_point + (1 - alpha) * new_point

def safe_distance(p1, p2):
    # si l’un des deux points n’est pas assez visible, renvoyer nan
    if p1.visibility < VISIBILITY_THRESHOLD or p2.visibility < VISIBILITY_THRESHOLD:
        return np.nan
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_ratio(lm):
    ratio = {}
    a = safe_distance(lm[Landmark.LEFT_SHOULDER], lm[Landmark.RIGHT_SHOULDER])
    b = safe_distance(lm[Landmark.LEFT_WRIST],    lm[Landmark.RIGHT_WRIST])
    c = safe_distance(lm[Landmark.LEFT_ELBOW],    lm[Landmark.RIGHT_ELBOW])
    ratio["wrist"] = a / b if b and not np.isnan(a) and not np.isnan(b) else np.nan
    ratio["elbow"] = a / c if c and not np.isnan(a) and not np.isnan(c) else np.nan
    return ratio
def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def safe_angle(a, b, c):
    # a, b, c sont des tuples [x, y, visibility]
    if b[3] < VISIBILITY_THRESHOLD or a[3] < VISIBILITY_THRESHOLD or c[3] < VISIBILITY_THRESHOLD:
        return np.nan
    # sinon on calcule l’angle via votre fonction calculate_angle
    return calculate_angle([a[0], a[1],a[2]], [b[0], b[1],b[2]], [c[0], c[1],c[2]])

def extract_angles(lm):
    angles = {}
    # on récupère x, y, visibility pour chaque landmark utile
    def lm3(idx):
        l = lm[idx]
        return (l.x, l.y,l.z, l.visibility)

    angles['left_elbow_shoulder_hip'] = safe_angle(
        lm3(Landmark.LEFT_ELBOW),
        lm3(Landmark.LEFT_SHOULDER),
        lm3(Landmark.LEFT_HIP)
    )
    angles['right_elbow_shoulder_hip'] = safe_angle(
        lm3(Landmark.RIGHT_ELBOW),
        lm3(Landmark.RIGHT_SHOULDER),
        lm3(Landmark.RIGHT_HIP)
    )
    angles['left_elbow_angle'] = calculate_angle(
        [lm[Landmark.LEFT_SHOULDER].x, lm[Landmark.LEFT_SHOULDER].y],
        [lm[Landmark.LEFT_ELBOW].x, lm[Landmark.LEFT_ELBOW].y],
        [lm[Landmark.LEFT_WRIST].x, lm[Landmark.LEFT_WRIST].y]
    )

    angles['right_elbow_angle'] = calculate_angle(
        [lm[Landmark.RIGHT_SHOULDER].x, lm[Landmark.RIGHT_SHOULDER].y],
        [lm[Landmark.RIGHT_ELBOW].x, lm[Landmark.RIGHT_ELBOW].y],
        [lm[Landmark.RIGHT_WRIST].x, lm[Landmark.RIGHT_WRIST].y]
    )
   
    angles['left_hip_knee_angle'] = safe_angle(
        lm3(Landmark.LEFT_SHOULDER),
        lm3(Landmark.LEFT_HIP),
        lm3(Landmark.LEFT_KNEE)
    )
    angles['right_hip_knee_angle'] = safe_angle(
        lm3(Landmark.RIGHT_SHOULDER),
        lm3(Landmark.RIGHT_HIP),
        lm3(Landmark.RIGHT_KNEE)
    )
    return angles

def analyse_pose(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    if not cap.isOpened():
        print(f" Impossible d’ouvrir la vidéo : {video_path}")
        return

    print("Vidéo ouverte avec succès.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(" Impossible de lire les FPS.")
        return

    print(f"🎥 FPS de la vidéo : {fps}")
    frame_interval = int(fps //20)

    # Dictionnaire pour stocker les points précédents
    prev_landmarks = {}
    alpha = 0.75  # coefficient de lissage

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(output_csv) == 0:
            writer.writerow(['label','left_elbow_shoulder_hip' ,'right_elbow_shoulder_hip','right_hip_knee_angle', 'left_hip_knee_angle',
                             'right_elbow_angle', 'left_elbow_angle', 'wrist', 'elbow'])
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("📽️ Fin de la vidéo.")
                break

            resized_frame = cv2.resize(frame, (1000, 700))
            if frame_count % frame_interval == 0:
                print(f" Frame {frame_count} analysée.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    raw_landmarks = results.pose_landmarks.landmark
                    smoothed_landmarks = []

                    for i, lm in enumerate(raw_landmarks):
                        new_point = np.array([lm.x, lm.y, lm.z])
                        if i in prev_landmarks:
                            prev_point = prev_landmarks[i]
                            smoothed = smooth_point(prev_point, new_point, alpha)
                        else:
                            smoothed = new_point
                        prev_landmarks[i] = smoothed

                        class SmoothedLandmark:
                            def __init__(self, x, y, z, visibility):
                                self.x = x
                                self.y = y
                                self.z = z
                                self.visibility = visibility

                        smoothed_landmarks.append(SmoothedLandmark(
                            smoothed[0], smoothed[1], smoothed[2], lm.visibility))

                    # Utiliser les points lissés
                    angles = extract_angles(smoothed_landmarks)
                    ratios = extract_ratio(smoothed_landmarks)

                    if not (np.isnan(angles['right_elbow_angle']) or 
                            np.isnan(angles['left_elbow_angle'])):
                        features = np.array([[angles['right_elbow_angle'],
                                              angles['left_elbow_angle']]])
                        prediction = model2.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                    else:
                        label = ''

                    cv2.putText(resized_frame, f'Phase: {label}', (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    if label != 'milieu' and label != '':
                        writer.writerow([
                            label, angles['left_elbow_shoulder_hip'],
                            angles['right_elbow_shoulder_hip'],
                            angles['right_hip_knee_angle'],
                            angles['left_hip_knee_angle'],
                            angles['right_elbow_angle'],
                            angles['left_elbow_angle'],
                            ratios['wrist'],
                            ratios["elbow"]
                        ])
                    print(f" Frame annotée avec le label : {label}")
                else:
                    print(" Aucun corps détecté.")

                cv2.imshow("Annotation (appuie sur q pour quitter)", resized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(" Arrêt manuel demandé.")
                    break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(" Vidéo et fenêtres fermées.")

#  Lancer le script
if __name__ == "__main__":
    L=['1','5','7','20','36','37']
    L2=['8','9','11','12','38','39','40']
    for i in L2:
     video_path = 'C:/Users/lanouar/sources/Activity_recognition/dataset/push-up/push-up_'+i+'.mp4'
     analyse_pose(video_path, 'push_up_counting/data2.csv')