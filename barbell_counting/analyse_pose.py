# This script runs once to generate a dataset CSV from a video for model training
import cv2
import csv
import mediapipe as mp
import math
from annotate import calculate_angle
import joblib
import os

# Chargement du modèle et de l'encodeur
model2 = joblib.load(r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\barbell_counting\model2.pkl")
label_encoder = joblib.load(r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\barbell_counting\label_encoder.pkl")

# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
Landmark = mp.solutions.pose.PoseLandmark

import numpy as np

VISIBILITY_THRESHOLD = 0.5


def safe_distance(p1, p2):
    # si l’un des deux points n’est pas assez visible, renvoyer nan
    if p1.visibility < VISIBILITY_THRESHOLD or p2.visibility < VISIBILITY_THRESHOLD:
        return np.nan
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def extract_ratio(lm):
    ratio = {}
    a = safe_distance(lm[Landmark.LEFT_SHOULDER], lm[Landmark.RIGHT_SHOULDER])
    b = safe_distance(lm[Landmark.LEFT_WRIST], lm[Landmark.RIGHT_WRIST])
    c = safe_distance(lm[Landmark.LEFT_ELBOW], lm[Landmark.RIGHT_ELBOW])
    ratio["wrist"] = a / b if b and not np.isnan(a) and not np.isnan(b) else np.nan
    ratio["elbow"] = a / c if c and not np.isnan(a) and not np.isnan(c) else np.nan
    return ratio


def safe_angle(a, b, c):
    # a, b, c sont des tuples [x, y, visibility]
    if b[2] < VISIBILITY_THRESHOLD or a[2] < VISIBILITY_THRESHOLD or c[2] < VISIBILITY_THRESHOLD:
        return np.nan
    # sinon on calcule l’angle via votre fonction calculate_angle
    return calculate_angle([a[0], a[1]], [b[0], b[1]], [c[0], c[1]])


def extract_angles(lm):
    angles = {}

    # on récupère x, y, visibility pour chaque landmark utile
    def lm3(idx):
        l = lm[idx]
        return (l.x, l.y, l.visibility)

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
    return angles


def analyse_pose(video_path, OUTPUT_CSV):
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

    print(f" FPS de la vidéo : {fps}")
    frame_interval = int(fps // 15)

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(OUTPUT_CSV) == 0:
            writer.writerow(['label', 'left_elbow_shoulder_hip', 'right_elbow_shoulder_hip',
                             'right_elbow_angle', 'left_elbow_angle', 'wrist', 'elbow'])
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo.")
                break

            resized_frame = cv2.resize(frame, (1000, 700))
            if frame_count % frame_interval == 0:
                print(f" Frame {frame_count} analysée.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    angles = extract_angles(landmarks)
                    ratios = extract_ratio(landmarks)
                    if not (np.isnan(angles['right_elbow_angle']) or
                            np.isnan(angles['left_elbow_angle'])):
                        features = np.array([[angles['right_elbow_angle'],
                                              angles['left_elbow_angle']]])
                        prediction = model2.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                    else:
                        label = ''
                    # Affichage de la phase sur la frame
                    cv2.putText(resized_frame, f'Phase: {label}', (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Enregistrement dans le CSV
                    if label != 'milieu' and label != '':
                        writer.writerow([
                            label, angles['left_elbow_shoulder_hip'],
                            angles['right_elbow_shoulder_hip'],
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
    video_path = r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos\barbell biceps curl\barbell biceps curl_19.mp4'
    analyse_pose(video_path, 'biceps_curl_dataset.csv')

