import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from src.preprocessing.data_prepare import le
from src.preprocessing.data_transform import ANGLE_JOINTS, normalize_keypoints, calculate_angle, IMPORTANT_LMS

model = load_model(r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\Pose_estimation\models_saved\lstm_et_pose_estim2.keras')

mp_pose = mp.solutions.pose

# Fonction pour extraire les keypoints d'une frame
def extract_keypoints(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])

        normalized_keypoints = normalize_keypoints(keypoints)
        normalized_keypoints = np.array(normalized_keypoints).flatten()

        # Calculate angles and append them only once
        angles = []
        for joint1, joint2, joint3 in ANGLE_JOINTS:
            a = landmarks[mp_pose.PoseLandmark[joint1].value]
            b = landmarks[mp_pose.PoseLandmark[joint2].value]
            c = landmarks[mp_pose.PoseLandmark[joint3].value]

            angle = calculate_angle([a.x, a.y], [b.x, b.y], [c.x, c.y])
            angles.append(angle)

        # Concatenate keypoints and angles
        all_features = np.concatenate([normalized_keypoints, angles])

        return all_features


# Fonction pour préparer la séquence des keypoints
def prepare_sequence(video_path, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return None

    sequence = []

    success, frame = cap.read()
    while success:
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            sequence.append(keypoints)

        if len(sequence) == sequence_length:
            cap.release()
            break

        success, frame = cap.read()




    if len(sequence) == sequence_length:
        return np.array(sequence)  # Retourne une séquence complète de keypoints
    else:
        return None  # Si la séquence est trop courte

# Chemin de la nouvelle vidéo à prédire

new_video_path = r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\Pose_estimation\videos_for_test\video_test_1.mp4'
os.path.exists(new_video_path)
# Préparer la séquence de keypoints
sequence = prepare_sequence(new_video_path)

if sequence is not None:
    # Reshaper la séquence pour qu'elle soit compatible avec l'entrée du modèle LSTM
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])  # (1, 30, 36)

    # Prédiction
    prediction = model.predict(sequence)

    # Décodage du label
    predicted_label = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_label)

    # Afficher le résultat
     # Décodage du label avec l'encodeur
    print(f"L'activité prédite est : {predicted_label}")
else:
    print("La séquence est trop courte ou aucun keypoint détecté dans la vidéo.")


