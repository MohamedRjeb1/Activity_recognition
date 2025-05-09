import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from model.correction.pushup import correct_pushup
from model.correction.shoulder_press import correct_shoulder_press
from model.correction.biceps_curl import correct_barbell_biceps
from model.correction.squat import correct_squat
import streamlit as st
import pyttsx3
engine = pyttsx3.init()

def speak(text):
    """
    Utilise la synthèse vocale pour prononcer le texte donné.
    """
    engine.say(text)
    engine.runAndWait()
def extract_coordinates_features(landmarks):
    """
    Extrait les coordonnées x, y des points clés spécifiés.

    Args:
        landmarks: Liste des landmarks détectés par MediaPipe.

    Returns:
        Vecteur de 26 coordonnées (13 points * x et y).
    """
    keypoints_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]

    features = []
    for idx in keypoints_indices:
        landmark = landmarks[idx]
        features.extend([landmark.x, landmark.y])
    return np.array(features)

# === Angle Calculation Utilities ===
def calculate_angle(a, b, c):
    """
    Calcule l'angle formé par trois points a, b et c (en 2D).
    
    Args:
        a, b, c: Coordonnées (x, y) des points.

    Returns:
        Angle en degrés entre les segments ab et bc.
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_angle_sin_cos(a, b, c):
    """
    Calcule le sinus et le cosinus de l'angle formé par trois points.
    
    Args:
        a, b, c: Coordonnées (x, y) des points.

    Returns:
        Tuple (sin(angle), cos(angle)).
    """
    angle_deg = calculate_angle(a, b, c)
    angle_rad = np.radians(angle_deg)
    return np.sin(angle_rad), np.cos(angle_rad)

# === Feature Extraction ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
landmark_spec   = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(thickness=1)


# Indices des articulations utilisées pour extraire les angles
ANGLE_JOINTS_INDICES = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

def extract_angles_features(landmarks):
    """
    Extrait les sinus et cosinus des angles des principales articulations.
    
    Args:
        landmarks: Liste des points de repère détectés par MediaPipe.

    Returns:
        Vecteur de caractéristiques (sinus et cosinus) des angles des articulations.
    """
    features = np.zeros(8)
    for i, (j1, j2, j3) in enumerate(ANGLE_JOINTS_INDICES):
        try:
            a = landmarks[j1]
            b = landmarks[j2]
            c = landmarks[j3]
            sin, cos = calculate_angle_sin_cos([a.x, a.y], [b.x, b.y], [c.x, c.y])
            features[2*i] = sin
            features[2*i+1] = cos
        except:
            features[2*i] = 0
            features[2*i+1] = 0
    return features

# === Prediction from Frame ===
def predict_from_frame(prediction_label, frame, pose, sequence, model, label_encoder, l, seq_length=30):
    """
    Traite une image pour détecter une activité à partir des angles articulaires.
    
    Args:
        prediction_label: Libellé actuel de la prédiction.
        frame: Image vidéo à traiter.
        pose: Objet MediaPipe Pose.
        sequence: Liste des séquences de caractéristiques temporelles.
        model: Modèle LSTM entraîné.
        label_encoder: Encodeur pour convertir les prédictions en libellés.
        l: Liste contenant un compteur d’images et un intervalle de prédiction.
        seq_length: Longueur de la séquence à considérer pour la prédiction.

    Returns:
        Tuple (libellé de prédiction, confiance, résultats MediaPipe).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    confidence = 0.0

    if results.pose_landmarks:
        if l[0] % 2 == 0:  # Extraction un frame sur deux
            features = extract_angles_features(results.pose_landmarks.landmark)
            sequence.append(features)

        if len(sequence) > seq_length:
            sequence.pop(0)

        l[0] += 1

        if len(sequence) == seq_length and l[0] % l[1] == 0:
            input_seq = np.array(sequence).reshape(1, seq_length, 8)
            prediction = model.predict(input_seq, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]
            if confidence > 0.9:
                prediction_label = label_encoder.inverse_transform([predicted_index])[0]

    return prediction_label, confidence, results

# === Main Video Processing Loop ===
def predict(video_path):
    frame_placeholder = st.empty()
    """
    Lance la détection et la correction d'activités sportives depuis une vidéo.

    Args:
        video_path: Chemin de la vidéo à analyser.
    """
    model = load_model(r'model\lstm_model24.keras')
    le = LabelEncoder()
    le.fit(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])

    sequence = []
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    l = [0, 15]  # compteur d’images et fréquence d’analyse
    activity_summary = {
    "push-up": 0,
    "shoulder press": 0,
    "barbell biceps curl": 0,
    "squat": 0  # si tu veux l'ajouter plus tard
    }

    prediction_label = "En traitement..."
    prev_landmarks = {}
    prev_keypoints = []  # utilisé pour barbell biceps
    phase_prediction=''
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1000, 800))
        activity, conf, results = predict_from_frame(prediction_label, frame, pose, sequence, model, le, l)

        if prediction_label != activity:
            prediction_label = activity
            phase_prediction=''
        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

            

            # Correction spécifique selon l'activité détectée
            if activity == "push-up":
                frame, prev_landmarks,  activity_summary[activity], phase_prediction = correct_pushup(frame, results, prev_landmarks, activity_summary[activity] , phase_prediction)
            elif activity == "shoulder press":
                frame, prev_landmarks,  activity_summary[activity], phase_prediction = correct_shoulder_press(frame, results, prev_landmarks, activity_summary[activity] , phase_prediction)
            elif activity == "barbell biceps curl":
                frame, prev_keypoints,  activity_summary[activity], phase_prediction = correct_barbell_biceps(frame, results, prev_keypoints, activity_summary[activity] , phase_prediction)
            elif activity=='squat':
                  frame, prev_keypoints,  activity_summary[activity], phase_prediction = correct_squat(frame, results, prev_keypoints, activity_summary[activity] , phase_prediction)
        # Affichage de l’activité en cours
        cv2.putText(frame, f"Activité: {activity}", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame_placeholder.image(frame, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    return activity_summary
