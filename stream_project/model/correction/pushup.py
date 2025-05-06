import cv2
import mediapipe as mp
import numpy as np
import json
import joblib
from collections import deque
from .analyse_pose import extract_ratio, extract_angles

# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.4)
# Chargement du modèle et des labels
model = joblib.load(r'model\correction\models\model2_push_up.pkl')
label = joblib.load(r'model\correction\models\label_encoder_push_up.pkl')

# Chargement des seuils de correction
with open(r'model\correction\thresholds\thresholds_push_up.json', 'r') as f:
    thresholds = json.load(f)

# Historique des angles pour le lissage
angle_history = {}
window_size = 5


def smooth_point(prev_point, new_point, alpha=0.5):
    """Applique un lissage exponentiel entre deux points."""
    return alpha * prev_point + (1 - alpha) * new_point


def smooth_angles(new_angles):
    """Lisse les angles articulaires sur une fenêtre glissante."""
    smoothed = {}
    for name, value in new_angles.items():
        if name not in angle_history:
            angle_history[name] = deque(maxlen=window_size)
        angle_history[name].append(value)
        smoothed[name] = np.mean(angle_history[name])
    return smoothed


def in_thresholds(angle, min_val, max_val):
    """Vérifie si un angle est dans les seuils avec une marge de tolérance."""
    return (min_val - 10) <= angle <= (max_val + 15)


def predict_phase(angles):
    """Prédit la phase du push-up à partir des angles donnés."""
    features = np.array([
        angles['right_elbow_angle'],
        angles['left_elbow_angle']
    ]).reshape(1, -1)
    phase_encoded = model.predict(features)[0]
    return label.inverse_transform([phase_encoded])[0]


def correct_pushup(frame, results, prev_landmarks, counter, current_phase):
    """
    Corrige le push-up en analysant les angles articulaires et la posture.
    Retourne l'image annotée, les landmarks précédents, le compteur et la phase actuelle.
    """
    feedback = ""
    score = 0

    if results.pose_landmarks:
        raw_landmarks = results.pose_landmarks.landmark
        smoothed_landmarks = []

        for i, lm in enumerate(raw_landmarks):
            new_point = np.array([lm.x, lm.y, lm.z])
            prev_point = prev_landmarks.get(i, new_point)
            smoothed = smooth_point(prev_point, new_point)
            prev_landmarks[i] = smoothed

            smoothed_landmarks.append(type('Landmark', (), {
                'x': smoothed[0],
                'y': smoothed[1],
                'z': smoothed[2],
                'visibility': lm.visibility
            })())

        angles = extract_angles(smoothed_landmarks)
        ratio = extract_ratio(smoothed_landmarks)
        smoothed_angles = smooth_angles(angles)
        phase = predict_phase(smoothed_angles)

        bad_angles = {}
        badratio = {'elbow': '', 'wrist': ''}

        if phase != 'milieu':
         for angle_name, angle_value in smoothed_angles.items():
                if not np.isnan(angle_value):
                    min_val, max_val = thresholds[phase][angle_name][0], thresholds[phase][angle_name][1]
                    if not in_thresholds(angle_value, min_val, max_val):
                        bad_angles[angle_name] = angle_value

         y_offset = 170
         for angle_name, angle_value in bad_angles.items():
            min_val, max_val = thresholds[phase][angle_name][0], thresholds[phase][angle_name][1]
            text = f"{angle_name}: {angle_value:.1f} (Out of [{min_val}-{max_val}])"
            print(text)
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25

        # Ratio feedback
         min_w, max_w = thresholds[phase]['wrist'][0],thresholds[phase]['wrist'][1]
         min_e, max_e = thresholds[phase]['elbow'][0],thresholds[phase]['elbow'][1]

         if not in_thresholds(ratio['wrist'], min_w, max_w) and not np.isnan(ratio['wrist']):
            badratio['wrist'] = 'rapprocher les mains' if ratio['wrist'] > max_w else 'les mains sont trop proches'

         if not in_thresholds(ratio['elbow'], min_e, max_e) and not np.isnan(ratio['elbow']):
            badratio['elbow'] = 'rapprocher les coudes' if ratio['elbow'] > max_e else 'les coudes sont trop proches'

        # Feedback par phase
        if phase == "bas":
            current_phase = "bas" if not bad_angles else current_phase
            feedback = "Descente OK" if not bad_angles else "Mauvaise posture (bas)"

        elif phase == "haut" and current_phase == 'milieu':
            if not bad_angles:
                current_phase = "haut"
                counter += 1
                score += 5
                feedback = "Push-Up valide !"
            else:
                feedback = "Mauvaise posture (haut)"
                angle = smoothed_angles['right_elbow_angle']
                score += 4 if angle > 90 else 3 if angle > 60 else 1

        # Alignement du dos
        feedback2 = (" Bon alignement du dos."
                     if smoothed_angles['left_hip_knee_angle'] >= 150 and smoothed_angles['right_hip_knee_angle'] >= 150
                     else " Mauvais alignement du dos !")

        # Overlay & affichage
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        green, red, white = (0, 255, 0), (0, 0, 255), (255, 255, 255)

        cv2.putText(frame, f"Repetitions : {counter}", (10, 55), font, 0.7, green, 2)
        cv2.putText(frame, f"Score : {score}", (10, 85), font, 0.7, green, 2)
        cv2.putText(frame, f"Feedback : {feedback}", (10, 115), font, 0.6, red if "Mauvaise" in feedback else green, 2)
        cv2.putText(frame, f"Dos : {feedback2}", (10, 135), font, 0.6, white, 1)

        if badratio['elbow']:
            cv2.putText(frame, f"elbow : {badratio['elbow']}", (10, 155), font, 0.6, white, 1)
        if badratio['wrist']:
            cv2.putText(frame, f"wrist : {badratio['wrist']}", (10, 175), font, 0.6, white, 1)

    cv2.putText(frame, f"Phase: {phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return frame, prev_landmarks, counter, phase
