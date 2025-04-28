import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from annotate import extract_angles, calculate_angle
from analyse_pose import extract_ratio
import math

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Chargement des modèles
model2 = joblib.load('shoulder_press_counting/model22_shoulder_press.pkl')
label_encoder = joblib.load('shoulder_press_counting/label_encoder_shoulder_press.pkl')

def calculate_hip_shoulder_elbow_angle(lm):
    angle = calculate_angle(
        [lm[Landmark.RIGHT_HIP].x, lm[Landmark.RIGHT_HIP].y],
        [lm[Landmark.RIGHT_SHOULDER].x, lm[Landmark.RIGHT_SHOULDER].y],
        [lm[Landmark.RIGHT_ELBOW].x, lm[Landmark.RIGHT_ELBOW].y]
    )
    return angle

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.4)
Landmark = mp_pose.PoseLandmark
mp_drawing = mp.solutions.drawing_utils

# Chargement des seuils
with open('shoulder_press_counting/thresholds2.json', 'r') as f:
    thresholds = json.load(f)

# Vérifie si un angle est dans les seuils tolérés
def in_thresholds(angle, min_val, max_val):
    return (min_val - 10) <= angle <= (max_val + 15)

# Check if required body parts are detected
def are_key_points_detected(landmarks):
    if not landmarks:
        return False
    required_landmarks = [
        Landmark.RIGHT_SHOULDER,
        Landmark.LEFT_SHOULDER,
        Landmark.RIGHT_ELBOW,
        Landmark.LEFT_ELBOW,
        Landmark.RIGHT_WRIST,
        Landmark.LEFT_WRIST
    ]
    for landmark in required_landmarks:
        if not landmarks[landmark].visibility > 0.5:  # Check if landmark is visible
            return False
    return True

# Analyse d'une vidéo pour le shoulder press
def test_video(video_path):
    current_phase = None
    counter = 0
    feedback = ""
    score = 0
    phase = None
    y_offset = 150

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la vidéo : {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        badratio = {'elbow': '', 'wrist': ''}
        shoulder_angle = np.nan
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks :
          landmarks = results.pose_landmarks.landmark
          angles = extract_angles(landmarks)
          ratio = extract_ratio(landmarks)
          shoulder_angle = calculate_hip_shoulder_elbow_angle(landmarks)
          features = np.array([[angles['right_shoulder'], angles['left_shoulder']]])
          b = distance(landmarks[Landmark.LEFT_WRIST], landmarks[Landmark.RIGHT_WRIST])
          print(b)
          if b > 0.5:
                cv2.imshow(" shoulder_press incorrect", frame)
          else:
            phase = model2.predict(features)
            phase = label_encoder.inverse_transform(phase)[0]

            bad_angles = {}

            if phase != 'milieu':
                for angle_name, angle_value in angles.items():
                    if not np.isnan(angle_value):
                        min_val, max_val = thresholds[phase][angle_name][0], thresholds[phase][angle_name][1]
                        if not in_thresholds(angle_value, min_val, max_val):
                            bad_angles[angle_name] = angle_value

                for angle_name, angle_value in bad_angles.items():
                    min_val, max_val = thresholds[phase][angle_name][0], thresholds[phase][angle_name][1]
                    text = f"{angle_name}: {angle_value:.1f} (Out of [{min_val}-{max_val}])"
                    print(text)
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25

                min_w, max_w = thresholds[phase]['wrist'][0], thresholds[phase]['wrist'][1]
                min_e, max_e = thresholds[phase]['elbow'][0], thresholds[phase]['elbow'][1]

                if not np.isnan(ratio['wrist']) and not in_thresholds(ratio['wrist'], min_w, max_w):
                    badratio['wrist'] = "les mains sont rapprochées" if ratio['wrist'] < min_w else "écartez un peu les mains"
                if not np.isnan(ratio['elbow']) and not in_thresholds(ratio['elbow'], min_e, max_e):
                    badratio['elbow'] = "les coudes sont rapprochés" if ratio['elbow'] < min_e else "écartez un peu les coudes"

            # Analyse des phases
            if phase == "haut":
                if not bad_angles:
                    current_phase = "haut"
                    if not np.isnan(shoulder_angle) and shoulder_angle > 160:
                        feedback = "bon posture"
                    else:
                        feedback = 'your position is incorrect'
                else:
                    feedback = "Mauvaise posture (haut)"
            elif phase == "bas":
                if not bad_angles and current_phase == 'haut':
                    feedback = "shoulder press valide !"
                    counter += 1
                    current_phase = 'bas'
                    score += 5
            else:
                feedback = ""

            # UI
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            white = (255, 255, 255)
            green = (0, 255, 0)
            red = (0, 0, 255)
            feedback_color = red if "Mauvaise" in feedback else green

            cv2.putText(frame, f"Phase : {phase}", (10, 25), font, 0.7, white, 2)
            cv2.putText(frame, f"Repetitions : {counter}", (10, 55), font, 0.7, green, 2)
            cv2.putText(frame, f"Score : {score}", (10, 85), font, 0.7, green, 2)
            cv2.putText(frame, f"Feedback : {feedback}", (10, 115), font, 0.6, feedback_color, 2)

            if badratio['elbow']:
                cv2.putText(frame, f"elbow : {badratio['elbow']}", (10, 155), font, 0.6, red, 1)
            if badratio['wrist']:
                cv2.putText(frame, f"wrist : {badratio['wrist']}", (10, 175), font, 0.6, red, 1)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Show the frame only if key points are detected
        
            cv2.imshow("Phase shoulder_press - Prédiction", frame)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Lancer l'analyse
if __name__ == "__main__":
    video_path = r'C:\Users\lanouar\sources\Activity_recognition\dataset\shoulder press\shoulder press_25.mp4'
    test_video(video_path)