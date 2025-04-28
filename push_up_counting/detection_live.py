import cv2 
import mediapipe as mp
import numpy as np
import json
import joblib
from collections import deque
from analyse_pose import extract_angles,extract_ratio

# Chargement du modèle
model = joblib.load('push_up_counting/model2_push_up.pkl')
label = joblib.load('push_up_counting/label_encoder_push_up.pkl')
Landmark = mp.solutions.pose.PoseLandmark
VISIBILITY_THRESHOLD = 0.5
def smooth_point(prev_point, new_point, alpha=0.5):
    return alpha * prev_point + (1 - alpha) * new_point
# === Lissage ===
angle_history = {}
window_size = 5

def smooth_angles(new_angles):
    smoothed = {}
    for name, value in new_angles.items():
        if name not in angle_history:
            angle_history[name] = deque(maxlen=window_size)
        angle_history[name].append(value)
        smoothed[name] = np.mean(angle_history[name])
    return smoothed

# Chargement des seuils
with open('push_up_counting/thresholds2.json', 'r') as f:
    thresholds = json.load(f)
# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

# Lecture de la vidéo
video_path = r'C:\Users\lanouar\sources\Activity_recognition\dataset\push-up\pushup.mp4'
cap = cv2.VideoCapture(video_path)

current_phase = None
counter = 0
feedback = ""
score = 0

def in_thresholds(angle, min_val, max_val):
    return ((min_val-10) <= angle <= (max_val+15))

def predict_phase(angles):
    features = np.array([
        angles['right_elbow_angle'],
        angles['left_elbow_angle']
    ]).reshape(1, -1)
    phase_encoded = model.predict(features)[0]
    return label.inverse_transform([phase_encoded])[0]
prev_landmarks = {}


frame_id = 0
while cap.isOpened():
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps_source // 20)
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_id % frame_skip == 0:
        # → Le traitement que vous faites ici
        pass

    frame_id += 1
    
    frame = cv2.resize(frame, (1200, 700))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        raw_landmarks = results.pose_landmarks.landmark
        smoothed_landmarks = []

        for i, lm in enumerate(raw_landmarks):
                        new_point = np.array([lm.x, lm.y, lm.z])
                        if i in prev_landmarks:
                            prev_point = prev_landmarks[i]
                            smoothed = smooth_point(prev_point, new_point)
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
        ratio=extract_ratio(smoothed_landmarks)
        y_offset = 170 
        smoothed_angles = smooth_angles(angles)
        phase = predict_phase(smoothed_angles)
        bad_angles = {}
        badratio={}
        badratio['elbow']=''
        badratio['wrist']=''
        if phase != 'milieu':
         for angle_name, angle_value in smoothed_angles.items():
                if not np.isnan(angle_value):
                    min_val, max_val = thresholds[phase][angle_name][0],thresholds[phase][angle_name][1]
                    if not in_thresholds(angle_value, min_val, max_val):
                        bad_angles[angle_name] = angle_value
            
         for angle_name, angle_value in bad_angles.items():
            
           
           if not np.isnan(angle_value):
              min_val, max_val = thresholds[phase][angle_name][0],thresholds[phase][angle_name][1]
              text = f"{angle_name}: {angle_value:.1f} (Out of [{min_val}-{max_val}])"
              print(text)
              cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
              y_offset += 25
         min_w,max_w=thresholds[phase]['wrist'][0],thresholds[phase]['wrist'][1]
         min_e,max_e=thresholds[phase]['elbow'][0],thresholds[phase]['elbow'][1]
         if not in_thresholds(ratio['wrist'],min_w,max_w)and (not np.isnan(ratio['wrist'])):
            if ratio['wrist']<min_w:
             badratio['wrist']='les mains sont raproché '
            else:
             badratio['wrist']='rapprocher les mains ' 
         if not in_thresholds(ratio['elbow'],min_e,max_e)and not np.isnan(ratio['elbow']):
            if ratio['elbow']<min_w:
             badratio['elbow']='elbow sont raproché '
            else:
             badratio['elbow']='rapprocher elbow ' 
        if phase == "bas":
            if not bad_angles:
                current_phase = "bas"
                feedback = "Descente OK"
            else:
                feedback = "Mauvaise posture (bas)"
        elif phase == "haut"and current_phase=='bas':
            if not bad_angles:
                current_phase = "haut"
                feedback = "Push-Up valide !"
                counter += 1
                score += 5  # Bonus score pour bonne exécution
            else:
                feedback = "Mauvaise posture (haut)"
                angle = smoothed_angles['right_elbow_angle']
                if angle > 90:
                    score += 4
                elif angle > 60:
                    score += 3
                else:
                    score += 1
        else:
            feedback = ""

        # Vérification de l’alignement du dos
        if (smoothed_angles['left_hip_knee_angle'] < 150 or  
            smoothed_angles['right_hip_knee_angle'] < 150):
            feedback2 = " Mauvais alignement du dos !"+str(smoothed_angles['left_hip_knee_angle'])
        else:
            feedback2 = " Bon alignement du dos."

        # Affichage
                # Dessiner un rectangle semi-transparent
        h,w=frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)  # zone noire
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Police & couleur
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_white = (255, 255, 255)
        color_green = (0, 255, 0)
        color_red = (0, 0, 255)

        # Texte organisé
        cv2.putText(frame, f"Phase : {phase}", (10, 25), font, 0.7, color_white, 2)
        cv2.putText(frame, f"Repetitions : {counter}", (10, 55), font, 0.7, color_green, 2)
        cv2.putText(frame, f"Score : {score}", (10, 85), font, 0.7, color_green, 2)
        
        feedback_color = color_red if "Mauvaise" in feedback else color_green
        cv2.putText(frame, f"Feedback : {feedback}", (10, 115), font, 0.6, feedback_color, 2)
        cv2.putText(frame, f"Dos : {feedback2}", (10, 135), font, 0.6, color_white, 1)
        if badratio['elbow']!='':
            cv2.putText(frame, f"elbow : {badratio['elbow']}", (10, 155), font, 0.6, color_white, 1)
        if badratio['wrist']!='':
            cv2.putText(frame, f"wrist : {badratio['wrist']}", (10, 175), font, 0.6, color_white, 1)
            print(ratio['elbow'])
            
        # Affichage des erreurs angulaires si présentes
        y_offset = 160
       


    cv2.imshow("Push-Up Correction", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
