import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
from .analyse_pose import extract_angles_shoulder_press, calculate_angle,extract_ratio
import math
def distance(p1, p2):
    """
    Calcule la distance euclidienne entre deux points 2D.
    
    Paramètres :
        p1, p2 : objets contenant les attributs x et y (landmarks MediaPipe).
        
    Retour :
        float : distance entre p1 et p2.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
# Chargement des modèles
model2 = joblib.load(r'model\correction\models\model22_shoulder_press.pkl')
label_encoder = joblib.load(r'model\correction\models\label_encoder_shoulder_press.pkl')

def calculate_hip_shoulder_elbow_angle(lm):
    """
    Calcule l’angle entre la hanche, l’épaule et le coude droits.
    Cet angle aide à évaluer la posture durant le shoulder press.
    
    Paramètres :
        lm : liste des landmarks MediaPipe.
        
    Retour :
        float : angle entre hanche, épaule, coude.
    """
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
with open(r'model\correction\thresholds\thresholds_shoulder.json', 'r') as f:
    thresholds = json.load(f)

def in_thresholds(angle, min_val, max_val):
    """
    Vérifie si l’angle est dans les marges tolérées autour du seuil attendu.
    
    Paramètres :
        angle : valeur de l'angle mesuré.
        min_val : valeur minimale attendue.
        max_val : valeur maximale attendue.
        
    Retour :
        bool : True si l’angle est correct, sinon False.
    """
    return (min_val - 10) <= angle <= (max_val + 15)

# Check if required body parts are detected
def are_key_points_detected(landmarks):
    """
    Vérifie si les points clés nécessaires sont visibles.
    
    Paramètres :
        landmarks : liste des landmarks détectés par MediaPipe.
        
    Retour :
        bool : True si tous les points nécessaires sont visibles.
    """

# Analyse d'une vidéo pour le shoulder press
#frame,results,prev_landmarks,counter,phase_prediction
def correct_shoulder_press(frame,results,prev_landmarks,counter,current_phase):
        """
        Analyse une frame pour détecter les erreurs de posture dans un shoulder press.
    
        Paramètres :
          frame : image vidéo actuelle.
          results : résultats de MediaPipe pour cette frame.
          prev_landmarks : historique des landmarks précédents.
          counter : nombre de répétitions correctes.
          current_phase : phase actuelle du mouvement (haut, milieu, bas).
        
         Retour :
          frame : image annotée.
          prev_landmarks : mis à jour.
          counter : mis à jour si mouvement correct.
          phase : phase actuelle détectée.
        """
        feedback = ""
        score = 0
        y_offset = 150
        phase=''
        badratio = {'elbow': '', 'wrist': ''}
        shoulder_angle = np.nan
        if results.pose_landmarks :
          landmarks = results.pose_landmarks.landmark
          angles = extract_angles_shoulder_press(landmarks)
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
                        feedback = "mauvaise posture levez davantage vos mains"
                else:
                    feedback = "Mauvaise posture (haut)"
            elif phase == "bas":
                if  current_phase == 'milieu':
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
        return frame, prev_landmarks,counter,phase
