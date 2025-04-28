import cv2
import mediapipe as mp
import numpy as np
import math

# Initialisation de MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle if angle <= 180 else 360 - angle

def analyze_curl(shoulder, elbow, wrist):
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    feedback = ""
    
    if elbow_angle > 160:
        feedback = "Bras tendu"
    elif elbow_angle < 30:
        feedback = "Bonne contraction"
    else:
        feedback = f"Angle: {int(elbow_angle)}°"
    
    return feedback, elbow_angle

# Configuration de la capture vidéo
video_path = r'C:\Users\lanouar\sources\Activity_recognition\dataset\barbell biceps curl\barbell biceps curl_42.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
    exit()

# Variables pour le suivi
counter = 0
stage = None
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = max(1, int(1000/fps))  # Temps d'attente entre les frames en ms

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Fin de la vidéo ou erreur de lecture")
            break
            
        try:
            # Redimensionnement et conversion de couleur
            frame = cv2.resize(frame, (800, 1000))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Détection de la pose
            results = pose.process(image)
            
            # Conversion pour l'affichage
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Récupération des points clés
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Analyse et comptage
                feedback, elbow_angle = analyze_curl(shoulder, elbow, wrist)
                
                if elbow_angle > 140:
                    stage = "down"
                elif elbow_angle < 50 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(f"Répétition comptée! Total: {counter}")
                
                # Affichage des informations
                cv2.rectangle(image, (0,0), (300,100), (245,117,16), -1)
                cv2.putText(image, f'REPS: {counter}', (15,40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(image, feedback, (15,80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Dessin des landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            cv2.imshow('Analyse de Curl', image)
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Erreur de traitement: {e}")
            break

cap.release()
cv2.destroyAllWindows()