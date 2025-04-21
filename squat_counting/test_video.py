import cv2
import mediapipe as mp
import numpy as np
import joblib 
from annotate import extract_angles

model2=joblib.load( 'squat_counting/model2_squat.pkl')
label_encoder=joblib.load('squat_counting/label_encoder.pkl')
# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
Landmark = mp.solutions.pose.PoseLandmark
mp_drawing = mp.solutions.drawing_utils
# Fonction d’extraction des angles
def test_webcam():
    cap = cv2.VideoCapture(0)  # 0 = caméra par défaut

    if not cap.isOpened():
        print(" Impossible d’accéder à la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Erreur de lecture de la webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = extract_angles(landmarks)

            # Préparer les features pour prédiction
            features = np.array([[angles['left_knee'], angles['left_hip'], angles['right_hip'], angles['right_knee']]])
            prediction = model2.predict(features)
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            # Affichage
            cv2.putText(frame, f'Phase: {predicted_label}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Webcam - Prédiction en temps réel", frame)

        # Quitter avec 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# Analyse de la vidéo
def test_video(video_path):
    new_prediction=""
    old_prediction=""
    compteur=0
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f" Impossible d’ouvrir la vidéo : {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        old_prediction=new_prediction
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
                    return None
        lm = results.pose_landmarks.landmark
        angles = extract_angles(lm)

            # Préparation des données pour la prédiction
        features = np.array([[angles['left_knee'], angles['left_hip'], angles['right_hip'], angles['right_knee']]])
        prediction = model2.predict(features)
            
        new_prediction = label_encoder.inverse_transform(prediction)[0]
        if (new_prediction=="haut"and old_prediction=="milieu"):
                compteur+=1
            
        print(new_prediction)
        print("Angles:", features)


            # Affichage du label sur la vidéo
        cv2.putText(frame, f'Phase: {new_prediction} compteur={compteur}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Dessiner les articulations
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Phase Squat - Prédiction", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Lancer l’analyse
if __name__ == "__main__":
    video_path=''
    test_video(video_path)
