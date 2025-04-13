import cv2
import numpy as np
import joblib
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from sklearn.preprocessing import LabelEncoder



# Liste des landmarks importants (comme dans ton dataset)
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

mp_pose = mp.solutions.pose

# Fonction pour extraire les keypoints d'une frame
def extract_keypoints(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        keypoints = []
        for lm in IMPORTANT_LMS:
            landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark[lm].value]
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints).flatten()

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
            break

        success, frame = cap.read()

    cap.release()

    if len(sequence) == sequence_length:
        return np.array(sequence)  # Retourne une séquence complète de keypoints
    else:
        return None  # Si la séquence est trop courte

# Chemin de la nouvelle vidéo à prédire




def predict(video_path):
    model = load_model("saved_model/lstm_model.h5")
    sequence = prepare_sequence(video_path)
    if sequence is not None:
    # Reshaper la séquence pour qu'elle soit compatible avec l'entrée du modèle LSTM
         sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])  # (1, 30, 36)

    # Prédiction
         prediction = model.predict(sequence)
         le = LabelEncoder()
         le.fit(['barbell biceps curl', 'push-up', 'shoulder press', 'squat', 'hammer curl'])
    # Décodage du label
         predicted_label = np.argmax(prediction, axis=1)

         original_label = le.inverse_transform([predicted_label[0]])

     # Décodage du label avec l'encodeur
         print(f"L'activité prédite est : {original_label}")
    else:
     print("La séquence est trop courte ou aucun keypoint détecté dans la vidéo.")
  

def predict_realtime():
    model = load_model("saved_model/lstm_model.h5")
    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            sequence.append(keypoints)

        if len(sequence) == 30:
            input_seq = np.array(sequence).reshape(1, 30, 36)  # (batch, timesteps, features)
            prediction = model.predict(input_seq)

            # Affichage sur l'image
            cv2.putText(frame, f'{prediction}',
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sport Activity Recognition - Real Time', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

