import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Initialisation
le = LabelEncoder()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configuration des landmarks et angles
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "RIGHT_ELBOW", "LEFT_ELBOW",
    "RIGHT_WRIST", "LEFT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "RIGHT_KNEE", "LEFT_KNEE",
    "RIGHT_ANKLE", "LEFT_ANKLE"
]

ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
]

def calculate_angle(a, b, c):
    """Calcule l'angle entre trois points"""
    a = np.array(a[:2])  # Prend seulement x et y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def extract_features(image):
    """Extrait les caractéristiques (angles) d'une image"""
    with mp_pose.Pose(static_image_mode=False, 
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5) as pose:
        
        # Conversion et traitement de l'image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Dessiner les landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            features = []
            
            # Calcul des angles pour chaque jointure
            for joint1, joint2, joint3 in ANGLE_JOINTS:
                try:
                    a = landmarks[mp_pose.PoseLandmark[joint1].value]
                    b = landmarks[mp_pose.PoseLandmark[joint2].value]
                    c = landmarks[mp_pose.PoseLandmark[joint3].value]
                    
                    angle = calculate_angle([a.x, a.y], [b.x, b.y], [c.x, c.y])
                    features.extend([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
                except:
                    features.extend([0.0, 0.0])  # Valeurs par défaut si échec
                    
            return features
        return None

def prepare_sequence(video_path, sequence_length=30):
    """Prépare une séquence de caractéristiques à partir d'une vidéo"""
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened() and len(sequence) < sequence_length:
        success, frame = cap.read()
        if not success:
            break
            
        # Extraction des caractéristiques
        features = extract_features(frame)
        
        if features is not None:
            sequence.append(features)
        
        # Affichage de la vidéo en temps réel
        cv2.imshow('Reconnaissance d\'activité', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(sequence) == sequence_length:
        return np.array(sequence)
    return None

def predict_activity(video_path, model):
    """Prédit l'activité dans une vidéo"""
    sequence = prepare_sequence(video_path)
    
    if sequence is not None:
        # Préparation des données pour le modèle
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        # Prédiction
        prediction = model.predict(sequence)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_label = le.inverse_transform(predicted_label)
        
        print(f"Activité prédite : {predicted_label[0]}")
    else:
        print("Impossible d'analyser la vidéo (trop courte ou pas de pose détectée)")

if __name__ == "__main__":
    # Chargement du modèle
    model_path = r'C:\Users\lanouar\sources\Activity_recognition\Model _using_mediapipe\saved_model\lstm_model24.keras'
    model = load_model(model_path)
    
    # Encodage des labels
    le.fit(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])
    
    # Chemin de la vidéo
    video_path = r'C:\Users\lanouar\Videos\Vidéo sans titre ‐ Réalisée avec Clipchamp.mp4'
    
    # Prédiction
    predict_activity(video_path, model)