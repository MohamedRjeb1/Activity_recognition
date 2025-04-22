import cv2
import numpy as np
import joblib
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
le = LabelEncoder()

# les Landmarks importants
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP","RIGHT_KNEE","LEFT_KNEE","RIGHT_ANKLE","LEFT_ANKLE"
]
#les  angles importants
ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
]
#normalizing keypoints coordinations
def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints)
    return (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints))


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    a, b, c are each a list or array of [x, y]
    Returns the angle in degrees
    """
    a = np.array(a[:2])  # Only x and y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)
#fonction to rescale the frame
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# Fonction pour extraire les keypoints
mp_pose = mp.solutions.pose

# Fonction pour extraire les keypoints d'une frame (utiliser dans le test)
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

        # Calculate angles
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


# Fonction pour préparer la séquence des keypoints (cette fonction est utiliser dans le test)
def prepare_sequence(video_path, sequence_length=20):
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

# fonction pour faire la prédiction
def predict_activity(video_path,model):

# Préparer la séquence de keypoints
  sequence = prepare_sequence(video_path)

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
if __name__ == "__main__":
 sequence_length=20
 #load the model
 model = load_model(r'C:\Users\lanouar\sources\Activity_recognition\Model _using_mediapipe\saved_model\final_lstm_mode.keras')
 le.fit_transform(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])
 video_path = r'Model _using_mediapipe\model\test_video.mp4'
 predict_activity(video_path,model)
