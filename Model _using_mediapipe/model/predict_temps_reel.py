import cv2
import numpy as np
from predict import extract_keypoints  # Assurez-vous que cette fonction existe
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialiser le label encoder avec les labels utilisés pendant l'entraînement
le = LabelEncoder()
le.fit(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])

def prepare_sequence(video_path, sequence_length=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Erreur lors de l'ouverture de la vidéo.")
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

    cap.release()

    if len(sequence) == sequence_length:
        return np.array(sequence)  # Séquence complète de keypoints
    else:
        return None  # Séquence incomplète

def predict_activity(video_path, model):
    # Préparer la séquence de keypoints
    sequence = prepare_sequence(video_path)

    if sequence is not None:
        # Reshape pour LSTM : (1, sequence_length, keypoint_dim)
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

        # Prédiction
        prediction = model.predict(sequence)

        # Décoder la prédiction
        predicted_label = np.argmax(prediction, axis=1)
        predicted_label = le.inverse_transform(predicted_label)

        print(f"✅ L'activité prédite est : {predicted_label[0]}")
    else:
        print("⚠️ La séquence est trop courte ou aucun keypoint détecté dans la vidéo.")

# Exemple d'utilisation
if __name__ == "__main__":
     #load the model
 model = load_model(r'C:\Users\lanouar\sources\Activity_recognition\Model _using_mediapipe\saved_model\final_lstm_mode.keras')
 le.fit_transform(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])
 video_path = r"C:\Users\lanouar\sources\Activity_recognition\dataset\barbell biceps curl\barbell biceps curl_7.mp4"
 predict_activity(video_path,model)
