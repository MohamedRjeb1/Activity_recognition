
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing_reconnaissance_activite_data.data_prepare import le
from src.preprocessing_reconnaissance_activite_data.data_transform import prepare_sequence, extract_keypoints


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


#preparing train and test data
sequence_length=20
output_csv = '/content/drive/MyDrive/output/keypoints7.csv'


#load the model
model = load_model('/content/drive/MyDrive/SportActivityDataset/DATA/final_lstm_mode.keras')






video_path = "/content/squat-8_3rFkvqkM.mp4"


output_path = "/content/squat-output.mp4"

# Global variable to store the sequence of keypoints
sequence = []
sequence_length = 20 # Same length used during training

# Fonction d'extraction des keypoints (à adapter selon ta logique)

# Traitement de la vidéo
cap = cv2.VideoCapture(video_path)

# Infos de la vidéo originale
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Définir VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extraire les keypoints (selon ton modèle)
    keypoints = extract_keypoints(frame)

    if keypoints is not None:
        # Add keypoints to the sequence
        sequence.append(keypoints)

        # Check if the sequence is long enough
        if len(sequence) == sequence_length:
            # Reshape the sequence for the LSTM model
            input_sequence = np.array(sequence).reshape(1, sequence_length, keypoints.shape[0])

            # Prédire l'activité
            prediction = model.predict(input_sequence)
            activity = np.argmax(prediction)
            activity_name = le.inverse_transform([activity])[0]
            activity_name = f"Activity {activity_name}"  # ou ['push-up', 'squat', ...][activity]

            # Afficher l'activité sur la frame
            cv2.putText(frame, activity_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Reset the sequence for the next prediction
            sequence = []

    # Écrire la frame dans la vidéo de sortie
    cv2.putText(frame, activity_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()