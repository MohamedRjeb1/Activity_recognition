

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

from src.preprocessing.normalize_videos import CLASSES_LIST

class RealTimeActivitySwitcher:
    def __init__(self, model_path, sequence_length=20, smooth_window=5):
        self.model = load_model(r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\model_creation\train and test\LRCN_model___Date_Time_2025_03_28__03_19_14___Loss_0.3862980604171753___Accuracy_0.8837209343910217.h5")
        self.sequence = deque(maxlen=sequence_length)
        self.prediction_history = deque(maxlen=smooth_window)
        self.current_activity = None
        self.feedback = ""

        # Configuration temps réel
        self.frame_size = (128, 128)
        self.min_confidence = 0.7
        self.transition_threshold = 3  # Nombre de frames pour confirmer un changement

    def process_frame(self, frame):
        # Prétraitement accéléré
        processed = cv2.resize(frame, self.frame_size)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = (processed / 127.5) - 1.0  # Normalisation [-1, 1]

        self.sequence.append(processed)

        if len(self.sequence) == self.sequence.maxlen:
            self._update_predictions()
            self._generate_feedback(frame)

        return self._display_overlay(frame)

    def _update_predictions(self):
        # Inférence asynchrone
        sequence_array = np.expand_dims(np.array(self.sequence), axis=0)
        predictions = self.model.predict(sequence_array, verbose=0)[0]

        current_pred = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence > self.min_confidence:
            self.prediction_history.append(current_pred)

            # Détection de changement d'activité
            if len(set(self.prediction_history)) == 1:  # Consensus sur 5 frames
                new_activity = CLASSES_LIST[current_pred]
                if new_activity != self.current_activity:
                    self.current_activity = new_activity
                    self._on_activity_change()

    def _on_activity_change(self):
        """Déclenché quand une nouvelle activité est détectée"""
        # Personnaliser les feedbacks selon l'activité
        if self.current_activity == "squat":
            self.feedback = "Gardez le dos droit !"
        elif self.current_activity == "push-up":
            self.feedback = "Maintenez l'alignement du corps"
        else:
            self.feedback = ""

    def _generate_feedback(self, frame):
        # Ajouter ici de la logique d'analyse de posture
        pass

    def _display_overlay(self, frame):
        # Affichage optimisé
        frame = cv2.flip(frame, 1)

        if self.current_activity:
            cv2.putText(frame, f"Activite: {self.current_activity}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.feedback:
                cv2.putText(frame, f"Feedback: {self.feedback}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        return frame


# Utilisation
detector = RealTimeActivitySwitcher("workout_model.keras")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = detector.process_frame(frame)
    cv2.imshow('Dynamic Activity Recognition', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
