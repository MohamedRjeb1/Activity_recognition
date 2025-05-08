import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pyttsx3
import av

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def extract_coordinates_features(landmarks):
    keypoints_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]
    features = []
    for idx in keypoints_indices:
        landmark = landmarks[idx]
        features.extend([landmark.x, landmark.y])
    return np.array(features)

def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_angle_sin_cos(a, b, c):
    angle_deg = calculate_angle(a, b, c)
    angle_rad = np.radians(angle_deg)
    return np.sin(angle_rad), np.cos(angle_rad)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
landmark_spec   = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(thickness=1)

ANGLE_JOINTS_INDICES = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

def extract_angles_features(landmarks):
    features = np.zeros(8)
    for i, (j1, j2, j3) in enumerate(ANGLE_JOINTS_INDICES):
        try:
            a = landmarks[j1]
            b = landmarks[j2]
            c = landmarks[j3]
            sin, cos = calculate_angle_sin_cos([a.x, a.y], [b.x, b.y], [c.x, c.y])
            features[2*i] = sin
            features[2*i+1] = cos
        except:
            features[2*i] = 0
            features[2*i+1] = 0
    return features

def predict_from_frame(prediction_label, frame, pose, sequence, model, label_encoder, l, seq_length=30):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    confidence = 0.0
    if results.pose_landmarks:
        if l[0] % 2 == 0:
            features = extract_angles_features(results.pose_landmarks.landmark)
            sequence.append(features)
        if len(sequence) > seq_length:
            sequence.pop(0)
        l[0] += 1
        if len(sequence) == seq_length and l[0] % l[1] == 0:
            input_seq = np.array(sequence).reshape(1, seq_length, 8)
            prediction = model.predict(input_seq, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]
            if confidence > 0.9:
                prediction_label = label_encoder.inverse_transform([predicted_index])[0]
    return prediction_label, confidence, results

model = load_model(r'model\lstm_model24.keras')
le = LabelEncoder()
le.fit(['barbell biceps curl', 'push-up', 'shoulder press', 'squat'])
pose_model = mp_pose.Pose(static_image_mode=False)

class RealTimeProcessor(VideoProcessorBase):
    def __init__(self):
        self.sequence = []
        self.l = [0, 15]
        self.prev_landmarks = {}
        self.prev_keypoints = []
        self.phase_prediction = ''
        self.prediction_label = "En traitement..."
        self.activity_summary = {
            "push-up": 0,
            "shoulder press": 0,
            "barbell biceps curl": 0,
            "squat": 0
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        activity, conf, results = predict_from_frame(self.prediction_label, img, pose_model, self.sequence, model, le, self.l)

        if self.prediction_label != activity:
            self.prediction_label = activity
            self.phase_prediction = ''

        if results and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

        cv2.putText(img, f"Activité: {activity}", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


