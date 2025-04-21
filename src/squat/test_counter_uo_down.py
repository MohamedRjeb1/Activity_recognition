# test_model.py

import cv2
import pickle
import numpy as np
import mediapipe as mp

# ---- 1) Configuration ----
VIDEO_SOURCE = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos\squat\squat_18.mp4"
MODEL_PATH   = "./stage_model.pkl"  # le modèle binaire down/up
VISIBILITY_THRESH = 0.7           # ignorer frames mal détectées

# ---- 2) Charger le modèle ----
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ---- 3) Init Mediapipe ----
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

# ---- 4) Fonctions utilitaires ----
def rescale_frame(frame, percent=50):
    """Réduit la taille pour accélérer le traitement."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*percent/100), int(h*percent/100)),
                      interpolation=cv2.INTER_AREA)

def extract_keypoints(results):
    """
    Transforme les 9 landmarks essentiels en vecteur de 36 valeurs [x,y,z,v].
    Même ordre que pour l'entraînement !
    """
    lm = results.pose_landmarks.landmark
    names = [
        "NOSE","LEFT_SHOULDER","RIGHT_SHOULDER",
        "LEFT_HIP","RIGHT_HIP",
        "LEFT_KNEE","RIGHT_KNEE",
        "LEFT_ANKLE","RIGHT_ANKLE"
    ]
    feat = []
    for name in names:
        p = lm[mp_pose.PoseLandmark[name].value]
        feat += [p.x, p.y, p.z, p.visibility]
    return feat

def is_frame_valid(results):
    """Vérifie que les 4 points clés pour prediction sont visibles."""
    lm = results.pose_landmarks.landmark
    # On s'assure que épaules et hanches sont visibles
    idxs = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value
    ]
    return all(lm[i].visibility >= VISIBILITY_THRESH for i in idxs)

# ---- 5) Boucle de test ----
cap = cv2.VideoCapture(VIDEO_SOURCE)
with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5.a) Pré‑traitement
        frame = rescale_frame(frame, 50)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        label = "No pose"
        conf  = 0.0

        # 5.b) Si on a un résultat et qu'il est fiable
        if results.pose_landmarks and is_frame_valid(results):
            row = extract_keypoints(results)
            proba = clf.predict_proba([row])[0]
            idx   = int(np.argmax(proba))
            label = ["down","up"][idx]
            conf  = proba[idx]

        # 5.c) Dessiner le squelette
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1)
        )

        # 5.d) Afficher le label
        text = f"Phase: {label} ({conf:.2f})"
        cv2.putText(img, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

        # 5.e) Afficher la frame
        cv2.imshow("Test Model", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
