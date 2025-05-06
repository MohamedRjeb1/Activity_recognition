import cv2
import json
import pickle
import numpy as np
import mediapipe as mp
import warnings

# --- 1) Filtrer les warnings sklearn pour ne plus polluer la console ---
warnings.filterwarnings("ignore", category=UserWarning)

#VIDEO_SOURCE      = 0      # 0 = webcam, ou "chemin/vers/video.mp4"
VISIBILITY_THRESH = 0.7
PROBA_THRESH      = 0.7

# --- 3) Charger modèle et seuils ---
with open(r"model\correction\models\model_squat.pkl", "rb") as f:
    stage_clf = pickle.load(f)
thresholds = json.load(open(r"model\correction\thresholds\thresholds_squat.json"))

# --- 4) Initialisation Mediapipe & specs fins ---
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
landmark_spec   = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(thickness=1)

# --- 5) Helpers ---
def rescale(frame, pct=50):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*pct/100), int(h*pct/100)),
                      interpolation=cv2.INTER_AREA)

def extract_keypoints(results):
    lm = results.pose_landmarks.landmark
    names = ["NOSE","LEFT_SHOULDER","RIGHT_SHOULDER",
             "LEFT_HIP","RIGHT_HIP",
             "LEFT_KNEE","RIGHT_KNEE",
             "LEFT_ANKLE","RIGHT_ANKLE"]
    feat = []
    for name in names:
        p = lm[mp_pose.PoseLandmark[name].value]
        feat += [p.x, p.y, p.z, p.visibility]
    return feat

def get_pt_vis(lm, name):
    p = lm[mp_pose.PoseLandmark[name].value]
    return (p.x, p.y), p.visibility

def calc_dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# --- 6) Préparer la fenêtre ---
cv2.namedWindow("Squat Correction Live", cv2.WINDOW_NORMAL)

# --- 7) Boucle live avec compteur ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.4)
Landmark = mp_pose.PoseLandmark

def correct_squat(frame,results,prev_landmarks,counter,current_phase):
        h, w    = frame.shape[:2]

        # Barre noire en haut
        cv2.rectangle(frame, (0,0), (w,60), (0,0,0), -1)

        phase_text = "Phase: --"
        reps_text  = f"Reps: {counter}"
        fb1 = fb2 = ""

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # -- Prédiction de phase (binaire down/up) --
            row    = extract_keypoints(results)
            row_np = np.array(row).reshape(1,-1)            # <--- ici
            probs  = stage_clf.predict_proba(row_np)[0]     # <--- et ici
            idx    = int(np.argmax(probs))
            phase  = ["down","up"][idx]
            conf   = probs[idx]

            if conf >= PROBA_THRESH:
                if current_phase == "down" and phase == "up":
                    counter += 1
                current_phase = phase
            else:
                # si confiance trop faible, on garde l’ancienne phase
                phase = current_phase or phase

            phase_text = f"Phase: {phase} ({conf:.2f})"
            reps_text  = f"Reps: {counter}"

            # -- Ratios avec visibilité et coloration --
            # Récupérer et convertir en pixels + visibilité
            pts, vis = {}, {}
            for name in ("LEFT_SHOULDER","RIGHT_SHOULDER",
                         "LEFT_KNEE","RIGHT_KNEE",
                         "LEFT_ANKLE","RIGHT_ANKLE"):
                raw_pt, v = get_pt_vis(lm, name)
                pts[name] = (int(raw_pt[0]*w), int(raw_pt[1]*h))
                vis[name] = v

            # * Pieds / épaules
            if min(vis["LEFT_SHOULDER"], vis["RIGHT_SHOULDER"],
                   vis["LEFT_ANKLE"],    vis["RIGHT_ANKLE"]) >= VISIBILITY_THRESH:
                sw = calc_dist(pts["LEFT_SHOULDER"], pts["RIGHT_SHOULDER"])
                fw = calc_dist(pts["LEFT_ANKLE"],    pts["RIGHT_ANKLE"])
                r  = fw/(sw+1e-8)
                lo, hi = thresholds[phase]["foot_shoulder_ratio"]
                ok1 = (lo <= r <= hi)
                col1 = (0,255,0) if ok1 else (0,0,255)
                fb1  = f"Feet {'OK' if ok1 else 'Bad'} ({r:.2f})"
                cv2.line(frame, pts["LEFT_ANKLE"], pts["RIGHT_ANKLE"], col1, 1, cv2.LINE_AA)

            # * Genoux / pieds
            if min(vis["LEFT_KNEE"], vis["RIGHT_KNEE"],
                   vis["LEFT_ANKLE"], vis["RIGHT_ANKLE"]) >= VISIBILITY_THRESH:
                kw = calc_dist(pts["LEFT_KNEE"], pts["RIGHT_KNEE"])
                r2 = kw/(fw+1e-8)
                lo2, hi2 = thresholds[phase]["knee_feet_ratio"]
                ok2 = (lo2 <= r2 <= hi2)
                col2 = (0,255,0) if ok2 else (0,0,255)
                fb2  = f"Knees {'OK' if ok2 else 'Bad'} ({r2:.2f})"
                cv2.line(frame, pts["LEFT_KNEE"], pts["RIGHT_KNEE"], col2, 1, cv2.LINE_AA)

            # -- Dessiner squelette (traits fins) --
        
        # -- Affichage de la barre supérieure découpée en 4 colonnes --
        col_w = w // 4
        cv2.putText(frame, phase_text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, reps_text, (col_w+5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, fb1, (2*col_w+5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, fb2, (3*col_w+5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return frame, prev_landmarks, counter, phase

