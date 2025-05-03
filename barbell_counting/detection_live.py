import cv2
import json
import joblib
import numpy as np
import mediapipe as mp
import math
import warnings

# --- 1) Supprimer les warnings inutiles ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2) Config ---
VIDEO_SOURCE = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos\barbell biceps curl\barbell biceps curl_18.mp4"
VISIBILITY_THRESH = 0.7
PROBA_THRESH = 0.7
PHASE_SMOOTHING_FRAMES = 3  # Nombre de frames pour lisser la détection de phase

# --- 3) Charger modèle, encodeur et seuils ---
stage_clf = joblib.load(r"./model2.pkl")
label_encoder = joblib.load(r"./label_encoder.pkl")
thresholds = json.load(open("./thresholds.json"))

# --- 4) Initialisation MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Spécifications de dessin
landmark_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(thickness=1)  # Traits plus fins

# --- 5) Fonctions utilitaires ---
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (math.hypot(*ba) * math.hypot(*bc) + 1e-8)
    return math.degrees(math.acos(max(-1, min(1, cosv))))

def get_pt_vis(lm, name, w, h):
    p = lm[mp_pose.PoseLandmark[name].value]
    return (int(p.x * w), int(p.y * h)), p.visibility

def smooth_phase(current_phase, new_phase, phase_history):
    """Lisse la détection de phase pour éviter les fluctuations."""
    phase_history.append(new_phase)
    if len(phase_history) > PHASE_SMOOTHING_FRAMES:
        phase_history.pop(0)
    # Retourne la phase majoritaire dans l'historique
    return max(set(phase_history), key=phase_history.count)

# --- 6) Préparer la fenêtre ---
cv2.namedWindow("Biceps Curl Live", cv2.WINDOW_NORMAL)

# --- 7) Boucle principale ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
counter = 0
current_phase = None
phase_history = []  # Historique pour lisser les phases

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Réduire la taille pour performances
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    h, w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Initialisation des textes
    phase_text = "Phase: --"
    reps_text = f"Reps: {counter}"
    feedbacks = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Calcul des points et visibilités
        Lsh, vis_ls = get_pt_vis(lm, "LEFT_SHOULDER", w, h)
        Lel, vis_le = get_pt_vis(lm, "LEFT_ELBOW", w, h)
        Lwr, vis_lw = get_pt_vis(lm, "LEFT_WRIST", w, h)
        Rsh, vis_rs = get_pt_vis(lm, "RIGHT_SHOULDER", w, h)
        Rel, vis_re = get_pt_vis(lm, "RIGHT_ELBOW", w, h)
        Rwr, vis_rw = get_pt_vis(lm, "RIGHT_WRIST", w, h)

        if min(vis_ls, vis_le, vis_lw, vis_rs, vis_re, vis_rw) >= VISIBILITY_THRESH:
            # Calcul des angles des coudes
            left_angle = angle(Lsh, Lel, Lwr)
            right_angle = angle(Rsh, Rel, Rwr)

            # Prédiction de phase
            feats = np.array([[left_angle, right_angle]])
            idx = stage_clf.predict(feats)[0]
            phase = label_encoder.inverse_transform([idx])[0]
            prob = stage_clf.predict_proba(feats)[0].max()

            # Lisser la phase
            if prob >= PROBA_THRESH:
                phase = smooth_phase(current_phase, phase, phase_history)
                # Compter les reps sur transition bas→haut
                if current_phase == "bas" and phase == "haut":
                    counter += 1
                current_phase = phase
            else:
                phase = current_phase or phase

            phase_text = f"Phase: {phase} ({prob:.2f})"
            reps_text = f"Reps: {counter}"

            # Feedbacks pour les phases "haut" ou "bas"
            if phase in ("haut", "bas"):
                # Calcul des ratios et angles supplémentaires
                sh_dist = dist(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                el_dist = dist(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                               lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                wr_dist = dist(lm[mp_pose.PoseLandmark.LEFT_WRIST.value],
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                r_elb = el_dist / (sh_dist + 1e-8)
                r_wri = wr_dist / (sh_dist + 1e-8)

                Lhi, _ = get_pt_vis(lm, "LEFT_HIP", w, h)
                a_sh_el_hp_l = angle(Lhi, Lsh, Lel)
                Rhi, _ = get_pt_vis(lm, "RIGHT_HIP", w, h)
                a_sh_el_hp_r = angle(Rhi, Rsh, Rel)

                th = thresholds[phase]

                # Vérification des seuils pour les angles et ratios
                angle_checks = {
                    "Left Elbow Angle": (left_angle, th["left_shoulder_elobow_wrist_angle_ratio"]),
                    "Right Elbow Angle": (right_angle, th["right_shoulder_elobow_wrist_angle_ratio"]),
                    "Left Shoulder Angle": (a_sh_el_hp_l, th["left_shoulder_elbow_hip_angle_ratio"]),
                    "Right Shoulder Angle": (a_sh_el_hp_r, th["right_shoulder_elbow_hip_angle_ratio"])
                }

                ratio_checks = {
                    "Elbow Distance Ratio": (r_elb, th["shoulder_elbow_ratio"]),
                    "Wrist Distance Ratio": (r_wri, th["shoulder_wrist_ratio"])
                }

                # Générer les feedbacks
                for name, (val, (lo, hi)) in angle_checks.items():
                    if val < lo or val > hi:
                        feedbacks.append(f"{name}: {'low' if val < lo else 'high'} ({val:.1f}°)")
                    else:
                        feedbacks.append(f"{name}: OK ({val:.1f}°)")

                for name, (val, (lo, hi)) in ratio_checks.items():
                    if val < lo or val > hi:
                        feedbacks.append(f"{name}: {'low' if val < lo else 'high'} ({val:.1f})")
                    else:
                        feedbacks.append(f"{name}: OK ({val:.1f})")

                # Dessiner les lignes de distance avec la bonne couleur
                for name, (val, (lo, hi)) in ratio_checks.items():
                    color = (255, 255, 255) if lo <= val <= hi else (0, 0, 255)
                    if "Elbow" in name:
                        cv2.line(img_bgr, Lel, Rel, color, 2)
                    elif "Wrist" in name:
                        cv2.line(img_bgr, Lwr, Rwr, color, 2)

                # Dessiner la ligne entre les épaules en blanc (référence)
                cv2.line(img_bgr, Lsh, Rsh, (255, 255, 255), 2)

                # Dessiner les connexions du squelette
                for (start, end) in mp_pose.POSE_CONNECTIONS:
                    p1 = lm[start]
                    p2 = lm[end]
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(img_bgr, pt1, pt2, (0, 255, 0), 1)

                # Dessiner les points clés
                for lm_pt in lm:
                    pt = (int(lm_pt.x * w), int(lm_pt.y * h))
                    cv2.circle(img_bgr, pt, 3, (255, 255, 255), -1)

    # Affichage de toutes les informations dans une colonne à droite
    feedback_x = w - 200  # Position x pour les feedbacks
    feedback_y_start = 30  # Position y de départ pour les feedbacks
    line_height = 20  # Hauteur de chaque ligne de texte

    # Liste des informations à afficher
    info_texts = [
        "Activity: Barbell Biceps Curl",
        reps_text,
        phase_text
    ] + feedbacks

    # Afficher les textes avec fontScale ajusté et couleur adaptée
    for i, text in enumerate(info_texts):
        color = (255, 255, 255) if "OK" in text or "Activity" in text or "Reps" in text or "Phase" in text else (0, 0, 255)
        cv2.putText(img_bgr, text, (feedback_x, feedback_y_start + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.imshow("Biceps Curl Live", img_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()