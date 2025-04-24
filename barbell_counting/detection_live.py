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
VIDEO_SOURCE      = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos\barbell biceps curl\barbell biceps curl_18.mp4"             # 0 pour webcam ou chemin vers vidéo
VISIBILITY_THRESH = 0.7
PROBA_THRESH      = 0.7

# --- 3) Charger modèle de phase et encodeur, et seuils JSON ---
stage_clf     = joblib.load(r"./model2.pkl")
label_encoder = joblib.load(r"./label_encoder.pkl")
thresholds    = json.load(open("./thresholds.json"))

# --- 4) Initialisation MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)

# drawing specs
landmark_spec   = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(thickness=1)

# --- 5) Helpers ---
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosv = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc) + 1e-8)
    return math.degrees(math.acos(max(-1, min(1, cosv))))

def get_pt_vis(lm, name, w, h):
    p = lm[mp_pose.PoseLandmark[name].value]
    return (int(p.x*w), int(p.y*h)), p.visibility

# --- 6) Préparer la fenêtre ---
cv2.namedWindow("Biceps Curl Live", cv2.WINDOW_NORMAL)

# --- 7) Boucle live avec compteur ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
counter = 0
current_phase = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # réduire taille pour perf
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    h, w = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # barre de statut
    cv2.rectangle(img_bgr, (0,0), (w,60), (0,0,0), -1)
    phase_text = "Phase: --"
    reps_text  = f"Reps: {counter}"
    feedbacks  = []

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # calcul des deux angles de coude
        Lsh, vis_ls = get_pt_vis(lm, "LEFT_SHOULDER", w, h)
        Lel, vis_le = get_pt_vis(lm, "LEFT_ELBOW",    w, h)
        Lwr, vis_lw = get_pt_vis(lm, "LEFT_WRIST",    w, h)
        Rsh, vis_rs = get_pt_vis(lm, "RIGHT_SHOULDER",w, h)
        Rel, vis_re = get_pt_vis(lm, "RIGHT_ELBOW",   w, h)
        Rwr, vis_rw = get_pt_vis(lm, "RIGHT_WRIST",   w, h)

        if min(vis_ls,vis_le,vis_lw,vis_rs,vis_re,vis_rw) >= VISIBILITY_THRESH:
            left_angle  = angle(Lsh, Lel, Lwr)
            right_angle = angle(Rsh, Rel, Rwr)

            # prédiction de phase
            feats = np.array([[left_angle, right_angle]])
            idx   = stage_clf.predict(feats)[0]
            phase = label_encoder.inverse_transform([idx])[0]
            prob  = stage_clf.predict_proba(feats)[0].max()

            # compter les reps sur transition bas→haut
            if prob >= PROBA_THRESH:
                if current_phase=="bas" and phase=="haut":
                    counter += 1
                current_phase = phase
            else:
                phase = current_phase or phase

            phase_text = f"Phase: {phase} ({prob:.2f})"
            reps_text  = f"Reps: {counter}"

            # n’appliquer corrections que pour "haut" ou "bas"
            if phase in ("haut","bas"):
                # calcul ratios & angles supplémentaires
                sh_dist = dist(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                el_dist = dist(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                               lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                wr_dist = dist(lm[mp_pose.PoseLandmark.LEFT_WRIST.value],
                               lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
                r_elb = el_dist/(sh_dist+1e-8)
                r_wri = wr_dist/(sh_dist+1e-8)

                Lhi, _ = get_pt_vis(lm, "LEFT_HIP", w, h)
                a_sh_el_hp_l = angle(Lhi, Lsh, Lel)
                Rhi, _ = get_pt_vis(lm, "RIGHT_HIP", w, h)
                a_sh_el_hp_r = angle(Rhi, Rsh, Rel)

                th = thresholds[phase]

                checks = [
                    ("shoulder_elbow_ratio", r_elb, th["shoulder_elbow_ratio"], [(Lsh,Lel),(Rsh,Rel)]),
                    ("shoulder_wrist_ratio",  r_wri, th["shoulder_wrist_ratio"],  [(Lsh,Lwr),(Rsh,Rwr)]),
                    ("l_sh_el_wr_angle", left_angle,  th["left_shoulder_elobow_wrist_angle_ratio"], [(Lsh,Lel),(Lel,Lwr)]),
                    ("r_sh_el_wr_angle", right_angle, th["right_shoulder_elobow_wrist_angle_ratio"],[(Rsh,Rel),(Rel,Rwr)]),
                    ("l_sh_el_hp_angle", a_sh_el_hp_l, th["left_shoulder_elbow_hip_angle_ratio"], [(Lhi,Lsh),(Lsh,Lel)]),
                    ("r_sh_el_hp_angle", a_sh_el_hp_r, th["right_shoulder_elbow_hip_angle_ratio"],[(Rhi,Rsh),(Rsh,Rel)])
                ]

                # default skeleton color green
                conn_colors = {conn: (0,255,0) for _,_,_,conns in checks for conn in conns}

                for name, val, (lo,hi), conns in checks:
                    if val < lo or val > hi:
                        feedbacks.append(f"{name} {'low' if val<lo else 'high'} ({val:.1f})")
                        for conn in conns:
                            conn_colors[conn] = (0,0,255)

                # draw connections
                for (start,end) in mp_pose.POSE_CONNECTIONS:
                    p1 = lm[start]; p2 = lm[end]
                    pt1 = (int(p1.x*w), int(p1.y*h))
                    pt2 = (int(p2.x*w), int(p2.y*h))
                    color = (0,255,0)
                    for (a,b),c in conn_colors.items():
                        if (pt1,pt2)==(a,b) or (pt1,pt2)==(b,a):
                            color = c
                            break
                    cv2.line(img_bgr, pt1, pt2, color, 2)

                # draw keypoints
                for lm_pt in lm:
                    pt = (int(lm_pt.x*w), int(lm_pt.y*h))
                    cv2.circle(img_bgr, pt, 3, (255,255,255), -1)

    # affichage barre de statut
    col_w = w//4
    cv2.putText(img_bgr, phase_text, (5,20),  cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(img_bgr, reps_text,  (col_w+5,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    for i,msg in enumerate(feedbacks[:2]):
        cv2.putText(img_bgr, msg, ((2+i)*col_w+5,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    cv2.imshow("Biceps Curl Live", img_bgr)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
