
#this code is executed for  single time to create the file (squat_counting\annotated_angles.csv) as a dataset for model_training 
import cv2
import csv
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
Landmark = mp.solutions.pose.PoseLandmark

LABEL_MAP = {'h': 'haut', 'm': 'milieu', 'b': 'bas'}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_angles(lm):
    angles = {}
    angles['left_knee'] = calculate_angle(
        [lm[Landmark.LEFT_HIP].x, lm[Landmark.LEFT_HIP].y],
        [lm[Landmark.LEFT_KNEE].x, lm[Landmark.LEFT_KNEE].y],
        [lm[Landmark.LEFT_ANKLE].x, lm[Landmark.LEFT_ANKLE].y]
    )

    angles['left_hip'] = calculate_angle(
        [lm[Landmark.LEFT_SHOULDER].x, lm[Landmark.LEFT_SHOULDER].y],
        [lm[Landmark.LEFT_HIP].x, lm[Landmark.LEFT_HIP].y],
        [lm[Landmark.LEFT_KNEE].x, lm[Landmark.LEFT_KNEE].y]
    )

    angles['right_hip'] = calculate_angle(
        [lm[Landmark.RIGHT_SHOULDER].x, lm[Landmark.RIGHT_SHOULDER].y],
        [lm[Landmark.RIGHT_HIP].x, lm[Landmark.RIGHT_HIP].y],
        [lm[Landmark.RIGHT_KNEE].x, lm[Landmark.RIGHT_KNEE].y]
    )

    angles['right_knee'] = calculate_angle(
        [lm[Landmark.RIGHT_HIP].x, lm[Landmark.RIGHT_HIP].y],
        [lm[Landmark.RIGHT_KNEE].x, lm[Landmark.RIGHT_KNEE].y],
        [lm[Landmark.RIGHT_ANKLE].x, lm[Landmark.RIGHT_ANKLE].y]
    )

    return angles

def annotate_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Impossible d’ouvrir la vidéo : {video_path}")
        return
    print("✅ Vidéo ouverte avec succès.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("⚠️ Impossible de lire les FPS.")
        return
    print(f"🎞️ FPS de la vidéo : {fps}")

    frame_interval = int(fps // 10)  # Garder 5 frames par seconde
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
   
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("📤 Fin de la vidéo.")
                break

            if frame_count % frame_interval == 0:
                print(f"🎯 Frame {frame_count} analysée.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if not results.pose_landmarks:
                    return None
                lm = results.pose_landmarks.landmark
                angles = extract_angles(lm)
                if angles is None:
                    print("⚠️ Aucun corps détecté.")
                    frame_count += 1
                    continue

                # Affichage sur la frame
                text = f"L_Knee: {angles['left_knee']:.1f} | R_Knee: {angles['right_knee']:.1f}"
                cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("Annotation (appuie sur h/m/b ou q)", frame)
                key = cv2.waitKey(0) & 0xFF
                print(f"⌨️ Touche pressée : {chr(key) if key != 255 else 'Aucune'}")

                if key == ord('q'):
                    print("🛑 Fin de l'annotation.")
                    break
                elif key in [ord('h'), ord('m'), ord('b')]:
                    label = LABEL_MAP[chr(key)]
                    writer.writerow([
                        label,
                        angles['left_knee'],
                        angles['left_hip'],
                        angles['right_hip'],
                        angles['right_knee']
                    ])
                    print(f"✅ Frame annotée avec le label : {label}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("🎬 Vidéo et fenêtres fermées.")


# 👉 Utilisation :
if __name__ == "__main__":
    df = pd.read_csv('squat_annotation/annotated_angles.csv')
    print(df["label"].value_counts())

