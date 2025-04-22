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
    angles['left_shoulder'] = calculate_angle(
        [lm[Landmark.LEFT_SHOULDER].x, lm[Landmark.LEFT_SHOULDER].y],
        [lm[Landmark.LEFT_ELBOW].x, lm[Landmark.LEFT_ELBOW].y],
        [lm[Landmark.LEFT_WRIST].x, lm[Landmark.LEFT_WRIST].y]
    )

    angles['right_shoulder'] = calculate_angle(
        [lm[Landmark.RIGHT_SHOULDER].x, lm[Landmark.RIGHT_SHOULDER].y],
        [lm[Landmark.RIGHT_ELBOW].x, lm[Landmark.RIGHT_ELBOW].y],
        [lm[Landmark.RIGHT_WRIST].x, lm[Landmark.RIGHT_WRIST].y]
    )
    return angles

def annotate_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    # Définir la taille souhaitée (par exemple 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


    if not cap.isOpened():
        print(f"Impossible d’ouvrir la vidéo : {video_path}")
        return
    print(" Vidéo ouverte avec succès.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Impossible de lire les FPS.")
        return
    print(f"FPS de la vidéo : {fps}")
    import os
    frame_interval = int(fps // 10)  # Garder 5 frames par seconde
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(output_csv) == 0:
          writer.writerow(['label','right_shoulder','left_shoulder'])
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(" Fin de la vidéo.")
                break
            resized_frame = cv2.resize(frame, (1000, 700))
            if frame_count % frame_interval == 0:
                print(f" Frame {frame_count} analysée.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                  landmarks = results.pose_landmarks.landmark
                angles = extract_angles(landmarks)
            

                if angles is None:
                    print(" Aucun corps détecté.")
                    frame_count += 1
                    continue

                # Affichage sur la frame
                text = f"left_shoulder: {angles['left_shoulder']:.1f} right_shoulder: {angles['right_shoulder']:.1f}"
                cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("Annotation (appuie sur h/m/b ou q)", resized_frame)
                key = cv2.waitKey(0) & 0xFF
                print(f"⌨️ Touche pressée : {chr(key) if key != 255 else 'Aucune'}")

                if key == ord('q'):
                    print(" Fin de l'annotation.")
                    break
                elif key in [ord('h'), ord('m'), ord('b')]:
                    label = LABEL_MAP[chr(key)]
                    writer.writerow([
                        label,
                        angles['right_shoulder'],
                        angles['left_shoulder']
                       
                    ])
                    print(f"Frame annotée avec le label : {label}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Vidéo et fenêtres fermées.")


#  Utilisation :
if __name__ == "__main__":
    video_path='push_up_counting/pushup.mp4'
    annotate_video(video_path,'push_up_counting/annotated_angles.csv')
    #df = pd.read_csv('push_up_counting/annotated_angles_push_up.csv')
    #print(df["label"].value_counts())

