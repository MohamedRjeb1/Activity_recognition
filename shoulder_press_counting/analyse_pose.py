
#this code is executed for  single time to create the file (shoulder_press_counting\annotated_angles.csv) as a dataset for model_training 
import cv2
import csv
import mediapipe as mp
import math
from annotate import extract_angles
# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
Landmark = mp.solutions.pose.PoseLandmark
LABEL_MAP = {'h': 'haut', 'm': '', 'b': 'bas'}
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
def extract_ratio(lm):
    ratio = {}

    a = distance(lm[Landmark.LEFT_SHOULDER], lm[Landmark.RIGHT_SHOULDER])
    b = distance(lm[Landmark.LEFT_WRIST], lm[Landmark.RIGHT_WRIST])
    c = distance(lm[Landmark.LEFT_ELBOW], lm[Landmark.RIGHT_ELBOW])

    ratio["wrist"] = a / b if b != 0 else 0
    ratio["elbow"] = a / c if c != 0 else 0

    return ratio
def analyse_pose(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    # Définir la taille souhaitée (par exemple 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


    if not cap.isOpened():
        print(f" Impossible d’ouvrir la vidéo : {video_path}")
        return
    print(" Vidéo ouverte avec succès.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(" Impossible de lire les FPS.")
        return
    print(f" FPS de la vidéo : {fps}")
    import os
    frame_interval = int(fps // 10)  # Garder 10 frames par seconde
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(output_csv) == 0:
          writer.writerow(['label','right_shoulder','left_shoulder','wrist','elbow'])
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(" Fin de la vidéo.")
                break
            resized_frame = cv2.resize(frame, (1000, 700))
            if frame_count % frame_interval == 0:
                print(f"Frame {frame_count} analysée.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                  landmarks = results.pose_landmarks.landmark
                angles = extract_angles(landmarks)
                ratios=extract_ratio(landmarks)
            

                if angles is None:
                    print("Aucun corps détecté.")
                    frame_count += 1
                    continue

                # Affichage sur la frame
                text = f"left_shoulder: {angles['left_shoulder']:.1f} right_shoulder: {angles['right_shoulder']:.1f}"
                cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("Annotation (appuie sur h/m/b ou q)", resized_frame)
                key = cv2.waitKey(0) & 0xFF
                print(f" Touche pressée : {chr(key) if key != 255 else 'Aucune'}")

                if key == ord('q'):
                    print(" Fin de l'annotation.")
                    break
                elif key in [ord('h'), ord('m'), ord('b')]:
                    label = LABEL_MAP[chr(key)]
                    if label!='':
                     writer.writerow([
                        label,
                        angles['right_shoulder'],
                        angles['left_shoulder'],
                        ratios['wrist'],
                        ratios['elbow']
                       
                     ])
                    print(f" Frame annotée avec le label : {label}")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(" Vidéo et fenêtres fermées.")


#  Utilisation :
if __name__ == "__main__":
    video_path=r'C:\Users\lanouar\sources\Activity_recognition\dataset\shoulder press\shoulder press_25.mp4'
    analyse_pose(video_path,'shoulder_press_counting/analyse.csv')

       

        
    

