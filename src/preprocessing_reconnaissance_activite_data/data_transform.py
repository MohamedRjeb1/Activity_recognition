import cv2
import csv
import numpy as np
import os
import mediapipe as mp
data_path = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos"
activities = os.listdir(data_path)
print(activities)
# 2. Importer les librairies nécessaires

"""Step 2: Extract Frames from Videos"""



def extract_frames(video_file, output_folder):
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
    print(f'Frames extracted from {video_file}')

"""Step 3: Extract Keypoints (if using Pose Estimation)
Use Mediapipe or OpenPose to get keypoints:
"""

# Initialisation MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Landmarks importants
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "LEFT_KNEE",
    "RIGHT_ANKLE",
    "LEFT_ANKLE"
]
ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
]
angle_headers = [f"{a}_{b}_{c}_angle" for a, b, c in ANGLE_JOINTS]
HEADERS = ["label"]

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
# Création des colonnes du CSV
# 3. Fonction : Redimensionner l'image
HEADERS = HEADERS+ angle_headers
def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# Fonction pour extraire les keypoints
def extract_keypoints(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])

        normalized_keypoints = normalize_keypoints(keypoints)
        normalized_keypoints = np.array(normalized_keypoints).flatten()

        # Calculate angles
        angles = []
        for joint1, joint2, joint3 in ANGLE_JOINTS:
            a = landmarks[mp_pose.PoseLandmark[joint1].value]
            b = landmarks[mp_pose.PoseLandmark[joint2].value]
            c = landmarks[mp_pose.PoseLandmark[joint3].value]

            angle = calculate_angle([a.x, a.y], [b.x, b.y], [c.x, c.y])
            angles.append(angle)

        # Concatenate keypoints and angles
        all_features = np.concatenate([normalized_keypoints, angles])

        return all_features
# Fonction pour préparer la séquence des keypoints (cette fonction est utiliser dans le test)
def prepare_sequence(video_path, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return None

    sequence = []

    success, frame = cap.read()
    while success:
        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            sequence.append(keypoints)

        if len(sequence) == sequence_length:
            cap.release()
            break

        success, frame = cap.read()




    if len(sequence) == sequence_length:
        return np.array(sequence)  # Retourne une séquence complète de keypoints
    else:
        return None  # Si la séquence est trop courte

"""Step 4: Normalize the Data"""


def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints)
    return (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints))


# Initialiser le fichier CSV avec les headers si non existant
def init_csv(dataset_path: str):
    if os.path.exists(dataset_path):
        return  # Ne rien faire si le fichier existe déjà
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        csv_writer.writerow(HEADERS)


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    a, b, c are each a list or array of [x, y]
    Returns the angle in degrees
    """
    a = np.array(a[:2])  # Only x and y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

# Enregistrer les keypoints + label dans le CSV
def export_landmark_to_csv(dataset_path: str, results, label: str):
    try:
        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])
            normalized_keypoints = normalize_keypoints(keypoints)

        keypoints_flat = list(np.array(normalized_keypoints).flatten())

        # Add angle calculations
        angles = []
        for joint1, joint2, joint3 in ANGLE_JOINTS:
            a = landmarks[mp_pose.PoseLandmark[joint1].value]
            b = landmarks[mp_pose.PoseLandmark[joint2].value]
            c = landmarks[mp_pose.PoseLandmark[joint3].value]

            angle = calculate_angle(
                [a.x, a.y], [b.x, b.y], [c.x, c.y]
            )
            angles.append(angle)

        all_features = [label] + keypoints_flat + angles

        with open(dataset_path, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(all_features)

    except Exception as e:
        print("Erreur:", e)



# Extraire les frames et enregistrer les keypoints dans le CSV
def process_video(video_path, output_csv):
    # Label = nom du dossier parent (= activité)
    activity_label = os.path.basename(os.path.dirname(video_path))
    cap = cv2.VideoCapture("chemin/vers/ta_video.mp4")
    fps_video = 24
    desired_fps = 10
    frame_interval = fps_video // desired_fps
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    frame_count = 0
    saved_frame_count = 0

    cap = cv2.VideoCapture(video_path)
    while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                   export_landmark_to_csv(output_csv, results, activity_label)

            frame_count += 1


    cap.release()
    pose.close()


import os

if __name__ == '__main__':

    output_csv = r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\reconnaissance_d_activite\Data_csv\data_keypoints_final.csv'
    init_csv(output_csv)
    # Créer le dossier de sortie s'il n'existe pas


    # Initialisation du fichier CSV une seule fois

    for activity in activities:
        activity_path = os.path.join(data_path, activity)
        videos = os.listdir(activity_path)
        for video in videos:
            video_path = os.path.join(activity_path, video)
            process_video(video_path, output_csv)
            print("Le dataset   des keypoints a été créé for "+video_path)
        print("Le dataset   des keypoints a été créé avec succès fro "+activity)

    print("Le dataset des keypoints a été créé avec succès !")