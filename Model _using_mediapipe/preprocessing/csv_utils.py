import os
import csv
import numpy as np
from preprocessing.keypoint_extraction import HEADERS
from config import IMPORTANT_LMS
from preprocessing.angle_calculation import calculate_angle

ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
]
import mediapipe as mp
mp_pose = mp.solutions.pose
def init_csv(dataset_path: str):
    if os.path.exists(dataset_path):
        return  # Ne rien faire si le fichier existe déjà
    with open(dataset_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(HEADERS)

def export_keypoints_to_csv(path, keypoints, label):
    keypoints.insert(0, label)
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keypoints)
def export_landmark_to_csv(dataset_path: str, results, label: str):
    try:
        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in IMPORTANT_LMS:
            point = landmarks[mp_pose.PoseLandmark[lm].value]
            keypoints.append([point.x, point.y, point.z, point.visibility])

        keypoints_flat = list(np.array(keypoints).flatten())

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
