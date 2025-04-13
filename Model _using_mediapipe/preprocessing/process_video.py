import cv2
import numpy as np
import csv
import os
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
from preprocessing.csv_utils import export_landmark_to_csv
# Extraire les frames et enregistrer les keypoints dans le CSV
def process_video(video_path, output_csv):
    # Label = nom du dossier parent (= activité)
    activity_label = os.path.basename(os.path.dirname(video_path))

    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    while success:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    export_landmark_to_csv(output_csv, results, activity_label)

                success, image = cap.read()

    cap.release()