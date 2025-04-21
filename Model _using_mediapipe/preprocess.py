from preprocessing.process_video import process_video
import os
from preprocessing.csv_utils import init_csv
from config import DATASET_DIR
from config import KEYPOINT_CSV_PATH
output_csv=KEYPOINT_CSV_PATH
data_path =DATASET_DIR
activities = os.listdir(data_path)
init_csv(output_csv)

# Parcourir les activités
for activity in activities:
    activity_path = os.path.join(data_path, activity)
    videos = os.listdir(activity_path)

    for video in videos:
        video_path = os.path.join(activity_path, video)
        process_video(video_path, output_csv)
        print("Le dataset   des keypoints a été créé for "+video_path)
    print("Le dataset   des keypoints a été créé avec succès fro "+activity)

print("Le dataset des keypoints a été créé avec succès !")