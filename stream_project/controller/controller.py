# controller/controller.py

from model.activity_detection import predict


import tempfile
import shutil


# def save_detection_result(activity_name: str, prediction: str, timestamp: str):
#     db = get_db()
#     collection = db["detections"]
#     result = {
#         "activity": activity_name,
#         "prediction": prediction,
#         "timestamp": timestamp
#     }
#     collection.insert_one(result)



def detect_activity(video_path):

    return predict(video_path)

def correct_posture(video_file):

    return predict(video_file)
   
