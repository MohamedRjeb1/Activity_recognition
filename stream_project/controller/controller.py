# controller/controller.py

from model.activity_detection import predict
import tempfile
import shutil
from pymongo import MongoClient
import datetime
uri = "mongodb+srv://lanouarkhaled:123456781@cluster0.ul6cpjt.mongodb.net/"
client = MongoClient(uri)
db = client["sport_app_data"]
activity_summaries=db["activity_summaries"]
users_collection = db["users"]
def create_user(user_data):
    return users_collection.insert_one(user_data)
def get_user_by_email(email):
    return users_collection.find_one({"email": email})
def get_activity_summary(id):
    return list(activity_summaries.find({"user_id": id}))
def save_activity_summary(id, summary):
    summary_doc = {
        "user_id": id,
        "summary": summary,
        "timestamp": datetime.datetime.now()
    }
    activity_summaries.insert_one(summary_doc)

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
   
