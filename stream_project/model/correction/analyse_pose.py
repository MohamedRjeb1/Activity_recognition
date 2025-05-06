# This script runs once to generate a dataset CSV from a video for model training
import cv2
import csv
import mediapipe as mp
import math
import numpy as np
import joblib
import os
def extract_angles_shoulder_press(lm):
   
    
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
def calculate_angle(a, b, c):
    """Calcule l'angle entre trois points"""
    a = np.array(a[:2])  # Prend seulement x et y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)
# Initialisation de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)
Landmark = mp.solutions.pose.PoseLandmark

import numpy as np

VISIBILITY_THRESHOLD = 0.5
def smooth_point(prev_point, new_point, alpha=0.75):
    return alpha * prev_point + (1 - alpha) * new_point

def safe_distance(p1, p2):
    # si l’un des deux points n’est pas assez visible, renvoyer nan
    if p1.visibility < VISIBILITY_THRESHOLD or p2.visibility < VISIBILITY_THRESHOLD:
        return np.nan
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_ratio(lm):
    ratio = {}
    a = safe_distance(lm[Landmark.LEFT_SHOULDER], lm[Landmark.RIGHT_SHOULDER])
    b = safe_distance(lm[Landmark.LEFT_WRIST],    lm[Landmark.RIGHT_WRIST])
    c = safe_distance(lm[Landmark.LEFT_ELBOW],    lm[Landmark.RIGHT_ELBOW])
    ratio["wrist"] = a / b if b and not np.isnan(a) and not np.isnan(b) else np.nan
    ratio["elbow"] = a / c if c and not np.isnan(a) and not np.isnan(c) else np.nan
    return ratio
def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def safe_angle(a, b, c):
    # a, b, c sont des tuples [x, y, visibility]
    if b[3] < VISIBILITY_THRESHOLD or a[3] < VISIBILITY_THRESHOLD or c[3] < VISIBILITY_THRESHOLD:
        return np.nan
    # sinon on calcule l’angle via votre fonction calculate_angle
    return calculate_angle([a[0], a[1],a[2]], [b[0], b[1],b[2]], [c[0], c[1],c[2]])

def extract_angles(lm):
    angles = {}
    # on récupère x, y, visibility pour chaque landmark utile
    def lm3(idx):
        l = lm[idx]
        return (l.x, l.y,l.z, l.visibility)

    angles['left_elbow_shoulder_hip'] = safe_angle(
        lm3(Landmark.LEFT_ELBOW),
        lm3(Landmark.LEFT_SHOULDER),
        lm3(Landmark.LEFT_HIP)
    )
    angles['right_elbow_shoulder_hip'] = safe_angle(
        lm3(Landmark.RIGHT_ELBOW),
        lm3(Landmark.RIGHT_SHOULDER),
        lm3(Landmark.RIGHT_HIP)
    )
    angles['left_elbow_angle'] = calculate_angle(
        [lm[Landmark.LEFT_SHOULDER].x, lm[Landmark.LEFT_SHOULDER].y],
        [lm[Landmark.LEFT_ELBOW].x, lm[Landmark.LEFT_ELBOW].y],
        [lm[Landmark.LEFT_WRIST].x, lm[Landmark.LEFT_WRIST].y]
    )

    angles['right_elbow_angle'] = calculate_angle(
        [lm[Landmark.RIGHT_SHOULDER].x, lm[Landmark.RIGHT_SHOULDER].y],
        [lm[Landmark.RIGHT_ELBOW].x, lm[Landmark.RIGHT_ELBOW].y],
        [lm[Landmark.RIGHT_WRIST].x, lm[Landmark.RIGHT_WRIST].y]
    )
   
    angles['left_hip_knee_angle'] = safe_angle(
        lm3(Landmark.LEFT_SHOULDER),
        lm3(Landmark.LEFT_HIP),
        lm3(Landmark.LEFT_KNEE)
    )
    angles['right_hip_knee_angle'] = safe_angle(
        lm3(Landmark.RIGHT_SHOULDER),
        lm3(Landmark.RIGHT_HIP),
        lm3(Landmark.RIGHT_KNEE)
    )
    return angles
