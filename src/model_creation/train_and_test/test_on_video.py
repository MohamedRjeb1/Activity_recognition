import os
from collections import deque
import numpy as np
import cv2
import yt_dlp
from moviepy import VideoFileClip
from tensorflow.keras.models import load_model

from src.preprocessing.normalize_videos import SEQUENCE_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES_LIST

MODEL_PATH = r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\model_creation\train_and_test\LRCN_model___Date_Time_2025_03_28__03_19_14___Loss_0.3862980604171753___Accuracy_0.8837209343910217.h5'
TEST_VIDEO_DIR = r'C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\src\model_creation\train_and_test\test_video'

# Charger le modèle SANS compilation
model = load_model(MODEL_PATH, compile=False)


def download_youtube_video(url, output_dir):
    """Télécharge une vidéo YouTube avec yt-dlp"""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def process_video(input_path, output_path):
    """Effectue la prédiction sur la vidéo"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction = "Initialisation..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prétraitement
        processed = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0
        frame_buffer.append(processed)

        # Prédiction
        if len(frame_buffer) == SEQUENCE_LENGTH:
            pred = model.predict(np.expand_dims(frame_buffer, axis=0), verbose=0)[0]
            prediction = CLASSES_LIST[np.argmax(pred)]

        # Overlay
        cv2.putText(frame, prediction, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame)

    cap.release()
    writer.release()


# ...

if __name__ == "__main__":
    os.makedirs(TEST_VIDEO_DIR, exist_ok=True)

    try:
        input_path = download_youtube_video('https://youtu.be/hIkeJVV-Djk?si=d6LpjjDl-5s4ZC_r', TEST_VIDEO_DIR)
    except Exception as e:
        print(f"Erreur YouTube: {e}\nUtilisation d'une vidéo locale...")
        input_path = "chemin/vers/votre_video_test.mp4"

    output_path = os.path.join(TEST_VIDEO_DIR, "video_analysée.mp4")
    process_video(input_path, output_path)

    # Ouvrir la vidéo avec le lecteur par défaut
    import webbrowser

    webbrowser.open(output_path)
