import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paramètres globaux
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
DATASET_DIR = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos"
CLASSES_LIST = ["barbell biceps curl", "hammer curl", "push-up", "shoulder press", "squat"]
seed_constant = 27

# Générateur d'augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)


def augment_video(frames):
    """Applique la même transformation à toutes les frames d'une séquence"""
    if len(frames) == 0:
        return frames

    # Génère une transformation aléatoire unique pour toute la séquence
    transform_params = datagen.get_random_transform(frames[0].shape)
    return [datagen.apply_transform(frame, transform_params) for frame in frames]


def frames_extraction(video_path):
    """Extrait et normalise les frames d'une vidéo"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcul du pas de sélection
    step = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255.0
        frames.append(frame)

    cap.release()
    return frames


def create_dataset(augment=False):
    """Charge et augmente les données en conservant la cohérence temporelle"""
    features = []
    labels = []

    # Collecter tous les chemins et labels
    video_paths = []
    all_labels = []
    for class_id, class_name in enumerate(CLASSES_LIST):
        class_dir = os.path.join(DATASET_DIR, class_name)
        for video_file in os.listdir(class_dir):
            video_paths.append(os.path.join(class_dir, video_file))
            all_labels.append(class_id)

    # Split initial des chemins vidéo
    X_train, X_test, y_train, y_test = train_test_split(
        video_paths, all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=seed_constant
    )

    # Traitement des données d'entraînement avec augmentation
    for path, label in zip(X_train, y_train):
        frames = frames_extraction(path)
        if len(frames) == SEQUENCE_LENGTH:
            # Ajout de l'exemple original
            features.append(frames)
            labels.append(label)

            # Ajout de la version augmentée
            if augment:
                augmented = augment_video(frames)
                features.append(augmented)
                labels.append(label)

    # Conversion en array numpy
    features = np.array(features)
    labels = to_categorical(labels)

    # Traitement des données de test sans augmentation
    test_features = []
    test_labels = []
    for path, label in zip(X_test, y_test):
        frames = frames_extraction(path)
        if len(frames) == SEQUENCE_LENGTH:
            test_features.append(frames)
            test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = to_categorical(test_labels)

    return (features, labels), (test_features, test_labels)


# Création des datasets avec augmentation pour le train
(train_features, train_labels), (test_features, test_labels) = create_dataset(augment=True)

# Vérification des shapes
print(f"Train shape: {train_features.shape}")
print(f"Test shape: {test_features.shape}")