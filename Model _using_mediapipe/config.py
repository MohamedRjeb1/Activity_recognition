# config.py

# Paths
DATASET_DIR = "dataset"
FRAME_DIR = f"{DATASET_DIR}/frames"
KEYPOINT_CSV_PATH = f"output/keypoints.csv"
MODEL_SAVE_PATH = "saved_model/activity_model.h5"

# Video parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
NUM_FRAMES = 30  # nombre de frames utilisées par séquence

# Keypoints parameters
NUM_KEYPOINTS = 33  # Nombre de points de MediaPipe (Pose)
KEYPOINT_DIM = 4    # x, y, z, visibility

# Model parameters
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_CLASSES = 5  # à adapter selon le nombre d'activités

IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "RIGHT_ELBOW",
    "LEFT_ELBOW", "RIGHT_WRIST", "LEFT_WRIST", "LEFT_HIP", "RIGHT_HIP"
]
