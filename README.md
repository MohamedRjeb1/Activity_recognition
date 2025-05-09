# Activity_recognition#  Sport Activity Recognition and Correction

A machine learning project designed to recognize and correct sport activities in real time using pose estimation from videos. This project is part of a PCD (Projet de Conception et Développement) academic requirement.

##  Project Objectives

- Recognize 4 common sport activities using AI:
  - Barbell Biceps Curl
  - Push-Up
  - Shoulder Press
  - Squat
- Analyze movement correctness based on keypoint sequences.
- Provide real-time feedback for exercise correction.

##  Technologies Used

- **Python**
- **MediaPipe** – Pose Estimation
- **OpenCV** – Video Processing
- **TensorFlow (CNN + LSTM)** – Sequence Modeling
- **TSM (Temporal Shift Module)** – Advanced Temporal Analysis (planned)
- **Google Colab** (for training), **VS Code** (for local development)

##  Project Structure

```bash
sport-activity-recognition/
│
├── dataset/                # Organized keypoint CSVs per activity
├── model/
│   ├── train_model.py      # Model training script
│   └── predict.py          # Live prediction & correction logic
│
├── preprocessing/
│   ├── frame_extraction.py     # Extracts frames from videos
│   ├── keypoint_extraction.py  # Extracts pose keypoints with MediaPipe
│   └── csv_utils.py            # Creates structured CSVs for model input
│
├── saved_model/            # Trained models stored here
├── config.py               # Constants and paths
├── main.py                 # Entry point: live camera or video prediction
├── requirements.txt        # Python dependencies
└── README.md  


             # Project documentation
