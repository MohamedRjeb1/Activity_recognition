import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Spécifier le chemin du dataset
dataset_path = r"C:\Users\moham\OneDrive\Desktop\PCD_from_scratch\DATA\athlet_videos"

# Lister les classes (dossiers)
all_classes_names = os.listdir(dataset_path)
plt.figure(figsize = (200, 200))
random_range = random.sample(range(len(all_classes_names)), 4)
for counter, random_index in enumerate(random_range, 1):
    # Retrieve a Class Name using the Random Index.
    selected_class_Name = all_classes_names[random_index]

    # Retrieve the list of all the video files present in the randomly selected Class Directory.
    video_files_names_list = os.listdir(f"{dataset_path}/{selected_class_Name}")

    # Randomly select a video file from the list retrieved from the randomly selected Class Directory.
    selected_video_file_name = random.choice(video_files_names_list)

    # Initialize a VideoCapture object to read from the video File.
    video_reader = cv2.VideoCapture(f"{dataset_path}/{selected_class_Name}/{selected_video_file_name}")

    # Read the first frame of the video file.
    _, bgr_frame = video_reader.read()

    # Release the VideoCapture object.
    video_reader.release()

    # Convert the frame from BGR into RGB format.
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Write the class name on the video frame.
    cv2.putText(rgb_frame, selected_class_Name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the frame.
    plt.subplot(5, 4, counter);
    plt.imshow(rgb_frame);
    plt.axis('off')
    plt.show()
