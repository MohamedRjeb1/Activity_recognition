import cv2

def extract_frames(video_file, output_folder):
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
    print(f'Frames extracted from {video_file}')
