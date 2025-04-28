import cv2
import mediapipe as mp
import math
import numpy as np
from typing import Dict, Tuple, Optional

# Initialisation de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a: Tuple[float, float], 
                   b: Tuple[float, float], 
                   c: Tuple[float, float]) -> float:
    """Calcule l'angle entre trois points (b comme sommet)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return min(angle, 360-angle)

def analyze_pushup(body_parts: Dict[int, Tuple[float, float]]) -> float:
    """Analyse la posture pendant les pompes"""
    # Indices MediaPipe pour les parties du corps
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def get_average_or_single(part1: int, part2: int) -> Optional[Tuple[float, float]]:
        """Retourne la moyenne de deux points ou un seul si l'autre n'existe pas"""
        p1 = body_parts.get(part1)
        p2 = body_parts.get(part2)
        
        if p1 and p2:
            return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
        elif p1:
            return p1
        elif p2:
            return p2
        return None

    # Points clés pour l'analyse
    shoulder = get_average_or_single(LEFT_SHOULDER, RIGHT_SHOULDER)
    hip = get_average_or_single(LEFT_HIP, RIGHT_HIP)
    ankle = get_average_or_single(LEFT_ANKLE, RIGHT_ANKLE)

    if not shoulder or not hip or not ankle:
        return -1  # Points manquants

    try:
        angle = calculate_angle(shoulder, hip, ankle)
        # Pour une posture parfaite, l'angle devrait être proche de 180°
        deviation = abs(180 - angle)
        return deviation
    except:
        return -1

def process_pushup_video(video_path: str):
    """Traite une vidéo de pompes et analyse la posture"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return

    # Paramètres vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialiser MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        frame_count = 0
        good_posture_count = 0
        bad_posture_frames = []

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            frame_count += 1
            current_time = frame_count / fps

            # Conversion couleur et traitement
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Récupération des landmarks
            body_parts = {}
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    body_parts[idx] = (landmark.x * frame_width, landmark.y * frame_height)

                # Analyse de la posture
                deviation = analyze_pushup(body_parts)

                # Dessiner les landmarks
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Afficher les résultats
                if deviation == -1:
                    posture_text = "Points clés non détectés"
                    color = (0, 0, 255)  # Rouge
                elif deviation < 10:  # Seuil de déviation acceptable
                    posture_text = f"Posture bonne: {deviation:.1f}°"
                    color = (0, 255, 0)  # Vert
                    good_posture_count += 1
                else:
                    posture_text = f"Posture mauvaise: {deviation:.1f}°"
                    color = (0, 0, 255)  # Rouge
                    bad_posture_frames.append((current_time, deviation))

                # Affichage sur l'image
                cv2.putText(image, posture_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image, f"Temps: {current_time:.1f}s", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Afficher l'image
                cv2.imshow('Push-up Analysis', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        # Statistiques finales
        if frame_count > 0:
            good_percentage = (good_posture_count / frame_count) * 100
            print("\n=== Résultats d'analyse ===")
            print(f"Total de frames: {frame_count}")
            print(f"Postures correctes: {good_percentage:.1f}%")
            
            if bad_posture_frames:
                print("\nMoments avec mauvaise posture:")
                for time, dev in bad_posture_frames[:10]:  # Affiche les 10 premières erreurs
                    print(f"- {time:.1f}s: déviation de {dev:.1f}°")
            
            # Calculer la déviation moyenne
            if bad_posture_frames:
                avg_dev = sum(d for _, d in bad_posture_frames) / len(bad_posture_frames)
                print(f"\nDéviation moyenne: {avg_dev:.1f}°")

        cap.release()
        cv2.destroyAllWindows()

# Utilisation
if __name__ == "__main__":
    video_path = r"C:\Users\lanouar\sources\Activity_recognition\dataset\push-up\push-up_40.mp4"  # Remplacez par le chemin de votre vidéo
    process_pushup_video(video_path)