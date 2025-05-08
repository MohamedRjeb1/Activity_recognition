import cv2
import mediapipe as mp
import time

# Initialiser MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def fingers_up(hand_landmarks):
    """
    Retourne une liste de 0/1 pour indiquer quels doigts sont levés.
    Index : 0 = pouce, 1 = index, ..., 4 = auriculaire
    """
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Pouce (main droite supposée) : compare la position x
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Autres doigts : compare la position y
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detect_hand(timeout=10):
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    gesture = "Aucun geste détecté"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # effet miroir
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(hand_landmarks)
                total = sum(fingers)

                if total == 5:
                    gesture = "Main ouverte"
                    cap.release()
                    cv2.destroyAllWindows()
                    return gesture
                elif total == 0:
                    gesture = "Main fermée"
                else:
                    gesture = f"{total} doigt(s) levé(s)"

                cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calcul du temps restant
        remaining = timeout - (time.time() - start_time)
        if remaining <= 0:
            break

        cv2.putText(frame, f"Ouvrez la main avant : {int(remaining)}s",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Détection des mains", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # touche ESC pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()
    return gesture
