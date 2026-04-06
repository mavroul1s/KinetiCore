import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

photos = []
last_capture_time = 0.0
CAPTURE_COOLDOWN = 0.8  

def is_rock_gesture(landmarks):
    index_up = landmarks[8].y < landmarks[5].y
    pinky_up = landmarks[20].y < landmarks[17].y
    middle_down = landmarks[12].y > landmarks[9].y
    ring_down = landmarks[16].y > landmarks[13].y
    return index_up and pinky_up and middle_down and ring_down

def is_ok_gesture(landmarks):
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    index_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_up = landmarks[12].y < landmarks[9].y
    ring_up = landmarks[16].y < landmarks[13].y
    pinky_up = landmarks[20].y < landmarks[17].y
    distance = np.linalg.norm(thumb_tip - index_tip)
    return distance < 0.05 and middle_up and ring_up and pinky_up

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    if hand_result.multi_hand_landmarks:
        hand_landmarks = hand_result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = hand_landmarks.landmark

        if is_rock_gesture(landmarks):
            now = time.time()
            if now - last_capture_time > CAPTURE_COOLDOWN:
                last_capture_time = now
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                )

                if len(eyes) > 0:
                    def center_x(rect):
                        x, y, w_e, h_e = rect
                        return x + w_e / 2

                    chosen = max(eyes, key=center_x)
                    ex, ey, ew, eh = chosen
                    pad = int(max(ew, eh) * 0.6)
                    cx = int(ex + ew / 2)
                    cy = int(ey + eh / 2)
                    x1 = max(0, cx - (ew // 2) - pad)
                    x2 = min(w, cx + (ew // 2) + pad)
                    y1 = max(0, cy - (eh // 2) - pad)
                    y2 = min(h, cy + (eh // 2) + pad)

                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        max_size = 100
                        if crop.shape[0] > max_size or crop.shape[1] > max_size:
                            scale = max_size / max(crop.shape[0], crop.shape[1])
                            crop = cv2.resize(
                                crop,
                                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                            )
                        photos.append((crop, (cx, cy)))

        elif is_ok_gesture(landmarks):
            photos.clear()

    for photo, (cx, cy) in photos:
        try:
            ph, pw, _ = photo.shape
            y1, y2 = max(0, cy - ph // 2), min(h, cy - ph // 2 + ph)
            x1, x2 = max(0, cx - pw // 2), min(w, cx - pw // 2 + pw)
            if y2 - y1 > 0 and x2 - x1 > 0:
                region_h, region_w = y2 - y1, x2 - x1
                if photo.shape[0] != region_h or photo.shape[1] != region_w:
                    photo_resized = cv2.resize(photo, (region_w, region_h))
                else:
                    photo_resized = photo
                frame[y1:y2, x1:x2] = photo_resized
        except Exception as e:
            print("Overlay error:", e)
            continue

    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    cv2.imshow("devil", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
