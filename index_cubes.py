import cv2
import mediapipe as mp
import numpy as np
import random
import time

# --- Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Canvas setup
persistent_canvas = None

# --- Rate Limiting Variables ---
# Time (in seconds) between spawning blocks. Increase this number to make it slower.
SPAWN_DELAY = 0.15 
last_spawn_time = 0

def get_finger_status(landmarks):
    """
    Returns a dictionary indicating which fingers are 'open' (extended).
    True = Open, False = Closed/Curled
    """
    # Landmark Indices: Tip vs PIP (Pip is the middle joint)
    # 8=Index, 12=Middle, 16=Ring, 20=Pinky
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    fingers_open = []
    
    # Check each finger (Index through Pinky)
    for tip, pip in zip(tips, pips):
        # If tip is higher (smaller Y) than pip, finger is Open
        fingers_open.append(landmarks[tip].y < landmarks[pip].y)
        
    return fingers_open # Returns list: [Index, Middle, Ring, Pinky]

def is_pointing(fingers_open):
    # Index Open, others Closed
    return fingers_open[0] and not fingers_open[1] and not fingers_open[2] and not fingers_open[3]

def is_rock_gesture(fingers_open):
    # Index & Pinky Open, Middle & Ring Closed (The "Devil Horns")
    return fingers_open[0] and not fingers_open[1] and not fingers_open[2] and fingers_open[3]

# --- Main Loop ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if persistent_canvas is None:
        persistent_canvas = np.zeros((h, w, c), dtype=np.uint8)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_layer = np.zeros((h, w, c), dtype=np.uint8)

    # 1. Process Face (Green Dots)
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(current_layer, (x, y), 1, (0, 255, 0), -1)

    # 2. Process Hands
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            
            # Draw Hand Skeleton (Red)
            mp_drawing.draw_landmarks(
                current_layer, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Analyze Gestures
            fingers_status = get_finger_status(hand_landmarks.landmark)
            current_time = time.time()

            # CHECK 1: Rock Gesture -> Clear Screen
            if is_rock_gesture(fingers_status):
                persistent_canvas = np.zeros((h, w, c), dtype=np.uint8)
                cv2.putText(current_layer, "CLEARED!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # CHECK 2: Pointing Gesture -> Spawn Block (with Delay)
            elif is_pointing(fingers_status):
                if current_time - last_spawn_time > SPAWN_DELAY:
                    # Get Index Tip coordinates
                    ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                    
                    rand_color = [random.randint(50, 255) for _ in range(3)]
                    block_size = 15
                    
                    cv2.rectangle(persistent_canvas, 
                                  (ix - block_size, iy - block_size), 
                                  (ix + block_size, iy + block_size), 
                                  rand_color, -1)
                    
                    last_spawn_time = current_time

    # 3. Combine and Show
    combined_display = cv2.add(persistent_canvas, current_layer)
    cv2.imshow('me ?', frame)
    cv2.imshow('you ?', combined_display)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()