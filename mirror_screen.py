import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure Face Mesh (Refine landmarks ensures detail around eyes/lips)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configure Hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 1. Prepare the frames
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Create the black background canvas with the same size as the camera frame
    h, w, c = frame.shape
    black_background = np.zeros((h, w, c), dtype=np.uint8)

    # Convert the BGR image to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Process Face Mesh
    face_results = face_mesh.process(frame_rgb)
    
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Loop through all 468+ face landmarks
            for id, lm in enumerate(face_landmarks.landmark):
                # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
                x, y = int(lm.x * w), int(lm.y * h)
                
                # Draw ONLY the points in GREEN (0, 255, 0)
                # Radius 1, Thickness -1 (filled)
                cv2.circle(black_background, (x, y), 1, (0, 255, 0), -1)

    # 3. Process Hands
    hand_results = hands.process(frame_rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Define drawing specs for the Hand
            # Points: Red (BGR: 0, 0, 255)
            # Edges: Red (BGR: 0, 0, 255) - Note: Black edges would be invisible on black background
            
            # Draw landmarks (Points)
            landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            # Draw connections (Edges)
            connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

            mp_drawing.draw_landmarks(
                image=black_background,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

    # 4. Display the two windows
    cv2.imshow('you', frame)
    cv2.imshow('mirror', black_background)

    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()