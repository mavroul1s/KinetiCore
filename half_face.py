import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands and Face Mesh
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Setup Hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Setup Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB for MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process results
        hands_results = hands.process(image_rgb)
        face_results = face_mesh.process(image_rgb)

        # Draw annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape

        # 1. Draw Hand Landmarks (Points and Connections)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Overwrite with custom styles: Black points, Black connections
                # We redraw to ensure the colors are exactly as requested
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),       # Points: Black
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)        # Connections: Black
                )

        # 2. Draw Face Mesh (Left Half Only)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # We need a reference point to determine "left" vs "right".
                # Using the nose tip (index 1) is a good central reference.
                nose_tip = face_landmarks.landmark[1]
                nose_x = nose_tip.x

                # Draw Connections first (so points are on top)
                for connection in mp_face_mesh.FACEMESH_TESSELATION:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = face_landmarks.landmark[start_idx]
                    end_point = face_landmarks.landmark[end_idx]
                    
                    # Check if BOTH points are on the left side
                    if start_point.x < nose_x and end_point.x < nose_x:
                        start_px = (int(start_point.x * w), int(start_point.y * h))
                        end_px = (int(end_point.x * w), int(end_point.y * h))
                        cv2.line(image, start_px, end_px, (200, 200, 200), 1) # Light gray connections

                # Draw Points
                for idx, landmark in enumerate(face_landmarks.landmark):
                    # In the flipped image (mirror view):
                    # Left side of screen = User's Left side
                    # So we want points where x < nose_x
                    
                    if landmark.x < nose_x:
                        # Convert normalized coordinates to pixel coordinates
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
                        # Draw a small circle for the point
                        # Using a distinct color (e.g., Green)
                        cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)

        # Show the image
        cv2.imshow('Hand and Partial Face Tracking', image)

        # Exit on 'q' or 'ESC'
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
