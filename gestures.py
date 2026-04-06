import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

def finger_is_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y

def is_fist(lm):
    finger_tips = [8, 12, 16, 20]
    finger_base = [6, 10, 14, 18]
    return all(lm[tip].y > lm[base].y for tip, base in zip(finger_tips, finger_base))

def is_open_palm(lm):
    finger_tips = [8, 12, 16, 20]
    finger_base = [6, 10, 14, 18]
    return all(lm[tip].y < lm[base].y for tip, base in zip(finger_tips, finger_base))

def is_pointing(lm):
    return finger_is_up(lm, 8, 6) and all(lm[tip].y > lm[base].y for tip, base in zip([12, 16, 20], [10, 14, 18]))

def is_call_me(lm):
    return finger_is_up(lm, 4, 3) and finger_is_up(lm, 20, 18) and \
           all(lm[tip].y > lm[base].y for tip, base in zip([8, 12, 16], [6, 10, 14]))

def is_ok_sign(lm):
    dx = lm[4].x - lm[8].x
    dy = lm[4].y - lm[8].y
    dist = math.sqrt(dx*dx + dy*dy)
    return dist < 0.05

def is_rock_sign(lm):
    return (finger_is_up(lm, 8, 6) and
            finger_is_up(lm, 20, 18) and
            not finger_is_up(lm, 12, 10) and
            not finger_is_up(lm, 16, 14))

def draw_message_box(frame, text, x, y):
    font_scale = 0.8
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    box_x1 = x
    box_y1 = y - text_h - 10
    box_x2 = x + text_w + 10
    box_y2 = y

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)
    cv2.putText(frame, text, (x + 5, y - 5), font, font_scale, (0, 255, 255), thickness)

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection and label
        face_results = face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
                cv2.putText(frame, "N", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Hand detection and gesture recognition
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                xs = [int(p.x * frame.shape[1]) for p in lm]
                ys = [int(p.y * frame.shape[0]) for p in lm]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                box_x = min_x
                box_y = min_y - 20

                #if is_fist(lm):
                 #   draw_message_box(frame, "execute", box_x, box_y)
                #elif is_open_palm(lm):
                 #   draw_message_box(frame, "open palm", box_x, box_y)
                if is_pointing(lm):
                    draw_message_box(frame, "execute", box_x, box_y)
                elif is_call_me(lm):
                    draw_message_box(frame, "call me", box_x, box_y)
                elif is_ok_sign(lm):
                    draw_message_box(frame, "ok", box_x, box_y)
                elif is_rock_sign(lm):
                    draw_message_box(frame, "yeah", box_x, box_y)

        cv2.imshow("Hand & Face Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
