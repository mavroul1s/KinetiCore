import os, sys, cv2, mediapipe as mp

# Automatically find the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "images")

OUT = "monkey.PNG"             # shown when pointer OUTSIDE face square
IN  = "monkey-thinking.png"    # shown when pointer INSIDE face square
CAMW, CAMH = 1280, 720
IMG_H = 480

# ---------- load images ----------
def find_file(folder, name):
    if not os.path.isdir(folder): return None
    for f in os.listdir(folder):
        if f.lower()==name.lower(): return os.path.join(folder,f)
    return None

if not os.path.isdir(IMG_DIR):
    print("Error: Could not find the 'images' directory at:", IMG_DIR)
    sys.exit(1)
    
p_out, p_in = find_file(IMG_DIR, OUT), find_file(IMG_DIR, IN)
if not p_out or not p_in:
    print("Missing:", [n for n,p in [(OUT,p_out),(IN,p_in)] if p is None])
    sys.exit(1)

img_out = cv2.flip(cv2.imread(p_out), 1)
img_in = cv2.flip(cv2.imread(p_in), 1)

def resize_h(img, h=IMG_H): 
    return cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)

img_out, img_in = resize_h(img_out), resize_h(img_in)

# ---------- mediapipe + openCV ----------
mpf, mph = mp.solutions.face_detection, mp.solutions.hands
mpd = mp.solutions.drawing_utils
Ld = mpd.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4)  # white points
Cd = mpd.DrawingSpec(color=(0,0,255), thickness=2)                       # red connections

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)

fd = mpf.FaceDetection(model_selection=1, min_detection_confidence=0.5)
hd = mph.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", CAMW, CAMH)
cv2.resizeWindow("Image", 800, 600)

try:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face -> square
        fd_res = fd.process(rgb)
        sq = None
        if fd_res.detections:
            d = fd_res.detections[0].location_data.relative_bounding_box
            x, y, ww, hh = int(d.xmin*w), int(d.ymin*h), int(d.width*w), int(d.height*h)
            cx, cy = x + ww//2, y + hh//2
            side = max(ww, hh) // 2
            x1, y1, x2, y2 = max(0, cx-side), max(0, cy-side), min(w, cx+side), min(h, cy+side)
            sq = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # hands -> draw + fingertip
        hr = hd.process(rgb)
        tip = None
        if hr.multi_hand_landmarks:
            for hl in hr.multi_hand_landmarks:
                mpd.draw_landmarks(frame, hl, mph.HAND_CONNECTIONS, landmark_drawing_spec=Ld, connection_drawing_spec=Cd)
                lm = hl.landmark[mph.HandLandmark.INDEX_FINGER_TIP]
                tip = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, tip, 8, (0, 0, 255), -1)
                break

        inside = False
        if sq and tip:
            x1, y1, x2, y2 = sq
            tx, ty = tip
            inside = (x1 <= tx <= x2 and y1 <= ty <= y2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Image", img_in if inside else img_out)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'): 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    fd.close()
    hd.close()