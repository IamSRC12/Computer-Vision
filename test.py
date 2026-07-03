import cv2
import mediapipe as mp
import time
import math
from ctypes import windll

screen_w, screen_h = 1920, 1080
cam_w, cam_h = 640, 480

PINCH_THRESHOLD = 0.08
DETECT_EVERY = 4

MODEL_PATH = "hand_landmarker.task"

cursor_x, cursor_y = screen_w // 2, screen_h // 2
smoothing_factor = 0.7
frame_idx = 0
last_landmarks = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)

def map_to_screen(x, y):
    return int(x * screen_w), int(y * screen_h)

def set_mouse_pos(x, y):
    windll.user32.SetCursorPos(x, y)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up resizable preview window with premium default size
cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Test", 960, 720)

print("Starting...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    now = time.time()
    frame_idx += 1

    if frame_idx % DETECT_EVERY == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect_for_video(mp_image, int(now * 1000))

        if detection_result.hand_landmarks:
            last_landmarks = detection_result.hand_landmarks[0]
        elif frame_idx > 20:
            last_landmarks = None

    if last_landmarks is not None:
        lm = last_landmarks
        target_x, target_y = map_to_screen(lm[8].x, lm[8].y)
        cursor_x = int(cursor_x * (1 - smoothing_factor) + target_x * smoothing_factor)
        cursor_y = int(cursor_y * (1 - smoothing_factor) + target_y * smoothing_factor)
        set_mouse_pos(cursor_x, cursor_y)

        dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        status = "PINCH" if dist < PINCH_THRESHOLD else "OK"
    else:
        status = "NO HAND"

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done")