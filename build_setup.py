import os
import shutil
import sys

# Define base paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(current_dir, "dist")
latest_dir = os.path.join(dist_dir, "Latest version", "v9.0.0")
previous_dir = os.path.join(dist_dir, "Previous versions", "v8.0.0")

# Create directories
os.makedirs(latest_dir, exist_ok=True)
os.makedirs(previous_dir, exist_ok=True)

print("="*60)
print("AetherFlow Gesture Control - Distribution Builder (Console Edition)")
print("="*60)

# Step 1: Backup original v8.0.0 gesture_mouse.py to Previous versions/v8.0.0/
v8_content = """import cv2
import mediapipe as mp
import pyautogui
import time
import math
import os
from ctypes import windll

pyautogui.FAILSAFE = True

screen_w, screen_h = pyautogui.size()
cam_w, cam_h = 640, 480

ZONE_X = 0.05
ZONE_Y = 0.05
ZONE_W = 0.90
ZONE_H = 0.90

PINCH_THRESHOLD = 0.25  # Scale-invariant threshold (distance / hand scale)
THREE_FINGER_HOLD_TIME = 0.6
DETECT_EVERY = 2

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

is_dragging = False
was_index_pinching = False
was_middle_pinching = False
index_pinch_start_time = 0
three_finger_start_time = 0
three_finger_triggered = False
left_click_cooldown = 0
right_click_cooldown = 0

cursor_x, cursor_y = float(screen_w // 2), float(screen_h // 2)
smoothing_factor = 0.6  # Comfortably smooth for relative movement
prev_hand_x = None
prev_hand_y = None
SENSITIVITY = 2.0  # Trackpad-like speed factor

frame_idx = 0
last_landmarks = None
last_draw_time = 0
last_fps_time = time.time()
fps = 0
last_timestamp_ms = 0

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

def is_finger_extended(lm, tip, pip):
    tip_dist = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
    pip_dist = math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)
    return tip_dist > pip_dist

def count_extended(lm):
    c = 0
    if is_finger_extended(lm, 8, 6): c += 1
    if is_finger_extended(lm, 12, 10): c += 1
    if is_finger_extended(lm, 16, 14): c += 1
    if is_finger_extended(lm, 20, 18): c += 1
    return c

def map_to_screen(x, y):
    x = (x - ZONE_X) / ZONE_W
    y = (y - ZONE_Y) / ZONE_H
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return int(x * screen_w), int(y * screen_h)

def set_mouse_pos(x, y):
    windll.user32.SetCursorPos(x, y)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture", 250, 250)
cv2.moveWindow("Gesture", 10, 10)

print("Gesture Mouse v8 - Ultra Low Latency")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    gesture_text = "NO HAND"
    gesture_color = (0, 0, 255)
    now = time.time()

    frame_idx += 1
    should_detect = (frame_idx % DETECT_EVERY == 0)

    if should_detect:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        current_ms = int(now * 1000)
        if current_ms <= last_timestamp_ms:
            current_ms = last_timestamp_ms + 1
        last_timestamp_ms = current_ms
        
        detection_result = detector.detect_for_video(mp_image, current_ms)

        if detection_result.hand_landmarks:
            last_landmarks = detection_result.hand_landmarks[0]
        elif frame_idx > 10:
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
            last_landmarks = None

    if last_landmarks is not None:
        lm = last_landmarks
        hand_scale = max(0.0001, math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y))

        index_thumb_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / hand_scale
        thumb_middle_dist = math.hypot(lm[4].x - lm[12].x, lm[4].y - lm[12].y) / hand_scale

        is_index_pinching = (index_thumb_dist < PINCH_THRESHOLD) and (index_thumb_dist < thumb_middle_dist * 0.75)
        is_middle_pinching = (thumb_middle_dist < PINCH_THRESHOLD) and (thumb_middle_dist < index_thumb_dist * 0.75)

        extended_count = count_extended(lm)
        is_three_finger = extended_count >= 3
        is_fist = (extended_count == 0) and not is_index_pinching and not is_middle_pinching

        if is_fist:
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
            was_index_pinching = False
            was_middle_pinching = False
            prev_hand_x = None
            prev_hand_y = None
            gesture_text = "FIST"
            gesture_color = (0, 0, 255)
        else:
            is_index_extended = is_finger_extended(lm, 8, 6)
            should_move_cursor = is_index_extended or is_index_pinching or is_dragging

            if should_move_cursor:
                if is_index_pinching:
                    track_x = (lm[4].x + lm[8].x) / 2
                    track_y = (lm[4].y + lm[8].y) / 2
                else:
                    track_x = lm[8].x
                    track_y = lm[8].y

                if prev_hand_x is None or prev_hand_y is None:
                    prev_hand_x = track_x
                    prev_hand_y = track_y
                else:
                    dx = track_x - prev_hand_x
                    dy = track_y - prev_hand_y

                    target_cursor_x = cursor_x + dx * SENSITIVITY * screen_w
                    target_cursor_y = cursor_y + dy * SENSITIVITY * screen_h

                    cursor_x = cursor_x * (1 - smoothing_factor) + target_cursor_x * smoothing_factor
                    cursor_y = cursor_y * (1 - smoothing_factor) + target_cursor_y * smoothing_factor

                    cursor_x = max(0.0, min(float(screen_w), cursor_x))
                    cursor_y = max(0.0, min(float(screen_h), cursor_y))

                    set_mouse_pos(int(cursor_x), int(cursor_y))
                    
                    prev_hand_x = track_x
                    prev_hand_y = track_y
            else:
                prev_hand_x = None
                prev_hand_y = None

            if is_index_pinching:
                if not was_index_pinching:
                    index_pinch_start_time = now
                    was_index_pinching = True

                hold_duration = now - index_pinch_start_time
                if hold_duration >= 0.25 and not is_dragging:
                    is_dragging = True
                    pyautogui.mouseDown(button='left')
                    gesture_text = "DRAG"
                    gesture_color = (0, 165, 255)
                elif is_dragging:
                    gesture_text = "DRAGGING"
                    gesture_color = (0, 165, 255)
                else:
                    gesture_text = "PINCH"
                    gesture_color = (0, 255, 0)
            else:
                if was_index_pinching:
                    if is_dragging:
                        pyautogui.mouseUp(button='left')
                        is_dragging = False
                        gesture_text = "DROP"
                        gesture_color = (0, 255, 0)
                    elif (now - left_click_cooldown) > 0.3:
                        pyautogui.click(button='left')
                        left_click_cooldown = now
                        gesture_text = "LEFT"
                        gesture_color = (0, 255, 0)
                    was_index_pinching = False

            if is_middle_pinching:
                if not was_middle_pinching and (now - right_click_cooldown) > 0.3:
                    pyautogui.click(button='right')
                    right_click_cooldown = now
                    gesture_text = "RIGHT"
                    gesture_color = (0, 255, 0)
                was_middle_pinching = True
            else:
                was_middle_pinching = False

            if is_three_finger and not is_index_pinching and not is_middle_pinching:
                if three_finger_start_time == 0:
                    three_finger_start_time = now
                    three_finger_triggered = False

                hold_duration = now - three_finger_start_time
                if hold_duration >= THREE_FINGER_HOLD_TIME and not three_finger_triggered:
                    pyautogui.hotkey('win', 'tab')
                    gesture_text = "TASK VIEW"
                    gesture_color = (255, 0, 255)
                    three_finger_triggered = True
                elif not three_finger_triggered:
                    gesture_text = f"HOLD {max(0, int(THREE_FINGER_HOLD_TIME - hold_duration) + 1)}"
                    gesture_color = (255, 255, 0)
            else:
                three_finger_start_time = 0

        if gesture_text == "NO HAND":
            gesture_text = "READY"
    else:
        if is_dragging:
            pyautogui.mouseUp(button='left')
            is_dragging = False
        was_index_pinching = False
        was_middle_pinching = False
        prev_hand_x = None
        prev_hand_y = None

    if now - last_fps_time >= 1.0:
        fps = frame_idx
        frame_idx = 0
        last_fps_time = now

    if now - last_draw_time > 0.05:
        h_crop, w_crop, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w_crop - 1, h_crop - 1), (50, 50, 50), 2)
        cv2.putText(frame, f"{fps}fps {gesture_text}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 1)
        if is_dragging:
            cv2.putText(frame, "DRAG", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        last_draw_time = now

    cv2.imshow("Gesture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("+"):
        PINCH_THRESHOLD = min(0.50, PINCH_THRESHOLD + 0.02)
    elif key == ord("-"):
        PINCH_THRESHOLD = max(0.10, PINCH_THRESHOLD - 0.02)

if is_dragging:
    pyautogui.mouseUp(button='left')

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
"""

v8_path = os.path.join(previous_dir, "gesture_mouse.py")
with open(v8_path, "w") as f:
    f.write(v8_content)
print(f"[Backup] Restored v8 source script to: {v8_path}")

# Copy the v8 model file too if it exists
model_source = os.path.join(current_dir, "hand_landmarker.task")
if os.path.exists(model_source):
    shutil.copy2(model_source, os.path.join(previous_dir, "hand_landmarker.task"))
    print("[Backup] Copied v8 model to previous versions folder.")

# Step 2: Distribute the new v9.0.0 python script and launcher batch file
print("\n[Distribution] Copying optimized console version script and assets...")
shutil.copy2(os.path.join(current_dir, "gesture_mouse.py"), os.path.join(latest_dir, "gesture_mouse.py"))
print(f"[Distribution] Script copied to: {os.path.join(latest_dir, 'gesture_mouse.py')}")

if os.path.exists(model_source):
    shutil.copy2(model_source, os.path.join(latest_dir, "hand_landmarker.task"))
    print(f"[Distribution] Model task copied to: {os.path.join(latest_dir, 'hand_landmarker.task')}")

shutil.copy2(os.path.join(current_dir, "start.bat"), os.path.join(latest_dir, "start.bat"))
print(f"[Distribution] Launcher bat copied to: {os.path.join(latest_dir, 'start.bat')}")

print("\n" + "="*60)
print("Distribution package v9.0.0 built successfully (Console Edition)!")
print("="*60)
