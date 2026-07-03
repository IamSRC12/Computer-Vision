import cv2
import mediapipe as mp
import time
import math
import os
import ctypes
import threading
from queue import Queue
import msvcrt

# Ensure high DPI awareness on Windows to prevent coordinate scaling mismatches
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # 2 = PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Constants for Windows API mouse and keyboard input emulation
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
KEYEVENTF_KEYUP = 0x0002

# Get actual physical monitor resolution
screen_w = ctypes.windll.user32.GetSystemMetrics(0)
screen_h = ctypes.windll.user32.GetSystemMetrics(1)

# Hand landmarker configuration path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Hand connections mapping for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Knuckles
]

class AppState:
    """Thread-safe state manager for application settings and statistics."""
    def __init__(self):
        self.lock = threading.Lock()
        
        # User Configurable Settings (Dynamic via keyboard)
        self.sensitivity = 2.5
        self.responsiveness = 0.50 # 0.05 (heavy smooth) to 0.95 (raw tracker)
        self.pinch_threshold = 0.25
        self.min_confidence = 0.70
        self.acceleration_enabled = True
        self.mouse_control_enabled = True
        
        # Running state
        self.running = True
        
        # Performance metrics
        self.fps_camera = 0.0
        self.fps_inference = 0.0
        
        # Active gesture indicators
        self.current_gesture = "NO HAND"
        self.is_dragging = False
        
        # Cursor positioning state
        self.cursor_x = float(screen_w // 2)
        self.cursor_y = float(screen_h // 2)
        self.prev_hand_x = None
        self.prev_hand_y = None
        
        # Gesture timing states
        self.was_index_pinching = False
        self.was_middle_pinching = False
        self.index_pinch_start_time = 0.0
        self.three_finger_start_time = 0.0
        self.three_finger_triggered = False
        self.left_click_cooldown = 0.0
        self.right_click_cooldown = 0.0


class CameraStream:
    """Asynchronously captures frames from the webcam in a dedicated thread to eliminate latency."""
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        
        # Camera FPS measurement
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.measured_fps = 30.0
        
        self.thread = threading.Thread(target=self._update, name="CameraStreamThread")
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Mirror the frame at capture stage for natural mirrored coordinate tracking
                    frame = cv2.flip(frame, 1)
                    now = time.time()
                    self.frame_count += 1
                    if now - self.last_fps_time >= 1.0:
                        self.measured_fps = self.frame_count / (now - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = now
                        
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                        self.frame_id += 1
                else:
                    time.sleep(0.002)
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy(), self.measured_fps, self.frame_id
            return False, None, 0.0, 0

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


class InferenceThread:
    """Handles MediaPipe Hand Landmarker execution and runs mouse control synchronously in background."""
    def __init__(self, app_state):
        self.state = app_state
        self.running = True
        self.frame_queue = Queue(maxsize=1)
        self.latest_result = None
        self.lock = threading.Lock()
        
        # Inference FPS measurement
        self.inference_count = 0
        self.last_fps_time = time.time()
        self.measured_fps = 0.0
        
        self.thread = threading.Thread(target=self._run, name="InferenceThread")
        self.thread.daemon = True
        self.thread.start()

    def submit_frame(self, frame):
        # Discard older frames if queue is full to prevent lagging pipeline
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass
        self.frame_queue.put(frame)

    def get_result(self):
        with self.lock:
            return self.latest_result, self.measured_fps

    def _run(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Initialize detector
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.state.min_confidence,
            min_tracking_confidence=0.5,
        )
        try:
            detector = HandLandmarker.create_from_options(options)
            print("[INFO] MediaPipe Hand Landmarker loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load MediaPipe model: {e}")
            self.running = False
            return
            
        last_timestamp_ms = 0
        
        while self.running and self.state.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except Exception:
                continue
                
            # Downscale input to 256x256 before running AI inference
            h, w, _ = frame.shape
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
            resized = cv2.resize(cropped, (256, 256))
            
            frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            now = time.time()
            current_ms = int(now * 1000)
            if current_ms <= last_timestamp_ms:
                current_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_ms
            
            try:
                result = detector.detect_for_video(mp_image, current_ms)
                
                # Compute Inference FPS
                self.inference_count += 1
                if now - self.last_fps_time >= 1.0:
                    self.measured_fps = self.inference_count / (now - self.last_fps_time)
                    self.inference_count = 0
                    self.last_fps_time = now
                    
                with self.lock:
                    self.latest_result = result
                
                # EXECUTE MOUSE MOVEMENT IMMEDIATELY inside the background thread!
                # This bypasses the main display thread completely and cuts latency to near zero.
                landmarks = result.hand_landmarks[0] if result and result.hand_landmarks else None
                process_gestures_and_mouse(landmarks, self.state)
            except Exception:
                pass
                
        detector.close()


def draw_skeleton(img, landmarks, current_gesture):
    h, w, _ = img.shape
    # Joint Connections
    for conn in HAND_CONNECTIONS:
        start_idx, end_idx = conn
        pt1 = landmarks[start_idx]
        pt2 = landmarks[end_idx]
        # Frame is already mirrored at capture stage
        p1_x = int(pt1.x * w)
        p1_y = int(pt1.y * h)
        p2_x = int(pt2.x * w)
        p2_y = int(pt2.y * h)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), (140, 140, 140), 2)
        
    # Draw Nodes
    for idx, pt in enumerate(landmarks):
        p_x = int(pt.x * w)
        p_y = int(pt.y * h)
        if idx in [0, 4, 8, 12, 16, 20]:
            color = (255, 240, 0) if current_gesture in ["PINCH", "DRAG", "DRAGGING"] else (255, 0, 0)
            cv2.circle(img, (p_x, p_y), 6, color, -1)
        else:
            cv2.circle(img, (p_x, p_y), 3, (255, 255, 255), -1)


def process_gestures_and_mouse(lm, state):
    now = time.time()
    if lm is None:
        if state.is_dragging:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            state.is_dragging = False
        state.was_index_pinching = False
        state.was_middle_pinching = False
        state.prev_hand_x = None
        state.prev_hand_y = None
        state.current_gesture = "NO HAND"
        return

    # Calculate hand size scale
    hand_scale = max(0.0001, math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y))

    # Check extended fingers count
    def is_extended(tip, pip):
        return math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y) > math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)

    extended_count = 0
    if is_extended(8, 6): extended_count += 1
    if is_extended(12, 10): extended_count += 1
    if is_extended(16, 14): extended_count += 1
    if is_extended(20, 18): extended_count += 1

    # Scale-invariant pinch calculations
    index_thumb_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / hand_scale
    thumb_middle_dist = math.hypot(lm[4].x - lm[12].x, lm[4].y - lm[12].y) / hand_scale

    is_index_pinching = (index_thumb_dist < state.pinch_threshold) and (index_thumb_dist < thumb_middle_dist * 0.75)
    is_middle_pinching = (thumb_middle_dist < state.pinch_threshold) and (thumb_middle_dist < index_thumb_dist * 0.75)
    is_three_finger = extended_count >= 3
    is_fist = (extended_count == 0) and not is_index_pinching and not is_middle_pinching

    gesture_name = "READY"

    if is_fist:
        gesture_name = "FIST"
        if state.is_dragging:
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            state.is_dragging = False
        state.was_index_pinching = False
        state.was_middle_pinching = False
        state.prev_hand_x = None
        state.prev_hand_y = None
    else:
        # 1. Cursor movement
        is_index_extended = is_extended(8, 6)
        should_move = is_index_extended or is_index_pinching or state.is_dragging

        if should_move and state.mouse_control_enabled:
            if is_index_pinching:
                track_x = (lm[4].x + lm[8].x) / 2
                track_y = (lm[4].y + lm[8].y) / 2
            else:
                track_x = lm[8].x
                track_y = lm[8].y

            if state.prev_hand_x is None or state.prev_hand_y is None:
                state.prev_hand_x = track_x
                state.prev_hand_y = track_y
            else:
                dx = track_x - state.prev_hand_x
                dy = track_y - state.prev_hand_y

                velocity = math.hypot(dx, dy)
                
                if state.acceleration_enabled:
                    accel_scale = state.sensitivity * (1.0 + (velocity * 120.0) ** 1.3)
                    adaptive_responsiveness = min(0.95, state.responsiveness + velocity * 8.0)
                else:
                    accel_scale = state.sensitivity
                    adaptive_responsiveness = state.responsiveness

                target_cursor_x = state.cursor_x + dx * accel_scale * screen_w
                target_cursor_y = state.cursor_y + dy * accel_scale * screen_h

                # Double exponential smoothing filter
                state.cursor_x = state.cursor_x * (1 - adaptive_responsiveness) + target_cursor_x * adaptive_responsiveness
                state.cursor_y = state.cursor_y * (1 - adaptive_responsiveness) + target_cursor_y * adaptive_responsiveness

                state.cursor_x = max(0.0, min(float(screen_w - 1), state.cursor_x))
                state.cursor_y = max(0.0, min(float(screen_h - 1), state.cursor_y))

                ctypes.windll.user32.SetCursorPos(int(state.cursor_x), int(state.cursor_y))

                state.prev_hand_x = track_x
                state.prev_hand_y = track_y
        else:
            state.prev_hand_x = None
            state.prev_hand_y = None

        # 2. Left Clicks & Left Drags
        if is_index_pinching:
            if not state.was_index_pinching:
                state.index_pinch_start_time = now
                state.was_index_pinching = True

            hold_time = now - state.index_pinch_start_time
            if hold_time >= 0.25 and not state.is_dragging:
                if state.mouse_control_enabled:
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                state.is_dragging = True
                gesture_name = "DRAG"
            elif state.is_dragging:
                gesture_name = "DRAGGING"
            else:
                gesture_name = "PINCH"
        else:
            if state.was_index_pinching:
                if state.is_dragging:
                    if state.mouse_control_enabled:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    state.is_dragging = False
                    gesture_name = "READY"
                elif (now - state.left_click_cooldown) > 0.25:
                    if state.mouse_control_enabled:
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        time.sleep(0.005)
                        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    state.left_click_cooldown = now
                    gesture_name = "PINCH"
                state.was_index_pinching = False

        # 3. Right Clicks
        if is_middle_pinching:
            if not state.was_middle_pinching and (now - state.right_click_cooldown) > 0.3:
                if state.mouse_control_enabled:
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    time.sleep(0.005)
                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                state.right_click_cooldown = now
                gesture_name = "RIGHT"
            state.was_middle_pinching = True
        else:
            state.was_middle_pinching = False

        # 4. Three-Finger Windows Task View
        if is_three_finger and not is_index_pinching and not is_middle_pinching:
            if state.three_finger_start_time == 0.0:
                state.three_finger_start_time = now
                state.three_finger_triggered = False

            hold_time = now - state.three_finger_start_time
            if hold_time >= 0.6 and not state.three_finger_triggered:
                if state.mouse_control_enabled:
                    ctypes.windll.user32.keybd_event(0x5B, 0, 0, 0)       # VK_LWIN Down
                    ctypes.windll.user32.keybd_event(0x09, 0, 0, 0)       # VK_TAB Down
                    ctypes.windll.user32.keybd_event(0x09, 0, KEYEVENTF_KEYUP, 0)
                    ctypes.windll.user32.keybd_event(0x5B, 0, KEYEVENTF_KEYUP, 0)
                state.three_finger_triggered = True
                gesture_name = "TASK VIEW"
            elif not state.three_finger_triggered:
                gesture_name = f"HOLD {max(0, int(0.6 - hold_time) + 1)}"
        else:
            state.three_finger_start_time = 0.0

    state.current_gesture = gesture_name


def handle_key(key_char, state):
    if key_char == 'q' or key_char == 'Q':
        state.running = False
        print("[INFO] Quitting application...")
    elif key_char == '+' or key_char == '=':
        with state.lock:
            state.sensitivity = min(5.0, state.sensitivity + 0.1)
        print(f"[CONFIG] Sensitivity set to: {state.sensitivity:.1f}x")
    elif key_char == '-' or key_char == '_':
        with state.lock:
            state.sensitivity = max(0.5, state.sensitivity - 0.1)
        print(f"[CONFIG] Sensitivity set to: {state.sensitivity:.1f}x")
    elif key_char == ']':
        with state.lock:
            state.pinch_threshold = min(0.45, state.pinch_threshold + 0.01)
        print(f"[CONFIG] Pinch threshold set to: {state.pinch_threshold:.2f}")
    elif key_char == '[':
        with state.lock:
            state.pinch_threshold = max(0.10, state.pinch_threshold - 0.01)
        print(f"[CONFIG] Pinch threshold set to: {state.pinch_threshold:.2f}")
    elif key_char == 'p' or key_char == 'P':
        with state.lock:
            state.mouse_control_enabled = not state.mouse_control_enabled
        print(f"[CONFIG] Mouse control: {'ENABLED' if state.mouse_control_enabled else 'DISABLED'}")
    elif key_char == '0' or key_char == ')':
        with state.lock:
            state.responsiveness = min(0.95, state.responsiveness + 0.05)
        print(f"[CONFIG] Responsiveness (speed) set to: {state.responsiveness:.2f}")
    elif key_char == '9' or key_char == '(':
        with state.lock:
            state.responsiveness = max(0.05, state.responsiveness - 0.05)
        print(f"[CONFIG] Responsiveness (speed) set to: {state.responsiveness:.2f}")


def main():
    print("=" * 60)
    print("  AetherFlow Gesture Control Mouse v9.0.0 - Console Edition")
    print("=" * 60)
    print("  Webcam frames and AI inference are fully multithreaded.")
    print("  Emulation is directly powered by zero-latency Win32 APIs.")
    print("  OpenCV feed acts as a premium HUD overlay.")
    print("\n  Controls (Active anywhere - Terminal or HUD focused):")
    print("    - Press '+' to increase sensitivity.")
    print("    - Press '-' to decrease sensitivity.")
    print("    - Press ']' to increase pinch click threshold.")
    print("    - Press '[' to decrease pinch click threshold.")
    print("    - Press '0' to increase speed responsiveness.")
    print("    - Press '9' to decrease speed responsiveness.")
    print("    - Press 'p' to pause / resume mouse pointer movement.")
    print("    - Press 'q' to quit.")
    print("=" * 60)

    state = AppState()
    camera = CameraStream()
    inference = InferenceThread(state)
    
    cv2.namedWindow("AetherFlow Gesture HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AetherFlow Gesture HUD", 380, 380)
    cv2.moveWindow("AetherFlow Gesture HUD", 15, 15)

    last_show_time = 0
    last_processed_frame_id = -1

    try:
        while state.running:
            success, frame, cam_fps, frame_id = camera.read()
            
            # Check terminal keyboard inputs via msvcrt (non-blocking)
            # This makes settings change instantly even when terminal window is focused!
            if msvcrt.kbhit():
                key_char = msvcrt.getch()
                try:
                    ch = key_char.decode('utf-8')
                    handle_key(ch, state)
                except Exception:
                    pass

            if not success or frame is None:
                time.sleep(0.002)
                continue
                
            state.fps_camera = cam_fps
            
            # If we already processed this frame, sleep 2ms to prevent pinning the CPU
            if frame_id == last_processed_frame_id:
                time.sleep(0.002)
                # Check OpenCV window keyboard inputs as well
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    try:
                        handle_key(chr(key), state)
                    except Exception:
                        pass
                continue
                
            last_processed_frame_id = frame_id
            
            # Submit frame for handlandmarker inference
            inference.submit_frame(frame)
            
            # Retrieve latest inference result
            result, inf_fps = inference.get_result()
            state.fps_inference = inf_fps
            
            landmarks = None
            if result and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
            
            # Refresh HUD display frame at a throttled rate (~30fps)
            now = time.time()
            if now - last_show_time > 0.033:
                h, w, _ = frame.shape
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                hud_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
                
                if landmarks:
                    draw_skeleton(hud_frame, landmarks, state.current_gesture)
                    
                # Create semi-transparent status bar at the top
                overlay = hud_frame.copy()
                cv2.rectangle(overlay, (0, 0), (min_dim, 65), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.55, hud_frame, 0.45, 0, hud_frame)
                
                # Draw dynamic stats on HUD
                gesture_text = f"GESTURE: {state.current_gesture}"
                fps_text = f"CAM FPS: {state.fps_camera:.1f} | INF FPS: {state.fps_inference:.1f}"
                config_text = f"SENS: {state.sensitivity:.1f}x | PINCH: {state.pinch_threshold:.2f} | SPEED: {state.responsiveness:.2f} | CTRL: {'ON' if state.mouse_control_enabled else 'OFF'}"
                
                cv2.putText(hud_frame, gesture_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 2)
                cv2.putText(hud_frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(hud_frame, config_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow("AetherFlow Gesture HUD", hud_frame)
                last_show_time = now
                
            # Check OpenCV window keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                try:
                    handle_key(chr(key), state)
                except Exception:
                    pass
                
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        camera.release()
        cv2.destroyAllWindows()
        print("\nEngines shut down. Clean exit.")

if __name__ == "__main__":
    main()