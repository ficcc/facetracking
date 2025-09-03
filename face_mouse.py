
import cv2
import dlib
import pyautogui as pag
import numpy as np
import time
from imutils import face_utils

# --- CONFIGURATION --- #
WINK_HOLD_DURATION = 1.5     # Seconds to hold a wink to trigger a click.
MOUSE_SENSITIVITY = 0.5      # Adjust mouse speed.
SCROLL_SENSITIVITY = 0.5      # Adjust scroll speed.
DEAD_ZONE_RADIUS = 30          # Prevents jitter.

# --- GLOBAL VARIABLES & INITIALIZATION --- #
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error: {e}\nPlease ensure 'shape_predictor_68_face_landmarks.dat' is in the same folder.")
    exit()

(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- HELPER FUNCTIONS --- #
def calculate_ear(eye):
    v1 = np.linalg.norm(eye[1] - eye[5]); v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

def calculate_mar(mouth):
    v = np.linalg.norm(mouth[14] - mouth[18]); h = np.linalg.norm(mouth[12] - mouth[16])
    return v / h if h > 0 else 0.0

def display_text(frame, text, y_pos=30, x_pos=10, color=(255, 255, 255)):
    cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- NEW TIMED CALIBRATION --- #
def calibrate():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return None
    
    cal_data = {}
    gesture_data = {}
    steps = [
        ("NEUTRAL_FACE", "Hold a neutral face"),
        ("LEFT_WINK", "Hold a clear LEFT WINK"),
        ("RIGHT_WINK", "Hold a clear RIGHT WINK"),
        ("MOUTH_OPEN", "Hold your mouth open for scrolling")
    ]
    
    for key, instruction in steps:
        start_time = time.time()
        readings = []
        
        while time.time() - start_time < 2.5: # Capture for 2.5 seconds
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            display_text(frame, f"{instruction}. Capturing...", y_pos=50, x_pos=150)
            
            progress = (time.time() - start_time) / 2.5
            cv2.rectangle(frame, (150, 80), (frame.shape[1] - 150, 100), (255, 255, 255), 2)
            cv2.rectangle(frame, (150, 80), (150 + int(progress * (frame.shape[1] - 300)), 100), (0, 255, 0), -1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if len(rects) > 0:
                shape = face_utils.shape_to_np(predictor(gray, rects[0]))
                readings.append({
                    'left_ear': calculate_ear(shape[L_START:L_END]),
                    'right_ear': calculate_ear(shape[R_START:R_END]),
                    'mar': calculate_mar(shape[48:68]),
                    'nose': (shape[30][0], shape[30][1])
                })
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)
            
        # Analyze captured data
        if key == "NEUTRAL_FACE":
            cal_data['neutral_nose'] = np.mean([r['nose'] for r in readings], axis=0).astype(int)
            gesture_data['NEUTRAL'] = {'left': np.mean([r['left_ear'] for r in readings]), 'right': np.mean([r['right_ear'] for r in readings])}
        elif key == "LEFT_WINK":
            gesture_data['LEFT_WINK'] = {'left': np.mean([r['left_ear'] for r in readings]), 'right': np.mean([r['right_ear'] for r in readings])}
        elif key == "RIGHT_WINK":
            gesture_data['RIGHT_WINK'] = {'left': np.mean([r['left_ear'] for r in readings]), 'right': np.mean([r['right_ear'] for r in readings])}
        elif key == "MOUTH_OPEN":
            cal_data['mar_scroll'] = np.mean([r['mar'] for r in readings]) * 0.8
        
        print(f"Captured {key}.")

    cap.release()
    cv2.destroyAllWindows()

    # Calculate final, robust thresholds
    cal_data['ear_closed'] = max(gesture_data['LEFT_WINK']['left'], gesture_data['RIGHT_WINK']['right']) + 0.04
    cal_data['left_wink_diff'] = (gesture_data['LEFT_WINK']['right'] - gesture_data['LEFT_WINK']['left']) * 0.7
    cal_data['right_wink_diff'] = (gesture_data['RIGHT_WINK']['left'] - gesture_data['RIGHT_WINK']['right']) * 0.7
    
    print("\n--- Calibration Complete! ---")
    return cal_data

# --- MAIN APPLICATION LOGIC --- #
def main():
    cal_data = calibrate()
    if not cal_data: return

    neutral_nose = np.array(cal_data['neutral_nose'])
    is_paused = True
    scroll_mode = False
    
    wink_start_time = {'left': 0, 'right': 0}
    click_fired = {'left': False, 'right': False}
    scroll_fired = False

    cap = cv2.VideoCapture(0)
    pag.FAILSAFE, pag.PAUSE = False, 0
    
    print("\n--- Control is Ready ---\nPAUSED. Press SPACEBAR to activate. Press 'r' to re-center.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): is_paused = not is_paused
        
        # --- FIX: Changed COLOR_BGR_GRAY to COLOR_BGR2GRAY ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        if len(rects) > 0:
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            nose = np.array((shape[30][0], shape[30][1]))
            left_ear = calculate_ear(shape[L_START:L_END])
            right_ear = calculate_ear(shape[R_START:R_END])

            # --- VISUALIZERS ---
            for (x, y) in shape: cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            cv2.circle(frame, tuple(neutral_nose.astype(int)), DEAD_ZONE_RADIUS, (255, 0, 0), 1)
            cv2.line(frame, tuple(neutral_nose.astype(int)), tuple(nose.astype(int)), (0, 255, 255), 2)
            cv2.circle(frame, tuple(nose.astype(int)), 5, (0, 0, 255), -1)

            # --- ON-SCREEN DEBUGGER ---
            debug_x = frame.shape[1] - 220
            display_text(frame, "--- LIVE DATA ---", y_pos=20, x_pos=debug_x, color=(0, 255, 255))
            display_text(frame, f"L EAR: {left_ear:.2f}", y_pos=50, x_pos=debug_x)
            display_text(frame, f"R EAR: {right_ear:.2f}", y_pos=70, x_pos=debug_x)
            display_text(frame, f"L-R Diff: {left_ear - right_ear:.2f}", y_pos=90, x_pos=debug_x)
            display_text(frame, "--- TARGETS ---", y_pos=120, x_pos=debug_x, color=(0, 255, 255))
            display_text(frame, f"Closed < {cal_data['ear_closed']:.2f}", y_pos=150, x_pos=debug_x)
            display_text(frame, f"L Wink Diff > {cal_data['left_wink_diff']:.2f}", y_pos=170, x_pos=debug_x)
            display_text(frame, f"R Wink Diff > {cal_data['right_wink_diff']:.2f}", y_pos=190, x_pos=debug_x)

            if not is_paused:
                if key == ord('r'): neutral_nose = nose; print("--- Re-centered ---")

                mar = calculate_mar(shape[48:68])

                # SCROLL MODE
                if mar > cal_data['mar_scroll']:
                    if not scroll_fired: scroll_mode = not scroll_mode; scroll_fired = True
                else:
                    scroll_fired = False
                
                if scroll_mode:
                    display_text(frame, "SCROLL MODE", y_pos=50, color=(0, 255, 0))
                    move_vector = nose - neutral_nose
                    if abs(move_vector[1]) > DEAD_ZONE_RADIUS: pag.scroll(int(-move_vector[1] * SCROLL_SENSITIVITY))
                else:
                    # MOUSE MOVEMENT
                    display_text(frame, "ACTIVE", y_pos=50, color=(0, 255, 0))
                    move_vector = nose - neutral_nose
                    if np.linalg.norm(move_vector) > DEAD_ZONE_RADIUS: pag.move(move_vector[0] * MOUSE_SENSITIVITY, move_vector[1] * MOUSE_SENSITIVITY)

                # WINK DETECTION
                is_left_wink = left_ear < cal_data['ear_closed'] and (right_ear - left_ear) > cal_data['left_wink_diff']
                if is_left_wink:
                    if wink_start_time['left'] == 0: wink_start_time['left'] = time.time()
                    wink_progress = (time.time() - wink_start_time['left']) / WINK_HOLD_DURATION
                    cv2.putText(frame, f"L-CLICK: {int(wink_progress*100)}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if wink_progress >= 1.0 and not click_fired['left']:
                        pag.click(button='left'); print("Left Click"); click_fired['left'] = True
                else:
                    wink_start_time['left'] = 0; click_fired['left'] = False

                is_right_wink = right_ear < cal_data['ear_closed'] and (left_ear - right_ear) > cal_data['right_wink_diff']
                if is_right_wink:
                    if wink_start_time['right'] == 0: wink_start_time['right'] = time.time()
                    wink_progress = (time.time() - wink_start_time['right']) / WINK_HOLD_DURATION
                    cv2.putText(frame, f"R-CLICK: {int(wink_progress*100)}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if wink_progress >= 1.0 and not click_fired['right']:
                        pag.click(button='right'); print("Right Click"); click_fired['right'] = True
                else:
                    wink_start_time['right'] = 0; click_fired['right'] = False
        
        cv2.imshow("Face Mouse Control", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()