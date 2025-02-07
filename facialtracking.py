# Import necessary libraries
from imutils import face_utils  # For facial landmark detection utilities
import dlib  # For detecting facial landmarks
import cv2  # For computer vision tasks like image capture and processing
import pyautogui as pag  # For controlling mouse and keyboard input
import numpy as np  # For performing array operations
from matplotlib import pyplot as plt  # For plotting calibration data
import time  # For handling timing operations
from scipy.ndimage import gaussian_filter  # For smoothing the calibration data

# --- FUNCTION DEFINITIONS --- #

# Calculates Mouth Aspect Ratio (MAR) based on facial landmarks
def MAR(point1, point2, point3, point4, point5, point6, point7, point8):
    mar = (dst(point1, point2) + dst(point3, point4) + dst(point5, point6)) / (3.0 * dst(point7, point8))
    return mar

# Calculates Eye Aspect Ratio (EAR) for eye state detection (open or closed)
def EAR(point1, point2, point3, point4, point5, point6):
    ear = (dst(point2, point6) + dst(point3, point5)) / (2 * dst(point1, point4))
    return ear

# Calculates Euclidean distance between two points (used for landmark-based measurements)
def dst(point1, point2):
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

# Calculates the angle between the nose tip and the horizontal line (used for cursor movement)
def angle(point1):
    slope = (point1[1] - 250) / (point1[0] - 250)
    agle = np.arctan(slope)
    return agle

# Placeholder function for trackbar (OpenCV requirement)
def nothing(x):
    pass

# --- INITIALIZATION --- #

# Arrays for storing data during calibration (EAR, MAR values, etc.)
lclick, rclick, scroll = np.array([]), np.array([]), np.array([])
eyeopen, lclickarea, rclickarea = np.array([]), np.array([]), np.array([])
t1, t2, t3, t4 = np.array([]), np.array([]), np.array([]), np.array([])

# Disable pyautogui's delay between commands
pag.PAUSE = 0

# Load the facial landmark predictor and initialize face detector
p = "shape_predictor.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Facial landmark indices for eyes and mouth
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# --- CALIBRATION PHASE --- #

# Capture video feed from webcam
cap = cv2.VideoCapture(0)

# Font settings for text display on the screen
font = cv2.FONT_HERSHEY_SIMPLEX
currenttime = time.time()

# Calibration loop for 23 seconds
while time.time() - currenttime <= 25:
    ret, image = cap.read()
    blackimage = np.zeros((480, 640, 3), dtype=np.uint8)
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # Process each detected face
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Compute EAR and MAR for calibration
        lefteye = EAR(shape[36], shape[37], shape[38], shape[39], shape[40], shape[41])
        righteye = EAR(shape[42], shape[43], shape[44], shape[45], shape[46], shape[47])
        mar = MAR(shape[50], shape[58], shape[51], shape[57], shape[52], shape[56], shape[48], shape[54])
        EARdiff = (lefteye - righteye) * 100

        # Draw facial landmarks and contours on the face
        leftEyeHull = cv2.convexHull(shape[lstart:lend])
        rightEyeHull = cv2.convexHull(shape[rstart:rend])
        mouthHull = cv2.convexHull(shape[mstart:mend])
        
        cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 1)

        # Time-based calibration prompts
        elapsedTime = time.time() - currenttime

        if elapsedTime < 5.0:
            cv2.putText(blackimage, 'Keep Both Eyes Open', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            eyeopen = np.append(eyeopen, [EARdiff])
            t1 = np.append(t1, [elapsedTime])
        
        elif 5.0 < elapsedTime < 10.0:
            cv2.putText(blackimage, 'Close Left Eye', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if 7.0 < elapsedTime < 10.0:
                lclick = np.append(lclick, [EARdiff])
                lclickarea = np.append(lclickarea, [cv2.contourArea(leftEyeHull)])
                t2 = np.append(t2, [elapsedTime])
        
        elif 12.0 < elapsedTime < 17.0:
            cv2.putText(blackimage, 'Open Left Eye and Close Right Eye', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if 14.0 < elapsedTime < 17.0:
                rclick = np.append(rclick, [EARdiff])
                rclickarea = np.append(rclickarea, [cv2.contourArea(rightEyeHull)])
                t3 = np.append(t3, [elapsedTime])
        
        elif 19.0 < elapsedTime < 24.0:
            cv2.putText(blackimage, 'Open Your Mouth', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if 21.0 < elapsedTime < 24.0:
                scroll = np.append(scroll, [mar])
                t4 = np.append(t4, [elapsedTime])
        
        # Draw facial landmarks on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Stack images and show the calibration phase
        res = np.vstack((image, blackimage))
        cv2.imshow('Calibration', res)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- PLOTTING THE CALIBRATION DATA --- #

# Smooth and plot the recorded calibration data
plt.subplot(2, 2, 1)
plt.title('Both Eyes Open')
plt.plot(t1, gaussian_filter(eyeopen, sigma=5))

plt.subplot(2, 2, 2)
plt.title('Left Click')
plt.plot(t2, gaussian_filter(lclick, sigma=5))

plt.subplot(2, 2, 3)
plt.title('Right Click')
plt.plot(t3, gaussian_filter(rclick, sigma=5))

plt.subplot(2, 2, 4)
plt.title('Scroll Mode')
plt.plot(t4, gaussian_filter(scroll, sigma=5))

plt.show()

# Release resources
cap.release()
cv2.destroyAllWindows()

# --- CURSOR CONTROL PHASE --- #

# Capture video feed for gesture control
cap = cv2.VideoCapture(0)
MARlist = np.array([])  # List for storing MAR values
scroll_status = 0  # Track scrolling state

# Sort and calculate median for calibration values
eyeopen = np.sort(eyeopen)
lclick = np.sort(lclick)
rclick = np.sort(rclick)
scroll = np.sort(scroll)
lclickarea = np.sort(lclickarea)
rclickarea = np.sort(rclickarea)

# Calibration median values
openeyes = np.median(eyeopen)
leftclick = np.median(lclick) - 1
rightclick = np.median(rclick) + 1
scrolling = np.median(scroll)
leftclickarea = np.median(lclickarea)
rightclickarea = np.median(rclickarea)

print(f"Left click value = {leftclick}")
print(f"Right click value = {rightclick}")

# Create an OpenCV window with trackbars for dynamic control
cv2.namedWindow('trackbar')
cv2.createTrackbar('EyeOpen', 'trackbar', int(openeyes), int(openeyes) + 5, nothing)
cv2.createTrackbar('LClick', 'trackbar', int(leftclick), int(leftclick) + 5, nothing)
cv2.createTrackbar('RClick', 'trackbar', int(rightclick), int(rightclick) + 5, nothing)
cv2.createTrackbar('Scroll', 'trackbar', int(scrolling), int(scrolling) + 5, nothing)

# Infinite loop for real-time gesture control
while True:
    ret, image = cap.read()
    blackimage = np.zeros((480, 640, 3), dtype=np.uint8)
    image = cv2.flip(image, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect faces in the video stream
    rects = detector(gray, 0)
    
    # Process each detected face
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Compute EAR and MAR for controlling the mouse
        lefteye = EAR(shape[36], shape[37], shape[38], shape[39], shape[40], shape[41])
        righteye = EAR(shape[42], shape[43], shape[44], shape[45], shape[46], shape[47])
        mar = MAR(shape[50], shape[58], shape[51], shape[57], shape[52], shape[56], shape[48], shape[54])
        EARdiff = (lefteye - righteye) * 100
        nose = shape[30]

        # Move the mouse based on nose position and calculated angle
        pag.move((nose[0] - 300) * 2, (nose[1] - 300) * 2)

        # Draw contours around the eyes and mouth
        leftEyeHull = cv2.convexHull(shape[lstart:lend])
        rightEyeHull = cv2.convexHull(shape[rstart:rend])
        mouthHull = cv2.convexHull(shape[mstart:mend])

        cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [mouthHull], -1, (0, 255, 0), 1)

        # Perform mouse clicks and scroll based on eye and mouth gestures
        if EARdiff < cv2.getTrackbarPos('EyeOpen', 'trackbar'):
            MARlist = np.append(MARlist, [mar])
            if cv2.contourArea(leftEyeHull) < leftclickarea - 100:
                cv2.putText(image, 'Left Click', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                pag.click()
                time.sleep(0.5)
            if cv2.contourArea(rightEyeHull) < rightclickarea - 100:
                cv2.putText(image, 'Right Click', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                pag.click(button='right')
                time.sleep(0.5)

        if mar > cv2.getTrackbarPos('Scroll', 'trackbar') / 100:
            cv2.putText(image, 'scroll', (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            scroll_status += 1

            # Handle scrolling
            if scroll_status % 2 == 1:
                pag.scroll(100)
            else:
                pag.scroll(-100)

        # Draw facial landmarks and contours
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Display the result
        cv2.imshow('Cursor Control', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
