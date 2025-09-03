# Face-Tracking Mouse Control

Control your computer's mouse using only your head and facial gestures. This Python script uses your webcam to provide a hands-free way to navigate your desktop, click, and scroll.

It features a robust calibration process and on-screen visual feedback to create an intuitive and responsive experience.

## ✨ Features

- 🖱️ **Relative Mouse Movement:** Move your head to move the cursor, like a virtual joystick.
- 👀 **Reliable Wink-to-Click:** Click by holding a left or right wink. The time-based detection prevents misfires from normal blinks.
- 📜 **Scroll Mode:** Open your mouth to toggle a scroll mode, then move your head up and down to scroll.
- ⚙️ **Interactive Calibration:** A guided setup process tunes the controls to your specific face and movements for maximum accuracy.
- 🔄 **On-the-Fly Re-centering:** Press a key to re-calibrate your neutral "center" position anytime you shift in your chair.
- 📊 **Live Visual Feedback:** An on-screen display shows your facial landmarks, a movement visualizer (red dot/blue circle), and a live data debugger to help you see exactly how your gestures are being interpreted.
- ⏯️ **Pause/Resume Control:** Instantly pause and resume all controls with a single key press.

## 📋 Requirements

- Python 3.8+
- A webcam
- Windows 10/11 (The batch files are for Windows, but the Python script is cross-platform)

## 🚀 Setup Instructions

Follow these steps to get the application running.

### Step 1: Clone or Download the Code

Get the project files onto your computer. This includes:

- `face_mouse.py`
- `setup.bat`
- `launch.bat`

### Step 2: Run the Automated Setup

Double-click the **`setup.bat`** file.

This script will automatically:

1.  Create a local Python virtual environment in a `venv` folder to keep dependencies isolated.
2.  Install all the required libraries (`OpenCV`, `dlib`, `PyAutoGUI`, `imutils`, `numpy`).

**Note for non-Windows users:** You will need to run these commands manually in your terminal:

```
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate  # On Windows

# Install packages
pip install opencv-python dlib numpy pyautogui imutils

```

### Step 3: Download the Facial Landmark Model

The script requires a pre-trained model file to detect facial landmarks.

1.  **Download the file here:** [**shape_predictor_68_face_landmarks.dat.bz2**](https://www.google.com/search?q=http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "null")
2.  **Unzip it.** You will get a file named `shape_predictor_68_face_landmarks.dat`.
3.  **Place this file in the same folder** as `face_mouse.py`.

Your project folder should now look like this:

```
/Face-Tracking-Mouse/
|-- face_mouse.py
|-- setup.bat
|-- launch.bat
|-- shape_predictor_68_face_landmarks.dat
|-- venv/

```

## 🕹️ How to Use

### 1. Launch the Application

Double-click **`launch.bat`**. This will activate the virtual environment and run the Python script.

### 2. Complete the Calibration

The application will guide you through a short, interactive calibration process. Follow the on-screen text and press the **`c`** key to confirm each step:

1.  **Neutral Face:** Look at the center of your screen to set your neutral point.
2.  **Left Wink:** Hold a clear left wink.
3.  **Right Wink:** Hold a clear right wink.
4.  **Mouth Open:** Hold your mouth open to set the threshold for scroll mode.

### 3. In-App Controls

Once calibrated, the main control window will appear. The application starts **PAUSED**.

Key

Action

`spacebar`

**Pause / Resume** all controls.

`r`

**Re-center** the neutral point.

`q`

**Quit** the application.

## 🛠️ Configuration & Tuning

You can fine-tune the performance by editing the variables at the top of the `face_mouse.py` script.

```
# --- CONFIGURATION --- #
WINK_HOLD_DURATION = 1.0      # Seconds to hold a wink. Increase for less sensitivity, decrease for faster clicks.
MOUSE_SENSITIVITY = 1.5       # Lower for slower mouse movement, higher for faster.
SCROLL_SENSITIVITY = 0.5      # Adjusts how fast scrolling is in scroll mode.
DEAD_ZONE_RADIUS = 7          # The radius (in pixels) around the center point where no mouse movement will occur.

```

## Troubleshooting

- **Error building wheel for `dlib`:** This means you are missing required build tools. You must install **CMake** and **Visual Studio with the "Desktop development with C++" workload**.
- **`can't grab frame` error:** This means another application (Zoom, Teams, Camera app) is using your webcam. Close all other apps or reboot your computer. Alternatively, check your Windows privacy settings to ensure desktop apps are allowed to access the camera.
- **Mouse movement is erratic:** Press the **`r`** key to re-center the neutral point. Make sure you are well-lit and facing the camera directly.
- **Clicks are not working:** Use the on-screen debugger during active use. It shows your live eye data versus the required "target" thresholds from calibration. This will help you see if you need to wink more clearly or hold it longer. If it's still an issue, re-run the calibration.
