@echo off
title Face Mouse Setup

echo ==========================================================
echo           Setting up Face Mouse Environment
echo ==========================================================
echo.

echo [1/3] Checking for Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not added to your PATH.
    echo Please install Python from python.org and try again.
    pause
    exit /b
)
echo Python found.
echo.

echo [2/3] Creating virtual environment in 'venv' folder...
python -m venv venv
echo Virtual environment created.
echo.

echo [3/3] Installing required libraries...
.\venv\Scripts\pip.exe install opencv-python dlib numpy pyautogui imutils
echo Libraries installed successfully.
echo.

echo ==========================================================
echo                      SETUP COMPLETE!
echo ==========================================================
echo.
echo IMPORTANT: You still need to download one file manually.
echo.
echo 1. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
echo 2. Unzip the file.
echo 3. Place the 'shape_predictor_68_face_landmarks.dat' file in this folder.
echo.
pause