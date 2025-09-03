@echo off
title Face Mouse Launcher

echo Starting Face Mouse...

:: Check if the virtual environment exists
if not exist ".\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b
)

:: Activate the virtual environment and launch the Python script
call .\venv\Scripts\activate.bat
python face_mouse.py