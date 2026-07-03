@echo off
echo ============================================================
echo   AetherFlow Gesture Control Mouse v9.0.0 - Console Edition
echo ============================================================
echo.
echo   Webcam frames and AI inference are fully multithreaded.
echo   Emulation is directly powered by zero-latency Win32 APIs.
echo.
echo   Keyboard Controls (in the video HUD window):
echo     - Press '+' to increase sensitivity.
echo     - Press '-' to decrease sensitivity.
echo     - Press ']' to increase pinch click threshold.
echo     - Press '[' to decrease pinch click threshold.
echo     - Press 'p' to pause / resume mouse pointer movement.
echo     - Press 'q' to quit.
echo.
python "%~dp0gesture_mouse.py"
pause
