@echo off
echo ============================================
echo   DeepGuard v2 - Starting Server
echo ============================================

:: Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARN] Virtual environment not found. Run setup_windows.bat first.
    echo Trying system Python...
)

:: Run
cd src
echo Starting on http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py

pause
