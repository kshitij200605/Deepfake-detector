@echo off
echo ============================================
echo   DeepGuard v2 - Windows Setup
echo ============================================

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Download from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found
python --version

:: Check pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found. Re-install Python with pip included.
    pause
    exit /b 1
)

:: Create virtual environment
echo.
echo [1/3] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

:: Activate and install
echo [2/3] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed. See error above.
    pause
    exit /b 1
)

echo [3/3] Setup complete!
echo.
echo ============================================
echo   To run the app:
echo     1. Double-click run_windows.bat
echo     OR
echo     1. venv\Scripts\activate
echo     2. cd src
echo     3. python app.py
echo   Then open: http://localhost:5000
echo ============================================
pause
