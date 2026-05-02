@echo off
REM ═══════════════════════════════════════════════════════════════
REM  RoadSense — First-time Windows setup
REM  Run this ONCE after you unzip the project.
REM ═══════════════════════════════════════════════════════════════

setlocal
cd /d "%~dp0"

echo.
echo ========================================================
echo   RoadSense Setup (Windows)
echo ========================================================
echo.

REM --- 1. Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% detected.
echo.

REM --- 2. Create venv ---
if not exist .venv (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create venv.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM --- 3. Install deps ---
echo Installing Python packages (this takes 3-5 minutes)...
echo.
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo [OK] All packages installed.
echo.

REM --- 4. Copy .env template ---
if not exist .env (
    copy .env.example .env >nul
    echo [OK] Created .env from template.
    echo.
    echo     Now open .env in VS Code and paste your
    echo     GOOGLE_MAPS_API_KEY before running the app.
    echo.
) else (
    echo [OK] .env already exists.
)

REM --- 5. Check weights ---
if not exist weights\STCrackNet_final.pth (
    echo.
    echo [INFO] STCrackNet_final.pth not found in weights\
    echo        Satellite mode will still work.
    echo        To enable pavement-upload mode, download
    echo        STCrackNet_final.pth from your Google Drive
    echo        and place it in:  weights\STCrackNet_final.pth
    echo.
) else (
    echo [OK] Model weights found.
)

echo.
echo ========================================================
echo   Setup complete!
echo ========================================================
echo.
echo   Next steps:
echo     1. Edit .env  and add your GOOGLE_MAPS_API_KEY
echo     2. Double-click  run.bat  to start the server
echo.
pause
