@echo off
REM ═══════════════════════════════════════════════════════════════
REM  RoadSense — Launch the server on Windows
REM  Double-click this file to run.
REM ═══════════════════════════════════════════════════════════════

setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    echo [ERROR] Virtual environment not found.
    echo        Run setup.bat first.
    echo.
    pause
    exit /b 1
)

if not exist .env (
    echo [ERROR] .env not found.
    echo        Run setup.bat first, then edit .env and add your API key.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo   Starting RoadSense server...
echo ========================================================
echo.
echo   Dashboard:  http://localhost:8000
echo   API docs:   http://localhost:8000/docs
echo   Health:     http://localhost:8000/health
echo.
echo   Press Ctrl+C to stop.
echo.

call .venv\Scripts\activate.bat

REM Open browser after a short delay so the server has time to start
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:8000"

python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
