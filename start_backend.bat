@echo off
REM Startup script for AI Video Analyzer Backend (gRPC)

echo Starting AI Video Analyzer Backend (gRPC)...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Activating virtual environment (if exists)...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using global Python installation.
)

echo.
echo Installing/updating dependencies...
pip install -r backend\requirements.txt

echo.
echo Starting gRPC server on grpc://localhost:50051 ...
echo Press Ctrl+C to stop the server
echo.

cd backend
python grpc_server.py

pause
