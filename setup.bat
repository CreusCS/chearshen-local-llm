@echo off
REM Setup script for AI Video Analyzer on Windows

echo 🎥 AI Video Analyzer Setup Script
echo =================================

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set python_version=%%i
echo ✅ Found Python %python_version%

REM Check if Node.js is installed
echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 18 or higher.
    pause
    exit /b 1
)

for /f %%i in ('node --version') do set node_version=%%i
echo ✅ Found Node.js %node_version%

REM Check if Rust is installed
echo Checking Rust installation...
rustc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Rust is not installed. Please install Rust from https://rustup.rs/
    echo After installing Rust, run this script again.
    pause
    exit /b 1
)

for /f "tokens=1,2" %%i in ('rustc --version') do set rust_version=%%i %%j
echo ✅ Found Rust: %rust_version%

REM Set up Python backend
echo.
echo Setting up Python backend...
cd backend

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Backend setup complete!

REM Set up frontend
echo.
echo Setting up React frontend...
cd ..\frontend

REM Install Node.js dependencies
echo Installing Node.js dependencies...
npm install --foreground-scripts=true

REM Install Tauri CLI if not present
echo Installing Tauri CLI...
npm install -g @tauri-apps/cli

echo ✅ Frontend setup complete!

REM Create startup scripts
echo.
echo Creating startup scripts...
cd ..

REM Backend startup script
if not exist start_backend.bat (
    echo @echo off > start_backend.bat
    echo cd backend >> start_backend.bat
    echo call venv\Scripts\activate >> start_backend.bat
    echo echo Starting AI Video Analyzer Backend... >> start_backend.bat
    echo python grpc_server.py >> start_backend.bat
    echo pause >> start_backend.bat
)

REM Frontend startup script
if not exist start_frontend.bat (
    echo @echo off > start_frontend.bat
    echo cd frontend >> start_frontend.bat
    echo echo Starting AI Video Analyzer Frontend... >> start_frontend.bat
    echo npm run tauri dev >> start_frontend.bat
    echo pause >> start_frontend.bat
)

echo.
echo 🎉 Setup complete!
echo.
echo To start the application:
echo 1. Double-click start_backend.bat
echo 2. Double-click start_frontend.bat
echo.
echo The application will be available on your desktop!
pause