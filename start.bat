@echo off
REM Quick Start Script for AI Video Analyzer
REM Provides helpers for running the local backend or MCP server

echo ============================================================
echo AI Video Analyzer - Quick Start
echo ============================================================
echo.

:menu
echo Choose a mode:
echo   1. Start Local Desktop Backend (gRPC)
echo   2. Start MCP Server (Claude Desktop)
echo   3. Install/Update Dependencies
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto desktop_mode
if "%choice%"=="2" goto mcp_mode
if "%choice%"=="3" goto install
if "%choice%"=="4" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:install
echo.
echo Installing Python dependencies...
pushd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    popd
    pause
    goto menu
)
popd
echo.
echo Dependencies installed successfully!
echo.
pause
goto menu

:desktop_mode
echo.
echo Starting gRPC backend for the desktop client...
echo Endpoint: grpc://127.0.0.1:50051
echo.
echo Press Ctrl+C to stop the server
echo.
pushd backend
python grpc_server.py
popd
goto end

:mcp_mode
echo.
echo Starting MCP Server for Claude Desktop...
echo.
echo Ensure Claude Desktop is configured (see MCP_IMPLEMENTATION.md).
echo Press Ctrl+C to stop the server
echo.
pushd backend
python mcp_server.py
popd
goto end

:end
echo.
echo Goodbye!
