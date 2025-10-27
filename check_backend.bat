@echo off
REM Quick check for the local gRPC backend

setlocal
set SERVER=127.0.0.1:50051

echo Checking backend server status at %SERVER% ...
echo.

python -c "import sys; from pathlib import Path; sys.path.append(str(Path('backend').resolve())); import grpc; from google.protobuf import empty_pb2; from generated import video_analyzer_pb2_grpc; channel = grpc.insecure_channel('%SERVER%'); try: stub = video_analyzer_pb2_grpc.VideoAnalyzerStub(channel); resp = stub.Health(empty_pb2.Empty()); print('✓ Backend server is running (status: ' + resp.status + ')'); except Exception as exc: print('✗ Backend server is not reachable:', exc); sys.exit(1)" 

if errorlevel 1 (
    echo.
    echo Hint: start the backend with:
    echo   start_backend.bat
    echo     or
    echo   cd backend ^&^& python grpc_server.py
)

echo.
pause
