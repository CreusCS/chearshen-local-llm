# AI Video Analyzer 🎥

A fully local AI desktop application that analyzes short MP4 videos (≈1 min), enables natural language queries, and runs entirely offline using local AI models. Built with React + Tauri frontend and Python backend.

## 🏗️ Architecture

```
+---------------------------------------------------+
|               Desktop (React + Tauri)             |
|---------------------------------------------------|
| React UI (TS)                                     |
|   ↳ Chat-first workflow, local persistence        |
|                                                   |
| Tauri Rust Layer                                  |
|   ↳ gRPC client (tonic)                           |
|   ↳ Commands exposed to React via invoke()        |
+-------------------------|-------------------------+
                          | gRPC (localhost:50051)
                          v
+-------------------------|-------------------------+
|              gRPC Transport Layer                 |
|---------------------------------------------------|
| Python gRPC server (grpcio, asyncio)              |
|   ↳ Streams uploads & chat over gRPC              |
|   ↳ Delegates requests to MCP application core    |
+-------------------------|-------------------------+
                          |
                          v
+---------------------------------------------------+
|         MCP Application & Shared Tooling         |
|---------------------------------------------------|
| Model Context Protocol server (stdio)             |
| ChatOrchestrator core (agents + storage)          |
| Agents / tools:                                   |
|   • Transcription (Whisper)                       |
|   • LLM Chat (TinyLlama)                          |
|   • Hybrid Action Planner                         |
| Support services:                                 |
|   ↳ PDF generator (ReportLab)                     |
|   ↳ SQLite storage                                |
+---------------------------------------------------+
                          ^
                          | stdio
+-------------------------|-------------------------+
|        External MCP Clients (optional)            |
|---------------------------------------------------|
| Claude Desktop, VS Code MCP, etc.                 |
+---------------------------------------------------+
```

### Components

**Frontend (React + Tauri)**
- Chat-style UI for natural interaction
- Video upload with drag & drop support
- Real-time processing feedback
- Local chat history storage

**Backend (Python)**
- **gRPC Server**: Primary interface for the desktop client
- **MCP Bridge**: Shared orchestrator layer reused by gRPC and external MCP clients
- **Transcription Agent**: Whisper model for speech-to-text
- **LLM Agent**: TinyLlama-1.1B for chat, Q&A, and structured tool planning
- **Action Planner**: LLM-guided tool planner with deterministic fallback and built-in clarifications
- **PDF Generator**: ReportLab for document creation
- **Storage System**: SQLite for persistent data

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Rust** (Tauri desktop shell)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chearshen-local-llm
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install desktop (React + Tauri) dependencies**
   ```bash
   cd ../frontend
   npm install
   npm install -g @tauri-apps/cli
   ```

### Run the Local Desktop App

- **Backend (Terminal 1)**
  ```bash
  cd backend
  python grpc_server.py
  ```

- **Frontend (Terminal 2)**
  ```bash
  cd frontend
  npm run tauri dev
  ```

Use `start_backend.bat` to launch the Python service and `start_frontend.bat` for the desktop shell if you ran `setup.bat`.

## 📖 Usage Guide

### Basic Workflow

1. **Upload Video**: Drag and drop an MP4 file or click the upload area
2. **Wait for Processing**: The app will automatically transcribe the video
3. **Chat with AI**: Ask questions, trigger transcription, or request PDFs via natural language
4. **Generate Reports**: Create PDF documents from summaries, transcriptions, or custom text via LLM-guided plans

## 🔧 Configuration

### Model Configuration

The application uses these default models (automatically downloaded on first use):

- **Transcription**: `openai/whisper-small`
- **LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 

Transcription is always forced to English by the backend to keep recognition consistent across requests.

---

**Note**: This application processes all data locally on your machine. No data is sent to external servers, ensuring complete privacy and offline functionality.