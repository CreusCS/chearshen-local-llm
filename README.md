# AI Video Analyzer 🎥

A fully local AI desktop application that analyzes short MP4 videos (≈1 min), enables natural language queries, and runs entirely offline using local AI models. Built with React + Tauri frontend and Python backend.

## 🏗️ Architecture

```
+---------------------------------------------------+
|                   Desktop (React + Tauri)         |
|---------------------------------------------------|
| React UI (TS)                                     |
|   ↳ Chat-first workflow, local persistence        |
|                                                   |
| Tauri Rust Layer                                 |
|   ↳ gRPC client (tonic)                           |
|   ↳ Commands exposed to React via invoke()        |
+-------------------------|-------------------------+
                          |
                          v  (gRPC localhost:50051)
+---------------------------------------------------+
|                   Local Backend (Python)          |
|---------------------------------------------------|
| gRPC Server (grpcio, asyncio)                     |
|   ↳ MCP wrapper remains available                 |
|                                                   |
| Tools / Agents:                                   |
|   • Transcription (Whisper)                       |
|   • LLM Chat (TinyLlama)                          |
|   • Action Planner                                |
|                                                   |
| Context + Storage                                 |
|   ↳ SQLite chat memory                            |
|   ↳ Human-in-loop routing                         |
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
- **MCP Server**: stdio-based protocol for Claude Desktop
- **Transcription Agent**: Whisper model for speech-to-text
- **LLM Agent**: TinyLlama-1.1B for chat and Q&A
- **Action Planner**: Intent detection and human-in-loop clarification
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

Use `start.bat` on Windows for a guided menu (option 1 launches the gRPC backend, option 2 launches the MCP server).

## 📖 Usage Guide

### Basic Workflow

1. **Upload Video**: Drag and drop an MP4 file or click the upload area
2. **Wait for Processing**: The app will automatically transcribe the video
3. **Chat with AI**: Ask questions about the content or request summaries
4. **Generate Reports**: Create PDF documents from summaries

## 🔧 Configuration

### Model Configuration

The application uses these default models (automatically downloaded on first use):

- **Transcription**: `openai/whisper-small`
- **Summarization**: `facebook/bart-large-cnn`
- **LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 

### Memory Requirements

- **Minimum**: 8GB RAM (CPU only)
- **Recommended**: 16GB RAM
- **Storage**: ~10GB for models and data


## 🚧 Known Limitations

- **Video Format**: Only MP4 files supported
- **File Size**: 100MB maximum for optimal performance
- **Languages**: Primary support for English (Whisper supports 99 languages)
- **GPU Memory**: Large models may require GPU with sufficient VRAM

## 🔮 Future Enhancements

- [ ] Support for additional video formats (AVI, MOV, MKV)
- [ ] Batch processing of multiple videos
- [ ] Advanced search functionality in chat history
- [ ] Custom model fine-tuning interface
- [ ] Advanced PDF templates and styling
- [ ] Audio-only processing support

### Getting Help

1. **Check Documentation**:
   - `QUICKSTART.md` - Fast setup guide
   - `REQUIREMENTS_STATUS.md` - Feature implementation details
   - `HUMAN_IN_LOOP_GUIDE.md` - Clarification system documentation
   - `TESTING_GUIDE.md` - Manual testing procedures

2. **Verify Installation**:
   - Start the backend and ensure it logs `Starting gRPC server`
   - Watch the backend terminal for model-loading status or stack traces
   - Confirm the frontend Tauri console shows successful command invokes

3. **Common Fixes**:
   - Restart backend server
   - Clear browser localStorage
   - Delete `chat_data.db` and restart (WARNING: loses history)
   - Reinstall Python dependencies: `pip install -r backend/requirements.txt`


## 🙏 Acknowledgments

- **Hugging Face** for hosting the TinyLlama and Whisper models
- **OpenAI** for Whisper speech recognition
- **TinyLlama Project** for the lightweight chat model
- **Tauri** for the cross-platform desktop framework

---

**Note**: This application processes all data locally on your machine. No data is sent to external servers, ensuring complete privacy and offline functionality.