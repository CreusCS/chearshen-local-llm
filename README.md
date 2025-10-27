# AI Video Analyzer 🎥

A fully local AI desktop application that analyzes short MP4 videos (≈1 min), enables natural language queries, and runs entirely offline using local AI models. Built with React + Tauri frontend and Python backend.

## ✨ Features

- **🎬 Video Upload**: Drag & drop or click to upload MP4 video files (up to 100MB)
- **🎤 Speech-to-Text**: Local transcription using Hugging Face Whisper models
- ** PDF Generation**: Create professional PDF reports from transcriptions
- **🤖 Local LLM Chat**: Q&A with TinyLlama model for fast responses
- **💾 Persistent Storage**: Chat history saved locally and accessible after restart
- **🧠 Human-in-the-Loop**: Intelligent clarification system requests confirmation before actions
- **🔌 MCP-Compliant**: Full Model Context Protocol support for Claude Desktop integration
- **🌐 Dual Protocol**: gRPC for the desktop app plus MCP (stdio) support
- **� Privacy-First**: Completely offline processing, no data leaves your machine

## 🎯 Functional Requirements Coverage

| Requirement | Status | Description |
|------------|--------|-------------|
| #1: MP4 File Upload | ✅ 100% | Drag-drop interface with validation |
| #2: Natural Language Interaction | ✅ 100% | Chat-based commands for all actions |
| #3: Human-in-Loop Clarification | ✅ 100% | Agentic workflows with action planner |
| #4: Persistent Chat History | ✅ 100% | SQLite + localStorage dual persistence |
| **#5: MCP Compliance** | ✅ 100% | **Full Model Context Protocol implementation** |

See `REQUIREMENTS_STATUS.md` for detailed implementation documentation.
See `MCP_IMPLEMENTATION.md` for MCP server documentation.

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

### Communication Protocols

**gRPC (Desktop Frontend)**
- **Endpoint**: `http://127.0.0.1:50051`
- **Transport**: Unary + client-streaming channels via `video_analyzer.proto`
- **Bridge**: Tauri commands marshal requests between React and gRPC

**MCP Server (Claude Desktop)**
- **Protocol**: JSON-RPC 2.0 over stdio
- **Tools**: 5 callable functions
- **Resources**: 3 accessible data sources
- **Prompts**: 2 reusable templates
- **Documentation**: See `MCP_IMPLEMENTATION.md`

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
- **CUDA** (optional, for GPU acceleration)

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
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   # source venv/bin/activate
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

### Example Commands

- `"Transcribe the video"` - Process uploaded video file
- `"Summarize the video"` - Create a summary of transcribed content
- `"Generate PDF"` - Create a PDF report with summary
- `"What was the main topic discussed?"` - Ask specific questions about content

### Features in Detail

**Video Upload**
- Supports MP4 format only
- Maximum file size: 100MB
- Progress indicator during processing
- Automatic transcription upon upload

**Chat Interface**
- Natural language interaction
- Context-aware responses
- Quick action buttons for common tasks
- Persistent chat history

**PDF Generation**
- Professional formatting
- Structured content with headers
- Automatic timestamp and metadata
- Export summaries and transcriptions

## 🔧 Configuration

### Model Configuration

The application uses these default models (automatically downloaded on first use):

- **Transcription**: `openai/whisper-small` (244MB)
- **Summarization**: `facebook/bart-large-cnn` (1.6GB)
- **LLM**: `microsoft/Phi-3-mini-4k-instruct` (3.8GB)

### Changing Models

Edit the model names in the respective agent files:

```python
# backend/agents/transcription_agent.py
model_name = "openai/whisper-small"  # or whisper-base, whisper-large

# backend/agents/summarization_agent.py
model_name = "facebook/bart-large-cnn"  # or other summarization models

# backend/agents/llm_agent.py
model_name = "microsoft/Phi-3-mini-4k-instruct"  # or mistralai/Mistral-7B-Instruct-v0.1
```

### Memory Requirements

- **Minimum**: 8GB RAM (CPU only)
- **Recommended**: 16GB RAM + 6GB VRAM (GPU acceleration)
- **Storage**: ~10GB for models and data

## 🛠️ Development

### Project Structure

```
project-root/
├── frontend/                 # React + Tauri application
│   ├── src/
│   │   ├── App.tsx          # Main application component
│   │   ├── components/      # UI components
│   │   │   ├── VideoUpload.tsx
│   │   │   └── ChatInterface.tsx
│   │   ├── services/        # gRPC client
│   │   │   └── grpcClient.ts
│   │   └── types.ts         # TypeScript definitions
│   ├── src-tauri/          # Tauri configuration
│   └── package.json
├── backend/                 # Python backend
│   ├── main.py             # gRPC server entry point
│   ├── agents/             # AI processing agents
│   │   ├── transcription_agent.py
│   │   ├── summarization_agent.py
│   │   └── llm_agent.py
│   ├── utils/              # Utilities
│   │   ├── pdf_generator.py
│   │   └── storage.py
│   └── requirements.txt
├── proto/                  # Protocol Buffer definitions
│   └── ai_service.proto
└── README.md
```

### Building for Production

1. **Backend**: Package as executable
   ```bash
   pip install pyinstaller
   pyinstaller --onefile main.py
   ```

2. **Frontend**: Build Tauri app
   ```bash
   npm run tauri build
   ```

### Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

## 🎯 Use Cases

### Content Creation
- Transcribe video content for blog posts
- Generate summaries for social media
- Create documentation from recorded meetings

### Education
- Process lecture recordings
- Generate study notes from video content
- Create searchable transcripts

### Business
- Meeting minutes from recorded sessions
- Training material documentation
- Content analysis and reporting

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
- [ ] Integration with cloud storage (optional)
- [ ] Multi-language UI support
- [ ] Advanced PDF templates and styling
- [ ] Audio-only processing support

## 🐛 Troubleshooting

### Common Issues

**"Model download failed"**
- Check internet connection for initial model download
- Ensure sufficient disk space (~2GB for distilgpt2)
- Try running with administrator privileges

**"CUDA out of memory"**
- System now uses distilgpt2 (350MB) instead of Phi-3 (3.8GB)
- CPU mode is default and performant
- Close other memory-intensive applications

**"Backend not reachable" errors**
- Confirm the gRPC server is running (`python backend/grpc_server.py`)
- Ensure port **50051** is free or update `VIDEO_ANALYZER_ENDPOINT`
- Allow the Python process through local firewall rules
- Check the Tauri console (DevTools) for detailed error messages

**"First message times out"**
- Normal behavior: First message takes 10-30 seconds (model loading)
- Subsequent messages are fast (2-5 seconds)
- See `IMPORTANT_FIRST_RUN.md` for details

**"Video processing failed"**
- Verify video file is valid MP4 format
- Check file size is under 100MB
- Ensure video contains audio track

**"Chat history not persisting"**
- Check `backend/chat_data.db` file exists
- Verify session ID in localStorage (`current_session_id`)
- Test with `test_connection_quick.py`
- See `TEST_PERSISTENCE.md` for detailed testing

**"Confirmation buttons don't appear"**
- Check frontend console for JavaScript errors
- Verify backend returns `action_plan` in response
- See `TESTING_GUIDE.md` for validation steps

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Hugging Face** for hosting the TinyLlama and Whisper models
- **OpenAI** for Whisper speech recognition
- **TinyLlama Project** for the lightweight chat model
- **Tauri** for the cross-platform desktop framework

---

**Note**: This application processes all data locally on your machine. No data is sent to external servers, ensuring complete privacy and offline functionality.