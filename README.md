# AI Video Analyzer 🎥

A fully local AI desktop application that analyzes short MP4 videos (≈1 min), enables natural language queries, and runs entirely offline using local AI models. Built with React + Tauri frontend and Python backend.

## ✨ Features

- **🎬 Video Upload**: Drag & drop or click to upload MP4 video files (up to 100MB)
- **🎤 Speech-to-Text**: Local transcription using Hugging Face Whisper models
- **📝 Summarization**: AI-powered content summarization with structured output
- **📄 PDF Generation**: Create professional PDF reports from summaries
- **🤖 Local LLM Chat**: Q&A with quantized models (Phi-3 Mini, Mistral 7B)
- **💾 Persistent Storage**: Chat history saved locally and accessible after restart
- **🔒 Privacy-First**: Completely offline processing, no data leaves your machine

## 🏗️ Architecture

```
┌─────────────────┐    gRPC     ┌─────────────────────┐
│  React + Tauri  │ ◄──────────► │   Python Backend   │
│    Frontend     │             │                     │
├─────────────────┤             ├─────────────────────┤
│ • Chat UI       │             │ • Transcription     │
│ • Video Upload  │             │ • Summarization     │
│ • Local Storage │             │ • LLM Integration   │
│ • gRPC Client   │             │ • PDF Generation    │
└─────────────────┘             └─────────────────────┘
```

### Components

**Frontend (React + Tauri)**
- Chat-style UI for natural interaction
- Video upload with drag & drop support
- Real-time processing feedback
- Local chat history storage

**Backend (Python)**
- **Transcription Agent**: Whisper model for speech-to-text
- **Summarization Agent**: BART model for content summarization
- **LLM Agent**: Quantized Phi-3 Mini for general Q&A
- **PDF Generator**: ReportLab for document creation
- **Storage System**: SQLite for persistent data

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Rust** (for Tauri development)
- **CUDA** (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm_desktop
   ```

2. **Set up the Python backend**
   ```bash
   cd backend
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the React frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Install Tauri CLI** (if not already installed)
   ```bash
   npm install -g @tauri-apps/cli
   ```

### Running the Application

1. **Start the Python backend** (in `backend/` directory)
   ```bash
   python main.py
   ```
   The gRPC server will start on `localhost:50051`

2. **Start the frontend** (in `frontend/` directory)
   ```bash
   npm run tauri dev
   ```
   This will launch the Tauri application

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
- Ensure sufficient disk space (~10GB)
- Try running with administrator privileges

**"CUDA out of memory"**
- Reduce model size or use CPU-only mode
- Close other GPU-intensive applications
- Adjust batch sizes in agent configurations

**"gRPC connection failed"**
- Ensure backend server is running on port 50051
- Check firewall settings
- Verify Python dependencies are installed

**"Video processing failed"**
- Verify video file is valid MP4 format
- Check file size is under 100MB
- Ensure video contains audio track

### Getting Help

1. Check the logs in the backend console for detailed error messages
2. Verify all dependencies are properly installed
3. Ensure sufficient system resources are available
4. Try with a smaller test video file first

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

- **Hugging Face** for providing excellent pre-trained models
- **OpenAI** for Whisper speech recognition
- **Facebook** for BART summarization model
- **Microsoft** for Phi-3 language model
- **Tauri** for the cross-platform desktop framework

---

**Note**: This application processes all data locally on your machine. No data is sent to external servers, ensuring complete privacy and offline functionality.