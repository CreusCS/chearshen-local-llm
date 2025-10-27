"""
REST API wrapper for the gRPC services
Provides HTTP endpoints that frontend can easily consume
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from agents.transcription_agent import TranscriptionAgent
from agents.llm_agent import LLMAgent
from agents.action_planner import ActionPlanner, ActionStatus, ActionPlan
from utils.storage import ChatStorage
from utils.pdf_generator import PDFGenerator
from services.chat_orchestrator import ChatOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Video Analyzer API",
    description="REST API for video transcription and LLM chat",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:1420", "http://127.0.0.1:5173", "http://127.0.0.1:1420"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
transcription_agent = TranscriptionAgent()
llm_agent = LLMAgent()
action_planner = ActionPlanner()
storage = ChatStorage()
pdf_generator = PDFGenerator()
orchestrator = ChatOrchestrator(
    transcription_agent=transcription_agent,
    llm_agent=llm_agent,
    action_planner=action_planner,
    storage=storage,
    pdf_generator=pdf_generator,
)

# Request/Response models
class TranscriptionResponse(BaseModel):
    transcription: str
    success: bool
    error_message: str = ""
    session_id: str

class PDFRequest(BaseModel):
    content: str
    title: str
    session_id: str

class PDFResponse(BaseModel):
    success: bool
    error_message: str = ""
    filename: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    success: bool
    error_message: str = ""
    session_id: str
    action_plan: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    services: dict

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and service status"""
    return {
        "status": "healthy",
        "services": {
            "transcription": "ready",
            "llm": "ready",
            "pdf": "ready"
        }
    }

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    file: UploadFile = File(...),
    session_id: str = None
):
    """
    Transcribe uploaded video file to text
    """
    try:
        logger.info(f"Transcribing video: {file.filename}")

        video_data = await file.read()
        result = await orchestrator.transcribe_video(
            video_bytes=video_data,
            filename=file.filename or "unknown.mp4",
            session_id=session_id,
        )

        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error_message', 'Transcription failed'))

        logger.info(f"Transcription stored in session: {result['session_id']}")

        return TranscriptionResponse(
            transcription=result.get('transcription', ''),
            success=True,
            error_message=result.get('error_message', ''),
            session_id=result['session_id']
        )

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-pdf")
async def generate_pdf(request: PDFRequest):
    """
    Generate PDF from content
    """
    try:
        logger.info(f"Generating PDF: {request.title}")
        
        pdf_result = orchestrator.generate_pdf_bytes(
            content=request.content,
            title=request.title,
            session_id=request.session_id
        )

        if not pdf_result['success']:
            raise HTTPException(status_code=500, detail=pdf_result.get('error_message', 'Failed to generate PDF'))
        
        filename = pdf_result['filename']
        pdf_data = pdf_result['pdf_data']
        
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle LLM chat messages with action planning and clarification
    """
    # Import ActionType and ActionStatus at the start
    from agents.action_planner import ActionType, ActionStatus
    
    try:
        logger.info(f"Processing chat for session: {request.session_id}")

        result = await orchestrator.process_chat(
            session_id=request.session_id,
            message=request.message,
            context=request.context,
        )

        return ChatResponse(
            response=result['response'],
            success=result['success'],
            error_message=result.get('error_message', ''),
            session_id=request.session_id,
            action_plan=result.get('action_plan')
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """
    Retrieve chat history for a session
    """
    try:
        history = orchestrator.get_chat_history(session_id, limit)
        return history
        
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_context(session_id: str):
    """
    Get full session context including video, transcription, and summary
    """
    try:
        return orchestrator.get_session_context(session_id)
        
    except Exception as e:
        logger.error(f"Session context error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear chat history for a session
    """
    try:
        return orchestrator.clear_chat_history(session_id)
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-info")
async def get_model_info():
    """
    Get information about the loaded LLM model
    """
    try:
        info = llm_agent.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the REST API server
    """
    logger.info(f"Starting REST API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
