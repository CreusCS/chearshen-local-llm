"""
MCP (Model Context Protocol) Server Implementation
Provides standardized tools, resources, and prompts for AI video analysis
"""

import asyncio
import logging
import json
from typing import Any, Sequence
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    Resource,
    Prompt,
    PromptArgument,
    PromptMessage,
    GetPromptResult,
    INTERNAL_ERROR,
    McpError
)

from agents.transcription_agent import TranscriptionAgent
from agents.llm_agent import LLMAgent
from agents.action_planner import ActionPlanner, ActionType, ActionStatus
from utils.storage import ChatStorage
from utils.pdf_generator import PDFGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize agents and services
transcription_agent = TranscriptionAgent()
llm_agent = LLMAgent()
action_planner = ActionPlanner()
storage = ChatStorage()
pdf_generator = PDFGenerator()

# Create MCP server instance
app = Server("ai-video-analyzer")

# ============================================================================
# MCP TOOLS - Callable functions for AI agents
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools that can be called by MCP clients
    """
    return [
        Tool(
            name="transcribe_video",
            description="Transcribe a video file to text using speech recognition. Returns the full transcription of the video's audio content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Absolute path to the video file to transcribe"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to associate this transcription with"
                    }
                },
                "required": ["video_path", "session_id"]
            }
        ),
        Tool(
            name="chat",
            description="Chat with the AI assistant about video content, ask questions, or get analysis. Supports context from transcribed videos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The user's message or question"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation continuity"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context (e.g., video transcription)"
                    }
                },
                "required": ["message", "session_id"]
            }
        ),
        Tool(
            name="generate_pdf",
            description="Generate a PDF document from text content",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to include in the PDF"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the PDF document"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to associate this PDF with"
                    }
                },
                "required": ["content", "title", "session_id"]
            }
        ),
        Tool(
            name="get_session_data",
            description="Retrieve session data including video filename, transcription, and chat history",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to retrieve data for"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="plan_action",
            description="Use the action planner to determine what action to take based on user intent. Returns an action plan with clarification questions if needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "User's message to analyze for intent"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for context"
                    }
                },
                "required": ["message", "session_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """
    Execute a tool based on its name and arguments
    """
    try:
        if name == "transcribe_video":
            return await tool_transcribe_video(arguments)
        elif name == "chat":
            return await tool_chat(arguments)
        elif name == "generate_pdf":
            return await tool_generate_pdf(arguments)
        elif name == "get_session_data":
            return await tool_get_session_data(arguments)
        elif name == "plan_action":
            return await tool_plan_action(arguments)
        else:
            raise McpError(INTERNAL_ERROR, f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Tool execution error ({name}): {str(e)}")
        raise McpError(INTERNAL_ERROR, f"Tool execution failed: {str(e)}")

# Tool implementations
async def tool_transcribe_video(arguments: dict) -> Sequence[TextContent]:
    """Transcribe video to text"""
    video_path = arguments.get("video_path")
    session_id = arguments.get("session_id")
    
    if not video_path or not session_id:
        raise McpError(INTERNAL_ERROR, "Missing required arguments: video_path and session_id")
    
    # Check if file exists
    if not Path(video_path).exists():
        raise McpError(INTERNAL_ERROR, f"Video file not found: {video_path}")
    
    # Perform transcription
    result = await transcription_agent.transcribe_video(video_path)
    
    if result['success']:
        # Store transcription in session
        video_filename = Path(video_path).name
        storage.store_transcription(session_id, result['transcription'], video_filename)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "transcription": result['transcription'],
                "video_filename": video_filename,
                "session_id": session_id
            }, indent=2)
        )]
    else:
        raise McpError(INTERNAL_ERROR, f"Transcription failed: {result.get('error_message', 'Unknown error')}")

async def tool_chat(arguments: dict) -> Sequence[TextContent]:
    """Chat with AI assistant"""
    message = arguments.get("message")
    session_id = arguments.get("session_id")
    context = arguments.get("context", "")
    
    if not message or not session_id:
        raise McpError(INTERNAL_ERROR, "Missing required arguments: message and session_id")
    
    # Get session data for context
    session_data = storage.get_session_data(session_id)
    if not session_data:
        # Create new session
        storage.create_session(session_id)
        session_data = storage.get_session_data(session_id)
    
    # Store user message
    storage.store_chat_message(session_id, 'user', message)
    
    # Build context
    full_context = context or session_data.get('transcription', '')
    
    # Generate response
    response = await llm_agent.generate_response(
        message=message,
        context=full_context,
        session_id=session_id
    )
    
    if response['success']:
        # Store assistant response
        storage.store_chat_message(session_id, 'assistant', response['response'])
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "response": response['response'],
                "session_id": session_id
            }, indent=2)
        )]
    else:
        raise McpError(INTERNAL_ERROR, f"Chat failed: {response.get('error_message', 'Unknown error')}")

async def tool_generate_pdf(arguments: dict) -> Sequence[TextContent]:
    """Generate PDF document"""
    content = arguments.get("content")
    title = arguments.get("title")
    session_id = arguments.get("session_id")
    
    if not content or not title or not session_id:
        raise McpError(INTERNAL_ERROR, "Missing required arguments: content, title, and session_id")
    
    # Generate PDF
    pdf_path = pdf_generator.generate_pdf(
        title=title,
        content=content,
        session_id=session_id
    )
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "pdf_path": str(pdf_path),
            "filename": Path(pdf_path).name
        }, indent=2)
    )]

async def tool_get_session_data(arguments: dict) -> Sequence[TextContent]:
    """Retrieve session data"""
    session_id = arguments.get("session_id")
    
    if not session_id:
        raise McpError(INTERNAL_ERROR, "Missing required argument: session_id")
    
    session_data = storage.get_session_data(session_id)
    
    if not session_data:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error_message": "Session not found"
            }, indent=2)
        )]
    
    # Get chat history
    messages = storage.get_chat_messages(session_id, limit=50)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "session": {
                "session_id": session_data.get('session_id'),
                "video_filename": session_data.get('video_filename'),
                "transcription": session_data.get('transcription'),
                "created_at": session_data.get('created_at'),
                "updated_at": session_data.get('updated_at')
            },
            "messages": messages
        }, indent=2)
    )]

async def tool_plan_action(arguments: dict) -> Sequence[TextContent]:
    """Plan action based on user intent"""
    message = arguments.get("message")
    session_id = arguments.get("session_id")
    
    if not message or not session_id:
        raise McpError(INTERNAL_ERROR, "Missing required arguments: message and session_id")
    
    # Get session data for context
    session_data = storage.get_session_data(session_id)
    if not session_data:
        storage.create_session(session_id)
        session_data = storage.get_session_data(session_id)
    
    # Build context
    context = {
        'has_video': bool(session_data.get('video_filename')),
        'video_filename': session_data.get('video_filename'),
        'has_transcription': bool(session_data.get('transcription')),
        'transcription': session_data.get('transcription')
    }
    
    # Plan action
    action_plan = action_planner.plan_action(message, context)
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "success": True,
            "action_plan": action_plan.to_dict()
        }, indent=2)
    )]

# ============================================================================
# MCP RESOURCES - Accessible data for AI agents
# ============================================================================

@app.list_resources()
async def list_resources() -> list[Resource]:
    """
    List all available resources that can be read by MCP clients
    """
    return [
        Resource(
            uri="session://list",
            name="List all sessions",
            description="Get a list of all available sessions",
            mimeType="application/json"
        ),
        Resource(
            uri="session://{session_id}/transcription",
            name="Session transcription",
            description="Get the video transcription for a specific session",
            mimeType="text/plain"
        ),
        Resource(
            uri="session://{session_id}/messages",
            name="Session chat messages",
            description="Get the chat message history for a specific session",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    """
    Read a resource by its URI
    """
    try:
        if uri == "session://list":
            # List all sessions
            sessions = storage.list_sessions()
            return json.dumps(sessions, indent=2)
        
        elif uri.startswith("session://") and uri.endswith("/transcription"):
            # Extract session_id
            session_id = uri.split("/")[2]
            session_data = storage.get_session_data(session_id)
            
            if not session_data:
                raise McpError(INTERNAL_ERROR, f"Session not found: {session_id}")
            
            transcription = session_data.get('transcription', '')
            return transcription or "No transcription available"
        
        elif uri.startswith("session://") and uri.endswith("/messages"):
            # Extract session_id
            session_id = uri.split("/")[2]
            messages = storage.get_chat_messages(session_id, limit=100)
            return json.dumps(messages, indent=2)
        
        else:
            raise McpError(INTERNAL_ERROR, f"Unknown resource URI: {uri}")
            
    except Exception as e:
        logger.error(f"Resource read error: {str(e)}")
        raise McpError(INTERNAL_ERROR, f"Failed to read resource: {str(e)}")

# ============================================================================
# MCP PROMPTS - Reusable prompt templates
# ============================================================================

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """
    List all available prompt templates
    """
    return [
        Prompt(
            name="analyze_video",
            description="Analyze a transcribed video and provide insights",
            arguments=[
                PromptArgument(
                    name="session_id",
                    description="Session ID containing the video transcription",
                    required=True
                ),
                PromptArgument(
                    name="focus",
                    description="What aspect to focus on (e.g., 'summary', 'key_points', 'action_items')",
                    required=False
                )
            ]
        ),
        Prompt(
            name="video_qa",
            description="Answer questions about a video based on its transcription",
            arguments=[
                PromptArgument(
                    name="session_id",
                    description="Session ID containing the video transcription",
                    required=True
                ),
                PromptArgument(
                    name="question",
                    description="The question to answer about the video",
                    required=True
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    """
    Get a specific prompt by name with arguments
    """
    if name == "analyze_video":
        session_id = arguments.get("session_id") if arguments else None
        focus = arguments.get("focus", "summary") if arguments else "summary"
        
        if not session_id:
            raise McpError(INTERNAL_ERROR, "Missing required argument: session_id")
        
        # Get session data
        session_data = storage.get_session_data(session_id)
        if not session_data:
            raise McpError(INTERNAL_ERROR, f"Session not found: {session_id}")
        
        transcription = session_data.get('transcription', '')
        video_filename = session_data.get('video_filename', 'the video')
        
        prompt_text = f"""Analyze the following video transcription and provide {focus}.

Video: {video_filename}

Transcription:
{transcription}

Please provide a detailed {focus} of the video content."""
        
        return GetPromptResult(
            description=f"Analyze video with focus on {focus}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )
    
    elif name == "video_qa":
        session_id = arguments.get("session_id") if arguments else None
        question = arguments.get("question") if arguments else None
        
        if not session_id or not question:
            raise McpError(INTERNAL_ERROR, "Missing required arguments: session_id and question")
        
        # Get session data
        session_data = storage.get_session_data(session_id)
        if not session_data:
            raise McpError(INTERNAL_ERROR, f"Session not found: {session_id}")
        
        transcription = session_data.get('transcription', '')
        
        prompt_text = f"""Based on the following video transcription, please answer this question:

Question: {question}

Transcription:
{transcription}

Please provide a detailed answer based on the video content."""
        
        return GetPromptResult(
            description="Answer question about video",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=prompt_text
                    )
                )
            ]
        )
    
    else:
        raise McpError(INTERNAL_ERROR, f"Unknown prompt: {name}")

# ============================================================================
# SERVER MAIN - Entry point for MCP server
# ============================================================================

async def main():
    """
    Main entry point for MCP server (stdio transport)
    """
    logger.info("Starting MCP server on stdio...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
