"""Model Context Protocol server and application bridge for AI Video Analyzer."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Dict, Optional, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, Resource, TextContent, Tool

from agents.action_planner import ActionPlanner
from agents.llm_agent import LLMAgent
from agents.transcription_agent import TranscriptionAgent
from services.chat_orchestrator import ChatOrchestrator
from utils.pdf_generator import PDFGenerator
from utils.storage import ChatStorage

logger = logging.getLogger(__name__)


class MCPApplication:
    """Shared application core backing both gRPC and MCP transports."""

    def __init__(self) -> None:
        self._storage = ChatStorage()
        self._orchestrator = ChatOrchestrator(
            transcription_agent=TranscriptionAgent(),
            llm_agent=LLMAgent(),
            action_planner=ActionPlanner(),
            storage=self._storage,
            pdf_generator=PDFGenerator(),
        )

    async def process_chat(
        self,
        *,
        session_id: str,
        message: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a chat request via the orchestrator."""

        return await self._orchestrator.process_chat(
            session_id=session_id,
            message=message,
            context=context,
        )

    async def transcribe_video(
        self,
        video_bytes: bytes,
        filename: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a video through the orchestrator."""

        return await self._orchestrator.transcribe_video(
            video_bytes=video_bytes,
            filename=filename,
            session_id=session_id,
        )

    async def generate_pdf(
        self,
        *,
        session_id: str,
        title: str,
        content: str,
    ) -> Dict[str, Any]:
        """Generate a PDF document on a worker thread."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._orchestrator.generate_pdf_bytes(
                content=content,
                title=title,
                session_id=session_id,
            ),
        )

    def get_chat_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        return self._orchestrator.get_chat_history(session_id, limit)

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        return self._orchestrator.get_session_context(session_id)

    def clear_chat_history(self, session_id: str) -> Dict[str, Any]:
        return self._orchestrator.clear_chat_history(session_id)

    def get_storage(self) -> ChatStorage:
        return self._storage


_APP: MCPApplication | None = None


def get_mcp_app() -> MCPApplication:
    """Return a module-level singleton of the MCP application."""

    global _APP
    if _APP is None:
        _APP = MCPApplication()
    return _APP


server = Server("ai-video-analyzer")


@server.list_resources()
async def list_resources() -> Sequence[Resource]:
    """Expose MCP resources for session metadata and chat history."""
    app = get_mcp_app()
    storage = app.get_storage()
    sessions = storage.get_all_sessions()
    resources: list[Resource] = []
    for session in sessions:
        session_id = session.get("session_id")
        if not session_id:
            continue
        resources.append(
            Resource(
                uri=f"memory://session/{session_id}",
                name=f"Session {session_id}",
                description="Stored session metadata and chat history.",
            )
        )
    return resources


@server.read_resource()
async def read_resource(uri: str) -> EmbeddedResource:
    """Return a JSON payload describing the requested session resource."""
    app = get_mcp_app()
    if not uri.startswith("memory://session/"):
        raise ValueError(f"Unsupported resource: {uri}")

    session_id = uri.split("/", 2)[-1]
    context = app.get_session_context(session_id)
    history = app.get_chat_history(session_id, limit=100)
    payload = {
        "session": context,
        "history": history,
    }
    return EmbeddedResource(
        uri=uri,
        mime_type="application/json",
        data=json.dumps(payload).encode("utf-8"),
    )


@server.list_tools()
async def list_tools() -> Sequence[Tool]:
    return [
        Tool(
            name="process_chat",
            description="Process a chat message via the agentic orchestrator.",
        ),
        Tool(
            name="transcribe_video",
            description="Transcribe an uploaded video using the local Whisper model.",
        ),
        Tool(
            name="generate_pdf",
            description="Generate a PDF report for a session using stored content.",
        ),
        Tool(
            name="get_session_context",
            description="Fetch stored metadata for a given session.",
        ),
        Tool(
            name="get_chat_history",
            description="Fetch stored chat history for a given session.",
        ),
        Tool(
            name="clear_chat_history",
            description="Clear chat history for a session.",
        ),
    ]


async def _handle_process_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    app = get_mcp_app()
    session_id = payload.get("session_id")
    message = payload.get("message")
    if not session_id or not message:
        raise ValueError("process_chat requires session_id and message")
    context = payload.get("context")
    return await app.process_chat(session_id=session_id, message=message, context=context)


async def _handle_transcription(payload: Dict[str, Any]) -> Dict[str, Any]:
    app = get_mcp_app()
    video_b64 = payload.get("video_base64")
    if not video_b64:
        raise ValueError("transcribe_video requires video_base64")
    filename = payload.get("filename", "upload.mp4")
    session_id = payload.get("session_id")
    video_bytes = base64.b64decode(video_b64)
    return await app.transcribe_video(video_bytes, filename, session_id)


async def _handle_generate_pdf(payload: Dict[str, Any]) -> Dict[str, Any]:
    app = get_mcp_app()
    session_id = payload.get("session_id")
    if not session_id:
        raise ValueError("generate_pdf requires session_id")
    title = payload.get("title", "Document")
    content = payload.get("content", "")
    return await app.generate_pdf(session_id=session_id, title=title, content=content)


async def _call_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


_TOOL_HANDLERS = {
    "process_chat": _handle_process_chat,
    "transcribe_video": _handle_transcription,
    "generate_pdf": _handle_generate_pdf,
    "get_session_context": lambda payload: _call_sync(
        get_mcp_app().get_session_context,
        payload.get("session_id", ""),
    ),
    "get_chat_history": lambda payload: _call_sync(
        get_mcp_app().get_chat_history,
        payload.get("session_id", ""),
        payload.get("limit", 50),
    ),
    "clear_chat_history": lambda payload: _call_sync(
        get_mcp_app().clear_chat_history,
        payload.get("session_id", ""),
    ),
}


@server.call_tool()
async def call_tool(name: str, arguments: Optional[str]) -> Sequence[TextContent]:
    payload = json.loads(arguments or "{}")
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported tool: {name}")

    result = await handler(payload)
    return [TextContent(type="text", text=json.dumps(result))]


async def run_stdio_server() -> None:
    """Launch the MCP server over stdio."""
    transport = await stdio_server(server).serve()
    await transport.wait_closed()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_stdio_server())
    except KeyboardInterrupt:
        logger.info("MCP server interrupted, shutting down")


if __name__ == "__main__":
    main()
