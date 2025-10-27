"""gRPC server exposing local agents and storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from importlib import import_module
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Optional

import grpc
from google.protobuf import empty_pb2

from agents.action_planner import ActionPlanner
from agents.llm_agent import LLMAgent
from agents.transcription_agent import TranscriptionAgent
from services.chat_orchestrator import ChatOrchestrator
from utils.pdf_generator import PDFGenerator
from utils.storage import ChatStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PROTO_DIR = PROJECT_ROOT / "proto"
GENERATED_DIR = BASE_DIR / "generated"
PROTO_FILE = PROTO_DIR / "video_analyzer.proto"

# Ensure transformers operate in offline-friendly mode by default
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def ensure_proto_compiled() -> tuple[ModuleType, ModuleType]:
    """Compile protobuf definitions if they are missing or stale."""
    GENERATED_DIR.mkdir(exist_ok=True)

    target_pb2 = GENERATED_DIR / "video_analyzer_pb2.py"
    target_pb2_grpc = GENERATED_DIR / "video_analyzer_pb2_grpc.py"

    needs_rebuild = (
        not target_pb2.exists()
        or not target_pb2_grpc.exists()
        or target_pb2.stat().st_mtime < PROTO_FILE.stat().st_mtime
        or target_pb2_grpc.stat().st_mtime < PROTO_FILE.stat().st_mtime
    )

    if needs_rebuild:
        logger.info("Compiling protobuf schema for gRPC service")
        try:
            from grpc_tools import protoc
        except ImportError as exc:  # pragma: no cover - startup validation
            raise RuntimeError(
                "grpcio-tools is required to build protobuf definitions."
            ) from exc

        proto_include = resources.files("grpc_tools").joinpath("_proto")

        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{PROTO_DIR}",
                f"-I{proto_include}",
                f"--python_out={GENERATED_DIR}",
                f"--grpc_python_out={GENERATED_DIR}",
                str(PROTO_FILE),
            ]
        )
        if result != 0:  # pragma: no cover - defensive
            raise RuntimeError("Failed to compile protobuf definitions")

    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))
    if str(GENERATED_DIR) not in sys.path:
        sys.path.append(str(GENERATED_DIR))

    pb2 = import_module("generated.video_analyzer_pb2")
    pb2_grpc = import_module("generated.video_analyzer_pb2_grpc")
    return pb2, pb2_grpc


video_pb2, video_pb2_grpc = ensure_proto_compiled()

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from generated import video_analyzer_pb2 as video_analyzer_types


class VideoAnalyzerService(video_pb2_grpc.VideoAnalyzerServicer):
    """Implementation of the VideoAnalyzer gRPC service."""

    def __init__(self, orchestrator: ChatOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def TranscribeVideo(  # noqa: N802 - gRPC naming
        self,
        request_iterator,
        context: grpc.aio.ServicerContext,
    ):
        buffer = bytearray()
        filename: Optional[str] = None
        session_id: Optional[str] = None

        async for chunk in request_iterator:
            if chunk.filename:
                filename = chunk.filename
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.data:
                buffer.extend(chunk.data)

        if not filename:
            filename = "upload.mp4"

        result = await self.orchestrator.transcribe_video(bytes(buffer), filename, session_id)

        return video_pb2.TranscriptionResponse(
            success=result.get('success', False),
            session_id=result.get('session_id', session_id or ''),
            transcription=result.get('transcription', ''),
            error_message=result.get('error_message', ''),
            video_filename=result.get('filename', filename),
        )

    async def Chat(  # noqa: N802
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        result = await self.orchestrator.process_chat(
            session_id=request.session_id,
            message=request.message,
            context=request.context or None,
        )

        action_plan = result.get('action_plan') or {}
        has_action_plan = bool(action_plan)

        return video_pb2.ChatResponse(
            success=result.get('success', True),
            session_id=request.session_id,
            response_text=result.get('response', ''),
            error_message=result.get('error_message', ''),
            action_plan=video_pb2.ActionPlan(
                action_type=action_plan.get('action_type', ''),
                status=action_plan.get('status', ''),
                parameters_json=json.dumps(action_plan.get('parameters', {})) if action_plan.get('parameters') is not None else '',
                missing_params=action_plan.get('missing_params', []),
                clarification_question=action_plan.get('clarification_question', ''),
                confirmation_message=action_plan.get('confirmation_message', ''),
                requires_user_input=bool(action_plan.get('requires_user_input')),
            ) if has_action_plan else None,
        )

    async def GetChatHistory(  # noqa: N802
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        history = self.orchestrator.get_chat_history(request.session_id, request.limit or 50)
        messages = []
        for message in history.get('messages', []):
            messages.append(
                video_pb2.ChatMessage(
                    id=str(message.get('id', '')),
                    role=message.get('role', ''),
                    content=message.get('content', ''),
                    timestamp=str(message.get('timestamp', '')),
                )
            )
        return video_pb2.ChatHistoryResponse(messages=messages)

    async def ClearChatHistory(  # noqa: N802
        self,
        request,
        context: grpc.aio.ServicerContext,
    ) -> empty_pb2.Empty:
        self.orchestrator.clear_chat_history(request.session_id)
        return empty_pb2.Empty()

    async def GeneratePdf(  # noqa: N802
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        result = self.orchestrator.generate_pdf_bytes(
            content=request.content,
            title=request.title,
            session_id=request.session_id,
        )
        return video_pb2.GeneratePdfResponse(
            success=result.get('success', False),
            pdf_data=result.get('pdf_data', b''),
            filename=result.get('filename', ''),
            error_message=result.get('error_message', ''),
        )

    async def GetSessionContext(  # noqa: N802
        self,
        request,
        context: grpc.aio.ServicerContext,
    ):
        context_payload = self.orchestrator.get_session_context(request.session_id)
        session = context_payload.get('session') or {}
        return video_pb2.SessionContextResponse(
            success=context_payload.get('success', False),
            error_message=context_payload.get('error_message', ''),
            session_id=session.get('session_id', ''),
            video_filename=session.get('video_filename', ''),
            transcription=session.get('transcription', ''),
            summary=session.get('summary', ''),
            created_at=str(session.get('created_at', '')),
            updated_at=str(session.get('updated_at', '')),
        )

    async def Health(  # noqa: N802
        self,
        request: empty_pb2.Empty,
        context: grpc.aio.ServicerContext,
    ):
        return video_pb2.HealthResponse(
            status="healthy",
            services={
                "transcription": "ready",
                "llm": "ready",
                "pdf": "ready",
            },
        )


async def serve(host: str = "0.0.0.0", port: int = 50051) -> None:
    orchestrator = ChatOrchestrator(
        transcription_agent=TranscriptionAgent(),
        llm_agent=LLMAgent(),
        action_planner=ActionPlanner(),
        storage=ChatStorage(),
        pdf_generator=PDFGenerator(),
    )

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 200 * 1024 * 1024),
            ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        ]
    )

    video_pb2_grpc.add_VideoAnalyzerServicer_to_server(
        VideoAnalyzerService(orchestrator),
        server,
    )

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info("Starting gRPC server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        logger.info("gRPC server interrupted, shutting down")


if __name__ == "__main__":
    main()
