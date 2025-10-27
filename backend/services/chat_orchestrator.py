import logging
from typing import Any, Dict, Optional

from agents.action_planner import (
    ActionPlanner,
    ActionPlan,
    ActionStatus,
    ActionType,
)
from agents.llm_agent import LLMAgent
from agents.transcription_agent import TranscriptionAgent
from utils.pdf_generator import PDFGenerator
from utils.storage import ChatStorage

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Coordinates chat, action planning, and storage across transports."""

    def __init__(
        self,
        transcription_agent: TranscriptionAgent,
        llm_agent: LLMAgent,
        action_planner: ActionPlanner,
        storage: ChatStorage,
        pdf_generator: PDFGenerator,
    ) -> None:
        self.transcription_agent = transcription_agent
        self.llm_agent = llm_agent
        self.action_planner = action_planner
        self.storage = storage
        self.pdf_generator = pdf_generator

    def ensure_session(self, session_id: str, video_filename: Optional[str] = None) -> Dict[str, Any]:
        """Ensure session exists and optionally update video metadata."""
        session = self.storage.get_session_data(session_id)
        if not session:
            logger.info("Creating new session %s", session_id)
            self.storage.create_session(session_id=session_id, video_filename=video_filename)
            session = self.storage.get_session_data(session_id)
        elif video_filename:
            self.storage.update_session_video(session_id, video_filename)
            session = self.storage.get_session_data(session_id)
        return session or {
            'session_id': session_id,
            'video_filename': video_filename,
            'transcription': None,
            'summary': None,
            'context': {},
        }

    async def process_chat(self, session_id: str, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Handle chat interaction with action planning and storage."""
        session_data = self.ensure_session(session_id)

        self.storage.store_chat_message(session_id, 'user', message)

        conversation_context = {
            'has_video': bool(session_data.get('video_filename')),
            'video_filename': session_data.get('video_filename'),
            'has_transcription': bool(session_data.get('transcription')),
            'transcription': session_data.get('transcription'),
            'has_summary': bool(session_data.get('summary')),
            'summary': session_data.get('summary'),
        }

        session_context = session_data.get('context', {})
        pending_action_raw = session_context.get('pending_action')

        if pending_action_raw:
            action_plan = self._deserialize_action_plan(pending_action_raw)
            updated_plan = self.action_planner.process_clarification_response(action_plan, message)
            response_text, action_payload = await self._handle_action_plan(
                updated_plan,
                session_context,
                session_id,
                session_data,
            )
            return {
                'success': True,
                'response': response_text,
                'action_plan': action_payload,
            }

        action_plan = self.action_planner.plan_action(message, conversation_context)
        response_text, action_payload = await self._handle_action_plan(
            action_plan,
            session_context,
            session_id,
            session_data,
            context_override=context,
        )
        return {
            'success': True,
            'response': response_text,
            'action_plan': action_payload,
        }

    async def _handle_action_plan(
        self,
        action_plan: ActionPlan,
        session_context: Dict[str, Any],
        session_id: str,
        session_data: Dict[str, Any],
        context_override: Optional[str] = None,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Process action plan lifecycle."""
        if action_plan.status in (ActionStatus.NEEDS_CLARIFICATION, ActionStatus.REQUIRES_CONFIRMATION):
            session_context['pending_action'] = action_plan.to_dict()
            self.storage.update_session_context(session_id, session_context)
            response_text = action_plan.clarification_question or action_plan.confirmation_message or "I need more details."
            self.storage.store_chat_message(session_id, 'assistant', response_text)
            return response_text, action_plan.to_dict()

        if action_plan.status == ActionStatus.READY_TO_EXECUTE:
            response_text = await self.execute_action(action_plan, session_data, session_id, context_override)
            self.storage.store_chat_message(session_id, 'assistant', response_text)
            session_context.pop('pending_action', None)
            self.storage.update_session_context(session_id, session_context)
            return response_text, None

        response = await self.llm_agent.generate_response(
            message=action_plan.parameters.get('question', ''),
            context=context_override or action_plan.parameters.get('context'),
            session_id=session_id,
        )
        self.storage.store_chat_message(session_id, 'assistant', response['response'])
        return response['response'], None

    async def execute_action(
        self,
        action_plan: ActionPlan,
        session_data: Dict[str, Any],
        session_id: str,
        context_override: Optional[str] = None,
    ) -> str:
        """Execute a resolved action plan."""
        try:
            if action_plan.action_type == ActionType.ANSWER_QUESTION:
                response = await self.llm_agent.generate_response(
                    message=action_plan.parameters.get('question', ''),
                    context=context_override or action_plan.parameters.get('context'),
                    session_id=session_id,
                )
                return response['response']

            if action_plan.action_type == ActionType.TRANSCRIBE_VIDEO:
                return (
                    "Video transcription requires uploading a video file through the upload interface. "
                    "Once uploaded, I can transcribe it for you."
                )

            if action_plan.action_type == ActionType.GENERATE_PDF:
                title = action_plan.parameters.get('title', 'Document')
                content_type = action_plan.parameters.get('content', 'summary')
                content = session_data.get(content_type) or session_data.get('transcription') or ''
                if not content:
                    return "No content available to generate PDF."
                pdf_path = self.pdf_generator.generate_pdf(
                    title=title,
                    content=content,
                    session_id=session_id,
                )
                return f"PDF generated successfully: {pdf_path}\n\nYou can download it from the interface."

            return "I'm not sure how to help with that. Can you rephrase your request?"

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Action execution error: %s", exc)
            return f"Sorry, I encountered an error while executing the action: {exc}"

    async def transcribe_video(self, video_bytes: bytes, filename: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe video and persist the results."""
        result = await self.transcription_agent.transcribe_video(video_bytes, filename)

        if not session_id:
            session_id = self.storage.create_session(video_filename=filename)
        else:
            self.ensure_session(session_id, filename)

        if result['success']:
            self.storage.store_transcription(session_id, result['transcription'])
            self.storage.update_session_video(session_id, filename)
            self.storage.store_chat_message(
                session_id,
                'system',
                (
                    f"Video '{filename}' has been transcribed successfully. "
                    "You can now ask questions about it or request a summary."
                ),
            )

        return {
            'session_id': session_id,
            'filename': filename,
            **result,
        }

    def get_chat_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """Return chat history payload."""
        messages = self.storage.get_chat_history(session_id, limit if limit > 0 else None)
        return {
            'messages': messages,
            'success': True,
            'error_message': '',
        }

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Return session metadata."""
        session = self.storage.get_session_data(session_id)
        if not session:
            return {
                'success': False,
                'error_message': 'Session not found',
            }
        return {
            'success': True,
            'session': {
                'session_id': session.get('session_id'),
                'video_filename': session.get('video_filename'),
                'transcription': session.get('transcription'),
                'summary': session.get('summary'),
                'created_at': session.get('created_at'),
                'updated_at': session.get('updated_at'),
            },
        }

    def clear_chat_history(self, session_id: str) -> Dict[str, Any]:
        """Clear chat history for a session."""
        self.storage.clear_chat_history(session_id)
        return {
            'success': True,
            'error_message': '',
        }

    def generate_pdf_bytes(self, content: str, title: str, session_id: str) -> Dict[str, Any]:
        """Generate a PDF and return binary data."""
        try:
            pdf_data = self.pdf_generator.create_pdf(content=content, title=title)
            filename = f"{title.replace(' ', '_')}.pdf"
            return {
                'success': True,
                'pdf_data': pdf_data,
                'filename': filename,
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("PDF generation error: %s", exc)
            return {
                'success': False,
                'error_message': str(exc),
                'pdf_data': b'',
                'filename': '',
            }

    def _deserialize_action_plan(self, payload: Dict[str, Any]) -> ActionPlan:
        """Recreate ActionPlan instance from stored context."""
        return ActionPlan(
            action_type=ActionType(payload.get('action_type', ActionType.UNKNOWN.value)),
            status=ActionStatus(payload.get('status', ActionStatus.READY_TO_EXECUTE.value)),
            parameters=payload.get('parameters') or {},
            missing_params=payload.get('missing_params') or [],
            clarification_question=payload.get('clarification_question'),
            confirmation_message=payload.get('confirmation_message'),
        )
