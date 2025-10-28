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
            if (
                action_plan.action_type == ActionType.TRANSCRIBE_VIDEO
                and action_plan.status == ActionStatus.NEEDS_CLARIFICATION
                and conversation_context.get('has_video')
            ):
                logger.info(
                    "Dropping stale pending transcription clarification now that a video is available."
                )
                session_context.pop('pending_action', None)
                self.storage.update_session_context(session_id, session_context)
                action_plan = None

            if action_plan:
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

        structured_plan = await self.llm_agent.plan_action(
            message=message,
            context=conversation_context,
            available_actions=self._available_actions_description(),
        )
        if structured_plan is not None:
            logger.info("Structured plan candidate from LLM: action=%s status=%s", structured_plan.get('action_type'), structured_plan.get('status'))
        else:
            logger.info("LLM planner returned no structured plan; using rule-based planner.")

        heuristic_plan = self.action_planner.plan_action(message, conversation_context)

        action_plan = self._build_action_plan_from_structured(structured_plan, conversation_context)
        if action_plan is None:
            if structured_plan is not None:
                logger.info("Structured plan rejected after validation; falling back to heuristic planner.")
            action_plan = heuristic_plan
        else:
            if (
                heuristic_plan
                and heuristic_plan.action_type != ActionType.UNKNOWN
                and heuristic_plan.action_type != action_plan.action_type
                and heuristic_plan.status != ActionStatus.FAILED
            ):
                logger.info(
                    "Structured plan (%s) conflicts with heuristic (%s); preferring heuristic planner.",
                    action_plan.action_type.value,
                    heuristic_plan.action_type.value,
                )
                action_plan = heuristic_plan
            else:
                logger.info(
                    "Structured plan accepted: action=%s status=%s params=%s",
                    action_plan.action_type.value,
                    action_plan.status.value,
                    {k: v for k, v in action_plan.parameters.items() if k != 'override_content'},
                )
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

        if action_plan.status == ActionStatus.FAILED:
            session_context.pop('pending_action', None)
            self.storage.update_session_context(session_id, session_context)
            response_text = action_plan.clarification_question or "Understood. Let me know if you need anything else."
            self.storage.store_chat_message(session_id, 'assistant', response_text)
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
                override_content = action_plan.parameters.get('override_content')
                content_type = action_plan.parameters.get('content', 'summary')
                content = override_content or session_data.get(content_type) or session_data.get('transcription') or ''
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

    def _available_actions_description(self) -> Dict[str, str]:
        """Describe actions for LLM planning prompts."""
        return {
            ActionType.TRANSCRIBE_VIDEO.value: "Transcribe the uploaded video when a video file is available.",
            ActionType.GENERATE_PDF.value: "Generate a PDF document from the summary, transcription, or custom text.",
            ActionType.ANSWER_QUESTION.value: "Answer the user's question using available transcription or summary context.",
        }

    def _build_action_plan_from_structured(
        self,
        payload: Optional[Dict[str, Any]],
        context_snapshot: Dict[str, Any],
    ) -> Optional[ActionPlan]:
        """Convert LLM structured payload into an ActionPlan instance with validation."""
        if not payload:
            return None

        action_value = payload.get('action_type')
        status_value = payload.get('status')

        try:
            action_type = ActionType(action_value)
        except Exception:
            logger.warning("Invalid action_type from LLM planner: %s", action_value)
            return None

        try:
            status = ActionStatus(status_value)
        except Exception:
            logger.warning("Invalid status from LLM planner: %s", status_value)
            return None

        parameters = payload.get('parameters') or {}
        if not isinstance(parameters, dict):
            logger.warning("LLM planner parameters invalid: %s", parameters)
            return None

        missing_params = payload.get('missing_params') or []
        if not isinstance(missing_params, list):
            logger.warning("LLM planner missing_params invalid: %s", missing_params)
            missing_params = []

        confidence = payload.get('confidence')
        if isinstance(confidence, (int, float)):
            parameters.setdefault('llm_confidence', float(confidence))

        clarification_message = payload.get('clarification_message')
        if clarification_message is not None and not isinstance(clarification_message, str):
            clarification_message = None

        if action_type == ActionType.TRANSCRIBE_VIDEO:
            # Enforce English transcription regardless of planner output.
            parameters['language'] = 'en'
            if 'language' in missing_params:
                missing_params = [item for item in missing_params if item != 'language']

        if status == ActionStatus.READY_TO_EXECUTE and action_type != ActionType.ANSWER_QUESTION:
            # Validate necessary context before executing tool.
            if action_type == ActionType.TRANSCRIBE_VIDEO and not context_snapshot.get('has_video'):
                logger.info("LLM requested transcription but no video present; deferring to rule-based planner.")
                return None
            if action_type == ActionType.GENERATE_PDF and not (
                context_snapshot.get('has_summary') or context_snapshot.get('has_transcription') or parameters.get('override_content')
            ):
                logger.info("LLM requested PDF without content; deferring to rule-based planner.")
                return None

        confirmation_message = payload.get('confirmation_message')
        if confirmation_message is not None and not isinstance(confirmation_message, str):
            confirmation_message = None

        logger.debug(
            "Validated structured plan: action=%s status=%s missing=%s",
            action_type.value,
            status.value,
            missing_params,
        )
        return ActionPlan(
            action_type=action_type,
            status=status,
            parameters=parameters,
            missing_params=missing_params,
            clarification_question=clarification_message,
            confirmation_message=confirmation_message,
        )
