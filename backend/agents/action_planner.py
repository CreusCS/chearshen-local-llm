"""
Action Planner - Implements human-in-the-loop clarification for agentic workflows
Detects user intent, plans actions, and requests clarification when needed
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions the system can perform"""
    TRANSCRIBE_VIDEO = "transcribe_video"
    GENERATE_PDF = "generate_pdf"
    ANSWER_QUESTION = "answer_question"
    UNKNOWN = "unknown"

class ActionStatus(Enum):
    """Status of an action"""
    NEEDS_CLARIFICATION = "needs_clarification"
    READY_TO_EXECUTE = "ready_to_execute"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class ActionPlan:
    """Represents a planned action with its parameters and status"""
    
    def __init__(
        self,
        action_type: ActionType,
        status: ActionStatus,
        parameters: Dict[str, Any] = None,
        missing_params: List[str] = None,
        clarification_question: str = None,
        confirmation_message: str = None
    ):
        self.action_type = action_type
        self.status = status
        self.parameters = parameters or {}
        self.missing_params = missing_params or []
        self.clarification_question = clarification_question
        self.confirmation_message = confirmation_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action plan to dictionary for JSON serialization"""
        return {
            "action_type": self.action_type.value,
            "status": self.status.value,
            "parameters": self.parameters,
            "missing_params": self.missing_params,
            "clarification_question": self.clarification_question,
            "confirmation_message": self.confirmation_message,
            "requires_user_input": self.status in [
                ActionStatus.NEEDS_CLARIFICATION, 
                ActionStatus.REQUIRES_CONFIRMATION
            ]
        }

class ActionPlanner:
    """Plans and validates actions based on user input"""
    
    def __init__(self):
        """Initialize the action planner"""
        self.action_patterns = {
            ActionType.TRANSCRIBE_VIDEO: [
                r"transcribe(?:\s+the)?\s+video",
                r"convert\s+(?:the\s+)?video\s+to\s+text",
                r"speech\s+to\s+text",
                r"extract\s+(?:the\s+)?audio",
            ],
            ActionType.GENERATE_PDF: [
                r"(?:generate|create|make|save)\s+(?:a\s+)?pdf",
                r"export\s+(?:to\s+)?pdf",
                r"save\s+(?:as|to)\s+pdf",
                r"pdf\s+(?:report|document)",
            ],
            ActionType.ANSWER_QUESTION: [
                r"^(?:what|when|where|who|why|how)",
                r"(?:can|could)\s+you\s+(?:tell|explain)",
                r"(?:please\s+)?(?:tell|explain)",
            ]
        }
        self.action_thresholds = {
            ActionType.TRANSCRIBE_VIDEO: 0.55,
            ActionType.GENERATE_PDF: 0.55,
            ActionType.ANSWER_QUESTION: 0.0,
        }
    
    def detect_action_type(self, message: str) -> Tuple[ActionType, float]:
        """
        Detect the type of action from user message
        
        Args:
            message: User's message
            
        Returns:
            Detected action type
        """
        message_lower = message.lower().strip()
        
        best_action = ActionType.ANSWER_QUESTION
        best_score = 0.0

        for action_type, patterns in self.action_patterns.items():
            matches: List[re.Match] = []
            unique_patterns = set()
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, message_lower))
                if pattern_matches:
                    matches.extend(pattern_matches)
                    unique_patterns.add(pattern)

            if matches:
                coverage = sum(len(match.group(0)) for match in matches) / max(len(message_lower), 1)
                diversity = len(unique_patterns) / len(patterns)
                score = min(1.0, 0.35 + 0.45 * coverage + 0.2 * diversity)
                logger.debug("Action %s scored %.2f", action_type.value, score)
                if score > best_score:
                    best_score = score
                    best_action = action_type

        if best_action == ActionType.ANSWER_QUESTION and message_lower.endswith("?"):
            best_score = max(best_score, 0.6)

        logger.info("Detected action type %s with confidence %.2f", best_action.value, best_score)
        return best_action, best_score
    
    def plan_action(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> ActionPlan:
        """
        Create an action plan based on user message and context
        
        Args:
            message: User's message
            context: Current context (video, transcription, etc.)
            
        Returns:
            ActionPlan with status and required parameters
        """
        action_type, confidence = self.detect_action_type(message)

        if action_type != ActionType.ANSWER_QUESTION:
            threshold = self.action_thresholds.get(action_type, 0.5)
            if confidence < threshold:
                return ActionPlan(
                    action_type=ActionType.UNKNOWN,
                    status=ActionStatus.NEEDS_CLARIFICATION,
                    parameters={
                        'candidate_action': action_type.value,
                        'original_message': message,
                        'context_snapshot': self._context_snapshot(context),
                        'confidence': confidence,
                    },
                    missing_params=['intent'],
                    clarification_question=self._build_disambiguation_prompt(action_type),
                )
        
        if action_type == ActionType.TRANSCRIBE_VIDEO:
            return self._plan_transcription(message, context)
        elif action_type == ActionType.GENERATE_PDF:
            return self._plan_pdf_generation(message, context)
        else:
            return self._plan_question_answer(message, context)
    
    def _plan_transcription(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> ActionPlan:
        """Plan video transcription action"""
        
        # Check if video is uploaded
        if not context.get('has_video'):
            return ActionPlan(
                action_type=ActionType.TRANSCRIBE_VIDEO,
                status=ActionStatus.NEEDS_CLARIFICATION,
                clarification_question=(
                    "I'd be happy to transcribe a video for you! "
                    "Please upload a video file using the upload area above."
                )
            )
        
        # Video is available, ask for confirmation
        video_filename = context.get('video_filename', 'the video')
        return ActionPlan(
            action_type=ActionType.TRANSCRIBE_VIDEO,
            status=ActionStatus.REQUIRES_CONFIRMATION,
            parameters={
                'video_filename': video_filename
            },
            confirmation_message=(
                f"I'll transcribe '{video_filename}'. "
                f"This may take a few minutes depending on the video length. "
                f"Proceed?"
            )
        )
    
    def _plan_pdf_generation(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> ActionPlan:
        """Plan PDF generation action"""
        
        # Check if we have content to generate PDF from
        has_summary = context.get('has_summary')
        has_transcription = context.get('has_transcription')
        
        if not has_summary and not has_transcription:
            return ActionPlan(
                action_type=ActionType.GENERATE_PDF,
                status=ActionStatus.NEEDS_CLARIFICATION,
                parameters={
                    'content_source': None,
                    'title': None,
                },
                missing_params=['content_source'],
                clarification_question=(
                    "I need content to create a PDF. What would you like to include?\n"
                    "1. Upload and transcribe a video first, or\n"
                    "2. Create a summary of existing content, or\n"
                    "3. Provide text content to include in the PDF?"
                )
            )
        
        # Extract PDF parameters
        pdf_params = self._extract_pdf_params(message, context)
        
        # Ask for missing parameters
        missing_params = []
        if not pdf_params.get('title'):
            missing_params.append('title')
        
        if missing_params:
            return ActionPlan(
                action_type=ActionType.GENERATE_PDF,
                status=ActionStatus.NEEDS_CLARIFICATION,
                parameters=pdf_params,
                missing_params=missing_params,
                clarification_question=(
                    "I'll generate a PDF for you. What would you like to title the document?"
                )
            )
        
        # All parameters available, ask for confirmation
        content_type = "summary" if has_summary else "transcription"
        return ActionPlan(
            action_type=ActionType.GENERATE_PDF,
            status=ActionStatus.REQUIRES_CONFIRMATION,
            parameters=pdf_params,
            confirmation_message=(
                f"I'll generate a PDF titled '{pdf_params['title']}' "
                f"containing the {content_type}. Proceed?"
            )
        )
    
    def _plan_question_answer(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> ActionPlan:
        """Plan question answering (no confirmation needed)"""
        
        return ActionPlan(
            action_type=ActionType.ANSWER_QUESTION,
            status=ActionStatus.READY_TO_EXECUTE,
            parameters={
                'question': message,
                'context': context.get('transcription')
            }
        )
    
    def _extract_summary_params(self, message: str) -> Dict[str, Any]:
        """Extract summary parameters from message"""
        params = {}
        
        message_lower = message.lower()
        
        # Detect summary length
        if any(word in message_lower for word in ['brief', 'short', 'quick']):
            params['length'] = 'brief'
        elif any(word in message_lower for word in ['detailed', 'long', 'comprehensive']):
            params['length'] = 'detailed'
        else:
            params['length'] = 'standard'
        
        # Detect bullet points preference
        if any(word in message_lower for word in ['bullet', 'points', 'list']):
            params['format'] = 'bullets'
        else:
            params['format'] = 'paragraph'
        
        return params
    
    def _extract_pdf_params(
        self, 
        message: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract PDF parameters from message"""
        params = {}
        
        # Try to extract title from message
        # Look for patterns like "save as X", "title: X", etc.
        title_patterns = [
            r'(?:save|title|name|call)\s+(?:as|it)\s+["\']?([^"\']+)["\']?',
            r'["\']([^"\']+)["\']',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                params['title'] = match.group(1).strip()
                break
        
        # Default title based on context
        if not params.get('title'):
            if context.get('video_filename'):
                base_name = context['video_filename'].rsplit('.', 1)[0]
                params['title'] = f"{base_name}_summary"
            else:
                params['title'] = None  # Will trigger clarification
        
        # Determine content
        if context.get('has_summary'):
            params['content'] = 'summary'
        elif context.get('has_transcription'):
            params['content'] = 'transcription'
        
        return params

    def _context_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        keys_to_keep = [
            'has_video',
            'video_filename',
            'has_transcription',
            'has_summary',
            'transcription',
            'summary',
        ]
        return {key: context.get(key) for key in keys_to_keep if key in context}

    def _build_disambiguation_prompt(self, action_type: ActionType) -> str:
        descriptions = {
            ActionType.TRANSCRIBE_VIDEO: "transcribe your video",
            ActionType.GENERATE_PDF: "generate a PDF report",
            ActionType.ANSWER_QUESTION: "answer a question about your content",
        }
        verb = descriptions.get(action_type, "help out")
        return (
            f"It sounds like you might want me to {verb}, but I'm not completely sure. "
            "Would you like me to do that, or is there something else you had in mind?"
        )
    
    def process_clarification_response(
        self,
        original_plan: ActionPlan,
        user_response: str
    ) -> ActionPlan:
        """
        Process user's response to clarification question
        
        Args:
            original_plan: The original action plan that needed clarification
            user_response: User's response to the clarification
            
        Returns:
            Updated action plan
        """
        response_lower = user_response.lower().strip()
        
        # Check for affirmative responses (yes, proceed, ok, etc.)
        affirmative = any(word in response_lower for word in [
            'yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed', 'go ahead', 'do it'
        ])
        
        # Check for negative responses
        negative = any(word in response_lower for word in [
            'no', 'nope', 'cancel', 'stop', 'don\'t', 'nevermind'
        ])
        
        if original_plan.status == ActionStatus.REQUIRES_CONFIRMATION:
            if affirmative:
                # User confirmed, mark as ready to execute
                original_plan.status = ActionStatus.READY_TO_EXECUTE
                return original_plan
            elif negative:
                # User cancelled
                original_plan.status = ActionStatus.FAILED
                original_plan.clarification_question = "Action cancelled. What else can I help you with?"
                original_plan.parameters.pop('pending_confirmation', None)
                return original_plan
        
        elif original_plan.status == ActionStatus.NEEDS_CLARIFICATION:
            # Process the clarification response based on action type
            
            # For transcription without video - user is answering
            if original_plan.action_type == ActionType.TRANSCRIBE_VIDEO:
                # User is responding to the upload prompt
                # Treat this as acknowledgment, cancel pending action
                original_plan.status = ActionStatus.FAILED
                original_plan.clarification_question = "Please upload a video file using the upload area, then I can transcribe it for you."
                return original_plan
            
            # For PDF generation - collect missing parameters
            elif original_plan.action_type == ActionType.GENERATE_PDF:
                if original_plan.missing_params and 'content_source' in original_plan.missing_params:
                    choice_text = response_lower.strip()
                    if choice_text.startswith('1') or 'upload' in choice_text:
                        original_plan.status = ActionStatus.FAILED
                        original_plan.missing_params.clear()
                        original_plan.clarification_question = (
                            "No problem. Please upload and transcribe a video first, then ask me to generate the PDF again."
                        )
                        return original_plan
                    if choice_text.startswith('2') or 'summary' in choice_text:
                        original_plan.status = ActionStatus.FAILED
                        original_plan.missing_params.clear()
                        original_plan.clarification_question = (
                            "Happy to help once we have a summary. Ask me to summarize your transcription or provide the text you'd like summarized, then we can make a PDF."
                        )
                        return original_plan
                    if choice_text.startswith('3') or 'text' in choice_text or 'provide' in choice_text:
                        original_plan.parameters['content_source'] = 'custom_text'
                        original_plan.missing_params = ['content_text']
                        original_plan.clarification_question = "Great! Please paste the text you'd like included in the PDF."
                        return original_plan

                    original_plan.clarification_question = (
                        "I didn't catch that. Please respond with 1, 2, or 3 so I know which option you prefer."
                    )
                    return original_plan

                if original_plan.parameters.get('content_source') == 'custom_text' and original_plan.missing_params and 'content_text' in original_plan.missing_params:
                    if negative:
                        original_plan.status = ActionStatus.FAILED
                        original_plan.missing_params.clear()
                        original_plan.clarification_question = "Okay, I'll wait. Let me know when you have the text ready."
                        return original_plan

                    original_plan.parameters['override_content'] = user_response.strip()
                    original_plan.missing_params.remove('content_text')

                    if not original_plan.parameters.get('title'):
                        if 'title' not in original_plan.missing_params:
                            original_plan.missing_params.append('title')
                        original_plan.clarification_question = "What title should I use for the PDF?"
                        return original_plan

                    original_plan.status = ActionStatus.REQUIRES_CONFIRMATION
                    original_plan.parameters['content'] = 'custom_text'
                    original_plan.confirmation_message = (
                        f"I'll generate a PDF titled '{original_plan.parameters['title']}' with the text you provided. Proceed?"
                    )
                    original_plan.parameters['pending_confirmation'] = True
                    original_plan.clarification_question = None
                    return original_plan

                # Likely a title response
                if original_plan.missing_params and 'title' in original_plan.missing_params:
                    if negative:
                        original_plan.status = ActionStatus.FAILED
                        original_plan.clarification_question = "Okay, I'll hold off on generating a PDF. Let me know when you're ready."
                        original_plan.missing_params.clear()
                        return original_plan

                    original_plan.parameters['title'] = user_response.strip('"\'')
                    original_plan.missing_params.remove('title')

                    if original_plan.parameters.get('content_source') == 'custom_text' and not original_plan.parameters.get('override_content'):
                        original_plan.parameters['content'] = 'custom_text'
                    
                    if not original_plan.missing_params:
                        # All parameters collected, ask for confirmation
                        original_plan.status = ActionStatus.REQUIRES_CONFIRMATION
                        original_plan.confirmation_message = (
                            f"I'll generate a PDF titled '{original_plan.parameters['title']}'. "
                            f"Proceed?"
                        )
                        original_plan.parameters['pending_confirmation'] = True
                        original_plan.clarification_question = None
                    
                    return original_plan

            elif original_plan.action_type == ActionType.UNKNOWN:
                candidate_value = original_plan.parameters.get('candidate_action')
                context_snapshot = original_plan.parameters.get('context_snapshot') or {}
                original_message = original_plan.parameters.get('original_message', user_response)

                if candidate_value:
                    candidate_action = ActionType(candidate_value)

                    if affirmative:
                        if candidate_action == ActionType.TRANSCRIBE_VIDEO:
                            return self._plan_transcription(original_message, context_snapshot)
                        if candidate_action == ActionType.GENERATE_PDF:
                            return self._plan_pdf_generation(original_message, context_snapshot)
                        if candidate_action == ActionType.ANSWER_QUESTION:
                            return self._plan_question_answer(original_message, context_snapshot)

                    if negative:
                        original_plan.status = ActionStatus.FAILED
                        original_plan.clarification_question = "No problem. What would you like me to help with instead?"
                        return original_plan

                # Treat the response as a new instruction and re-plan with available context
                return self.plan_action(user_response, context_snapshot)
        
        # Default: keep original plan
        return original_plan
