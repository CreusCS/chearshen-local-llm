"""
Action Planner - Implements human-in-the-loop clarification for agentic workflows
Detects user intent, plans actions, and requests clarification when needed
"""

import logging
import re
from typing import Dict, Any, List, Optional
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
    
    def detect_action_type(self, message: str) -> ActionType:
        """
        Detect the type of action from user message
        
        Args:
            message: User's message
            
        Returns:
            Detected action type
        """
        message_lower = message.lower().strip()
        
        for action_type, patterns in self.action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    logger.info(f"Detected action type: {action_type.value}")
                    return action_type
        
        logger.info("No specific action detected, treating as general question")
        return ActionType.ANSWER_QUESTION
    
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
        action_type = self.detect_action_type(message)
        
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
                # Likely a title response
                if original_plan.missing_params and 'title' in original_plan.missing_params:
                    original_plan.parameters['title'] = user_response.strip('"\'')
                    original_plan.missing_params.remove('title')
                    
                    if not original_plan.missing_params:
                        # All parameters collected, ask for confirmation
                        original_plan.status = ActionStatus.REQUIRES_CONFIRMATION
                        original_plan.confirmation_message = (
                            f"I'll generate a PDF titled '{original_plan.parameters['title']}'. "
                            f"Proceed?"
                        )
                    
                    return original_plan
        
        # Default: keep original plan
        return original_plan
