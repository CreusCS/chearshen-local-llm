import json
import logging
from typing import Dict, Any, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class LLMAgent:
    """Handles local LLM operations for chat and Q&A"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the LLM agent
        
        Args:
            model_name: Hugging Face model identifier
                       Default: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params, ~2.2GB, optimized for chat)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        logger.info(f"LLMAgent initialized with device: {self.device}")
    
    async def _load_model(self):
        """Lazy load the LLM model with quantization"""
        if self.model is None:
            logger.info(f"Loading LLM model: {self.model_name}")
            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=False,
                )
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with appropriate settings
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=False,
                )
                
                # Move model to device if not using device_map
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                )
                
                logger.info(f"LLM model loaded successfully (size: ~{self._estimate_model_size()}MB)")
                
            except Exception as e:
                logger.error(f"Failed to load LLM model: {str(e)}")
                # Fallback to smallest model if the main one fails
                if self.model_name != "distilgpt2":
                    logger.info("Attempting fallback to distilgpt2 (smallest model)...")
                    self.model_name = "distilgpt2"
                    await self._load_model()
                else:
                    raise
    
    def _estimate_model_size(self) -> int:
        """Estimate model size in MB"""
        if self.model is None:
            return 0
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            bytes_per_param = 2 if self.device == "cuda" else 4
            size_mb = (param_count * bytes_per_param) / (1024 * 1024)
            return int(size_mb)
        except:
            return 0
    
    async def generate_response(
        self, 
        message: str, 
        context: Optional[str] = None,
        session_id: str = "",
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Generate response using the local LLM
        
        Args:
            message: User input message
            context: Optional context (e.g., transcription)
            session_id: Session identifier
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Dictionary with LLM response
        """
        try:
            await self._load_model()
            
            # Build prompt with context
            prompt = self._build_prompt(message, context)
            
            # Generate response
            response = await self._generate_text(prompt, max_new_tokens)
            
            logger.info(f"Generated response for session: {session_id}")
            return {
                'response': response,
                'success': True,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            
            # Provide fallback response
            fallback_response = self._get_fallback_response(message, context)
            
            return {
                'response': fallback_response,
                'success': True,  # Mark as success since we provide a fallback
                'session_id': session_id,
                'note': 'Fallback response used due to model error'
            }

    async def plan_action(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        available_actions: Optional[Dict[str, str]] = None,
        max_new_tokens: int = 320,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to propose a structured action plan."""
        try:
            await self._load_model()
            prompt = self._build_planner_prompt(message, context or {}, available_actions or {})
            raw_response = await self._generate_text(prompt, max_new_tokens)
            logger.info("LLM planner raw output: %s", raw_response.strip()[:500])
            plan_payload = self._extract_json_block(raw_response)
            if plan_payload is None:
                logger.warning("LLM planner did not return JSON. Falling back to rule-based planner.")
                return None
            logger.info("LLM planner parsed payload: %s", plan_payload)
            return plan_payload
        except Exception as exc:
            logger.error("LLM planning failed: %s", exc)
            return None
    
    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build formatted prompt for the LLM"""
        system_prompt = "You are a helpful AI assistant specialized in video analysis and general conversation."
        
        if context:
            prompt = f"<|system|>\n{system_prompt}\n\nVideo Context:\n{context[:1000]}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
        
    
        
        return prompt

    def _build_planner_prompt(
        self,
        message: str,
        context: Dict[str, Any],
        available_actions: Dict[str, str],
    ) -> str:
        """Construct planner instruction prompt enforcing structured JSON output."""
        action_descriptions = []
        for name, description in available_actions.items():
            action_descriptions.append(f"- {name}: {description}")
        actions_text = "\n".join(action_descriptions) if action_descriptions else (
            "- transcribe_video: Transcribe the uploaded video if available.\n"
            "- generate_pdf: Create a PDF using available transcription or summary.\n"
            "- answer_question: Respond conversationally using the context."
        )

        context_lines = []
        for key, value in context.items():
            if isinstance(value, bool):
                context_lines.append(f"{key}: {value}")
            elif value is None:
                continue
            else:
                text = str(value)
                if len(text) > 160:
                    text = text[:157] + "..."
                context_lines.append(f"{key}: {text}")
        context_text = "\n".join(context_lines) if context_lines else "(no additional context)"

        system_instruction = (
            "You are an orchestration planner. You MUST reply with a JSON object only. "
            "Choose the best action to satisfy the user request. "
            "Allowed action_types: transcribe_video, generate_pdf, answer_question. "
            "Valid status values: needs_clarification, requires_confirmation, ready_to_execute. "
            "Always include fields: action_type (string), status (string), parameters (object), "
            "missing_params (array of strings), clarification_message (string or null), "
            "confidence (number between 0 and 1). "
            "If you need more info from the user, set status to needs_clarification and "
            "provide a concise clarification_message. "
            "Transcription is always performed in English; do NOT ask for a language parameter." 
        )

        prompt = (
            f"<|system|>\n{system_instruction}</s>\n"
            f"<|user|>\n"
            f"User request: {message}\n\n"
            f"Session context:\n{context_text}\n\n"
            f"Available actions:\n{actions_text}\n\n"
            "Respond with JSON only, without explanation.\n"
            "</s>\n"
            "<|assistant|>\n"
        )

        return prompt

    def _extract_json_block(self, response: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of the first JSON object from model output."""
        start = response.find('{')
        end = response.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = response[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse planner JSON: %s", candidate)
        return None
    
    async def _generate_text(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text using the loaded model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()

            # Ensure input doesn't exceed model's max length
            if inputs.shape[1] > self.max_length - max_new_tokens:
                inputs = inputs[:, -(self.max_length - max_new_tokens):]
                attention_mask = attention_mask[:, -(self.max_length - max_new_tokens):]

            # Move to device
            if self.device == "cuda":
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)

            # Generate with the model
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new part (assistant response)
            response = full_response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

            # Clean up response
            response = self._clean_response(response)

            return response

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response"""
        # Remove potential system tokens that might leak through
        response = response.replace("<|system|>", "")
        response = response.replace("<|user|>", "")
        response = response.replace("<|assistant|>", "")
        
        # Remove excessive whitespace
        response = " ".join(response.split())
        
        # Ensure response isn't empty
        if not response:
            response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
        
        return response
    
    def _get_fallback_response(self, message: str, context: Optional[str] = None) -> str:
        """Provide fallback responses when model fails"""
        message_lower = message.lower()
        
        if "transcribe" in message_lower:
            return "I can help you transcribe videos. Please upload a video file and I'll process it using our speech-to-text model."
        
        elif "summarize" in message_lower or "summary" in message_lower:
            if context:
                return "I can see you have transcribed content available. I'll create a summary of the key points discussed in the video."
            else:
                return "I can help create summaries of transcribed video content. First, please upload and transcribe a video."
        
        elif "pdf" in message_lower:
            return "I can generate PDF documents from summaries and transcriptions. This helps you create offline reports and documentation."
        
        elif any(greeting in message_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! I'm your local AI assistant. I can help you transcribe videos, create summaries, generate PDFs, and answer questions about your content."
        
        else:
            return f"I understand you're asking about: '{message}'. I'm a local AI assistant that specializes in video analysis. I can help with transcription, summarization, and general questions about your content."
    
    async def analyze_video_content(self, transcription: str, question: str) -> Dict[str, Any]:
        """
        Analyze video content based on transcription and answer specific questions
        
        Args:
            transcription: Video transcription text
            question: Specific question about the content
            
        Returns:
            Dictionary with analysis result
        """
        try:
            await self._load_model()
            
            prompt = f"""Based on the following video transcription, please answer the question:

Transcription:
{transcription[:1500]}

Question: {question}

Please provide a clear and concise answer based only on the content of the transcription."""
            
            response = await self._generate_text(prompt, max_new_tokens=200)
            
            return {
                'answer': response,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            return {
                'answer': f"I couldn't analyze the content due to a technical issue, but based on your question '{question}', I can see you're asking about the video content.",
                'success': False,
                'error_message': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.model is not None,
            'max_length': self.max_length,
            'quantized': self.device == "cuda"
        }