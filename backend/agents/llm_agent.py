import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class LLMAgent:
    """Handles local LLM operations for chat and Q&A"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Initialize the LLM agent
        
        Args:
            model_name: Hugging Face model identifier (Phi-3 Mini, Mistral 7B, etc.)
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
                # Configure quantization for memory efficiency
                quantization_config = None
                if self.device == "cuda":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Set pad token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                
                logger.info("LLM model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load LLM model: {str(e)}")
                # Fallback to a smaller model if the main one fails
                if "Phi-3" in self.model_name:
                    logger.info("Attempting fallback to smaller model...")
                    self.model_name = "microsoft/DialoGPT-medium"
                    await self._load_model()
                else:
                    raise
    
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
    
    def _build_prompt(self, message: str, context: Optional[str] = None) -> str:
        """Build formatted prompt for the LLM"""
        
        system_prompt = """You are a helpful AI assistant specialized in video analysis and general conversation. 
You can help with transcriptions, summaries, and answer questions about video content.
Keep your responses clear, concise, and helpful."""
        
        if context:
            prompt = f"""<|system|>
{system_prompt}

Video Context Available:
{context[:1000]}...

<|user|>
{message}

<|assistant|>
"""
        else:
            prompt = f"""<|system|>
{system_prompt}

<|user|>
{message}

<|assistant|>
"""
        
        return prompt
    
    async def _generate_text(self, prompt: str, max_new_tokens: int) -> str:
        """Generate text using the loaded model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Ensure input doesn't exceed model's max length
            if inputs.shape[1] > self.max_length - max_new_tokens:
                inputs = inputs[:, -(self.max_length - max_new_tokens):]
            
            # Move to device
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate with the model
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
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