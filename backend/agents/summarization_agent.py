import logging
from typing import Dict, Any
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

class SummarizationAgent:
    """Handles text summarization using Hugging Face models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarization agent
        
        Args:
            model_name: Hugging Face model identifier for summarization
        """
        self.model_name = model_name
        self.summarizer = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"SummarizationAgent initialized with device: {self.device}")
    
    async def _load_model(self):
        """Lazy load the summarization model"""
        if self.summarizer is None:
            logger.info(f"Loading summarization model: {self.model_name}")
            try:
                # Load tokenizer and model separately for better control
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Create pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
                
                logger.info("Summarization model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load summarization model: {str(e)}")
                raise
    
    async def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> Dict[str, Any]:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Dictionary with summarization result
        """
        try:
            await self._load_model()
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Text is too short to summarize")
            
            # Split long texts into chunks if needed
            chunks = self._split_text(text)
            summaries = []
            
            for chunk in chunks:
                summary = await self._summarize_chunk(chunk, max_length, min_length)
                summaries.append(summary)
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_summary = " ".join(summaries)
                # Re-summarize if the combined summary is too long
                if len(combined_summary.split()) > max_length:
                    final_summary = await self._summarize_chunk(
                        combined_summary, 
                        max_length, 
                        min_length
                    )
                else:
                    final_summary = combined_summary
            else:
                final_summary = summaries[0]
            
            logger.info("Text summarization completed successfully")
            return {
                'summary': final_summary,
                'success': True,
                'original_length': len(text.split()),
                'summary_length': len(final_summary.split())
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {
                'summary': '',
                'success': False,
                'error_message': str(e)
            }
    
    async def _summarize_chunk(self, text: str, max_length: int, min_length: int) -> str:
        """Summarize a single chunk of text"""
        try:
            # Adjust lengths based on input text length
            input_length = len(text.split())
            adjusted_max = min(max_length, input_length // 2)
            adjusted_min = min(min_length, adjusted_max // 2)
            
            # Ensure min is less than max
            if adjusted_min >= adjusted_max:
                adjusted_min = max(1, adjusted_max - 10)
            
            result = self.summarizer(
                text,
                max_length=adjusted_max,
                min_length=adjusted_min,
                do_sample=False,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {str(e)}")
            raise
    
    def _split_text(self, text: str, max_chunk_size: int = 900) -> list:
        """
        Split text into chunks for processing
        
        Args:
            text: Input text to split
            max_chunk_size: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed max length, start new chunk
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        # If no splits occurred, return original text
        if not chunks:
            chunks = [text]
        
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    
    async def create_structured_summary(self, text: str) -> Dict[str, Any]:
        """
        Create a structured summary with key points
        
        Args:
            text: Input text to summarize
            
        Returns:
            Dictionary with structured summary
        """
        try:
            # Get basic summary
            basic_result = await self.summarize_text(text)
            
            if not basic_result['success']:
                return basic_result
            
            # Extract key points (simple implementation)
            sentences = text.split('. ')
            key_points = []
            
            # Take first few sentences and last few as key points
            if len(sentences) > 5:
                key_points.extend(sentences[:2])
                key_points.extend(sentences[-2:])
            else:
                key_points = sentences[:3]
            
            return {
                'summary': basic_result['summary'],
                'key_points': [point.strip() + '.' for point in key_points if point.strip()],
                'success': True,
                'original_length': basic_result['original_length'],
                'summary_length': basic_result['summary_length']
            }
            
        except Exception as e:
            logger.error(f"Structured summary failed: {str(e)}")
            return {
                'summary': '',
                'key_points': [],
                'success': False,
                'error_message': str(e)
            }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.summarizer is not None
        }