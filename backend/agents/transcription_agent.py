import os
import io
import asyncio
import logging
from typing import Dict, Any
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from moviepy import VideoFileClip
import tempfile

logger = logging.getLogger(__name__)

class TranscriptionAgent:
    """Handles video-to-text transcription using Whisper model"""
    
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Initialize the transcription agent
        
        Args:
            model_name: Hugging Face model identifier for Whisper
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranscriptionAgent initialized with device: {self.device}")
    
    async def _load_model(self):
        """Lazy load the Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise
    
    async def transcribe_video(self, video_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Transcribe video to text
        
        Args:
            video_data: Raw video file bytes
            filename: Original filename for logging
            
        Returns:
            Dictionary with transcription result
        """
        try:
            await self._load_model()
            
            # Save video data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_data)
                temp_video_path = temp_video.name
            
            try:
                # Extract audio from video
                audio_path = await self._extract_audio(temp_video_path)
                
                # Load and preprocess audio
                audio_input = await self._preprocess_audio(audio_path)
                
                # Perform transcription
                transcription = await self._transcribe_audio(audio_input)
                
                # Cleanup temporary files
                os.unlink(temp_video_path)
                os.unlink(audio_path)
                
                logger.info(f"Successfully transcribed video: {filename}")
                return {
                    'transcription': transcription,
                    'success': True,
                    'filename': filename
                }
                
            except Exception as e:
                # Cleanup on error
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                raise e
                
        except Exception as e:
            logger.error(f"Transcription failed for {filename}: {str(e)}")
            return {
                'transcription': '',
                'success': False,
                'error_message': str(e),
                'filename': filename
            }
    
    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            # Create temporary file for audio
            audio_path = video_path.replace('.mp4', '.wav')
            
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                raise ValueError("No audio track found in video")
            
            # Save as WAV file
            audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Cleanup
            audio.close()
            video.close()
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise
    
    async def _preprocess_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Preprocess audio for Whisper model"""
        try:
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to numpy for processor
            audio_array = waveform.squeeze().numpy()
            
            # Process with Whisper processor
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise
    
    async def _transcribe_audio(self, audio_input: Dict[str, torch.Tensor]) -> str:
        """Perform transcription using Whisper model"""
        try:
            # Move inputs to device
            input_features = audio_input.input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=5,
                    temperature=0.0,
                    do_sample=False
                )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.model is not None
        }