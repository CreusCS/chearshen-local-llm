import asyncio
import logging
from concurrent import futures
import grpc
from grpc_reflection.v1alpha import reflection

# Import generated protobuf files (these would be generated from ai_service.proto)
# For this demo, we'll create mock implementations
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.llm_agent import LLMAgent
from utils.storage import ChatStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalysisServicer:
    """Handles video analysis operations"""
    
    def __init__(self):
        self.transcription_agent = TranscriptionAgent()
        self.summarization_agent = SummarizationAgent()
        self.storage = ChatStorage()
    
    async def TranscribeVideo(self, request, context):
        """Transcribe video to text"""
        try:
            logger.info(f"Transcribing video: {request.filename}")
            
            # Process video data through transcription agent
            result = await self.transcription_agent.transcribe_video(
                request.video_data, 
                request.filename
            )
            
            # Store session data
            session_id = self.storage.create_session()
            self.storage.store_transcription(session_id, result['transcription'])
            
            return {
                'transcription': result['transcription'],
                'success': result['success'],
                'error_message': result.get('error_message', ''),
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                'transcription': '',
                'success': False,
                'error_message': str(e),
                'session_id': ''
            }
    
    async def SummarizeTranscription(self, request, context):
        """Summarize transcribed text"""
        try:
            logger.info(f"Summarizing transcription for session: {request.session_id}")
            
            result = await self.summarization_agent.summarize_text(
                request.transcription
            )
            
            # Store summary
            self.storage.store_summary(request.session_id, result['summary'])
            
            return {
                'summary': result['summary'],
                'success': result['success'],
                'error_message': result.get('error_message', '')
            }
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            return {
                'summary': '',
                'success': False,
                'error_message': str(e)
            }
    
    async def GeneratePDF(self, request, context):
        """Generate PDF from content"""
        try:
            logger.info(f"Generating PDF: {request.title}")
            
            from utils.pdf_generator import PDFGenerator
            pdf_generator = PDFGenerator()
            
            pdf_data = pdf_generator.create_pdf(
                content=request.content,
                title=request.title
            )
            
            filename = f"{request.title.replace(' ', '_')}.pdf"
            
            return {
                'pdf_data': pdf_data,
                'success': True,
                'error_message': '',
                'filename': filename
            }
            
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            return {
                'pdf_data': b'',
                'success': False,
                'error_message': str(e),
                'filename': ''
            }

class LLMChatServicer:
    """Handles LLM chat operations"""
    
    def __init__(self):
        self.llm_agent = LLMAgent()
        self.storage = ChatStorage()
    
    async def Chat(self, request, context):
        """Handle chat messages"""
        try:
            logger.info(f"Processing chat message for session: {request.session_id}")
            
            # Store user message
            self.storage.store_chat_message(
                request.session_id,
                'user',
                request.message
            )
            
            # Get response from LLM
            response = await self.llm_agent.generate_response(
                message=request.message,
                context=request.context,
                session_id=request.session_id
            )
            
            # Store assistant response
            self.storage.store_chat_message(
                request.session_id,
                'assistant',
                response['response']
            )
            
            return {
                'response': response['response'],
                'success': response['success'],
                'error_message': response.get('error_message', ''),
                'session_id': request.session_id
            }
            
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {
                'response': '',
                'success': False,
                'error_message': str(e),
                'session_id': request.session_id
            }
    
    async def GetChatHistory(self, request, context):
        """Retrieve chat history"""
        try:
            messages = self.storage.get_chat_history(
                request.session_id,
                limit=request.limit if request.limit > 0 else None
            )
            
            return {
                'messages': messages,
                'success': True,
                'error_message': ''
            }
            
        except Exception as e:
            logger.error(f"Chat history error: {str(e)}")
            return {
                'messages': [],
                'success': False,
                'error_message': str(e)
            }
    
    async def ClearHistory(self, request, context):
        """Clear chat history"""
        try:
            self.storage.clear_chat_history(request.session_id)
            
            return {
                'success': True,
                'error_message': ''
            }
            
        except Exception as e:
            logger.error(f"Clear history error: {str(e)}")
            return {
                'success': False,
                'error_message': str(e)
            }

async def serve():
    """Start the gRPC server"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add services
    video_servicer = VideoAnalysisServicer()
    chat_servicer = LLMChatServicer()
    
    # Note: In a real implementation, these would be generated from protobuf
    # For now, we'll simulate the service registration
    logger.info("Services registered")
    
    # Enable reflection for debugging
    SERVICE_NAMES = (
        'ai_service.VideoAnalysisService',
        'ai_service.LLMChatService',
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Configure server
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop(5)

if __name__ == '__main__':
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("\nServer interrupted by user")