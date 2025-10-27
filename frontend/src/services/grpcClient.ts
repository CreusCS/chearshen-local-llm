import { TranscriptionResult, SummaryResult, PDFResult, ChatResult } from '../types';

// Mock gRPC client for now - in a real implementation, this would use actual gRPC-web
// For this demo, we'll simulate the backend calls

export class VideoAnalysisService {
  private baseUrl = 'http://localhost:50051';

  async transcribeVideo(videoData: Uint8Array, filename: string): Promise<TranscriptionResult> {
    try {
      // Simulate API call delay
      await this.delay(2000);
      
      // For demo purposes, return a mock transcription
      // In real implementation, this would send videoData to the backend
      const mockTranscription = `This is a mock transcription for the video file: ${filename}. 
      In a real implementation, this would be the actual speech-to-text output from the Whisper model.
      The video content would be processed by the backend transcription agent using Hugging Face models.`;
      
      return {
        transcription: mockTranscription,
        success: true,
        sessionId: 'mock-session-id'
      };
    } catch (error) {
      return {
        transcription: '',
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        sessionId: ''
      };
    }
  }

  async summarizeTranscription(transcription: string, sessionId: string): Promise<SummaryResult> {
    try {
      await this.delay(1500);
      
      const mockSummary = `Summary of the transcribed content:
      
      This video discusses various topics related to AI and machine learning. 
      Key points include the importance of local processing, privacy considerations, 
      and the effectiveness of quantized models for desktop applications.
      
      The content demonstrates practical applications of AI technology in 
      real-world scenarios, emphasizing user-friendly interfaces and offline capabilities.`;
      
      return {
        summary: mockSummary,
        success: true
      };
    } catch (error) {
      return {
        summary: '',
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async generatePDF(content: string, title: string, sessionId: string): Promise<PDFResult> {
    try {
      await this.delay(1000);
      
      // Mock PDF generation - in real implementation, this would return actual PDF bytes
      const mockPdfData = new Uint8Array([0x25, 0x50, 0x44, 0x46]); // PDF header bytes
      
      return {
        pdfData: mockPdfData,
        success: true,
        filename: `${title.replace(/\s+/g, '_')}.pdf`
      };
    } catch (error) {
      return {
        pdfData: new Uint8Array(),
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        filename: ''
      };
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export class LLMChatService {
  private baseUrl = 'http://localhost:50051';

  async sendMessage(message: string, sessionId: string, context?: string): Promise<ChatResult> {
    try {
      await this.delay(1500);
      
      // Mock LLM response - in real implementation, this would use the local LLM
      let mockResponse = '';
      
      if (message.toLowerCase().includes('transcribe')) {
        mockResponse = "I can help you transcribe video files. Please upload a video file using the upload area above, and I'll process it using our local speech-to-text model.";
      } else if (message.toLowerCase().includes('summarize')) {
        mockResponse = "I can create summaries of transcribed content. Once you have a transcription, I can generate a concise summary and even create a PDF document for you.";
      } else if (message.toLowerCase().includes('pdf')) {
        mockResponse = "I can generate PDF documents from summaries and transcriptions. This is useful for creating reports or saving content for offline reading.";
      } else {
        mockResponse = `I'm a local AI assistant running on your machine. I can help with video transcription, summarization, PDF generation, and answer general questions. Your message was: "${message}"`;
      }
      
      return {
        response: mockResponse,
        success: true,
        sessionId
      };
    } catch (error) {
      return {
        response: '',
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        sessionId
      };
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}