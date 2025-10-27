import { invoke } from '@tauri-apps/api/tauri';
import { TranscriptionResult, PDFResult, ChatResult, ChatHistoryResponse, SessionContextResponse, ChatMessage, ActionPlan } from '../types';

type TranscriptionCommandResult = {
  success: boolean;
  sessionId: string;
  transcription: string;
  errorMessage?: string;
  videoFilename?: string;
};

type ChatCommandResult = {
  success: boolean;
  response: string;
  errorMessage?: string;
  sessionId: string;
  actionPlan?: ActionPlan;
};

type ChatHistoryCommandResult = {
  success: boolean;
  messages: Array<{
    id: string;
    role: string;
    content: string;
    timestamp: string;
  }>;
};

type SessionContextCommandResult = {
  success: boolean;
  errorMessage?: string;
  session?: {
    session_id: string;
    video_filename?: string | null;
    transcription?: string | null;
    summary?: string | null;
    created_at?: string | null;
    updated_at?: string | null;
  };
};

type PdfCommandResult = {
  success: boolean;
  pdfData: number[];
  filename: string;
  errorMessage?: string;
};

export class VideoAnalysisService {
  async transcribeVideoFromPath(filePath: string, sessionId: string): Promise<TranscriptionResult> {
    try {
      const result = await invoke<TranscriptionCommandResult>('transcribe_video', {
        sessionId,
        filePath,
      });

      return {
        transcription: result.transcription,
        success: result.success,
        errorMessage: result.errorMessage || '',
        sessionId: result.sessionId,
      };
    } catch (error) {
      console.error('Transcription error:', error);
      return {
        transcription: '',
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        sessionId,
      };
    }
  }

  async generatePDF(content: string, title: string, sessionId: string): Promise<PDFResult> {
    try {
      const result = await invoke<PdfCommandResult>('generate_pdf', {
        sessionId,
        content,
        title,
      });

      if (!result.success) {
        throw new Error(result.errorMessage || 'Failed to generate PDF');
      }

      const pdfData = new Uint8Array(result.pdfData);
      const blob = new Blob([pdfData], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = result.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      return {
        pdfData,
        success: true,
        filename: result.filename,
      };
    } catch (error) {
      console.error('PDF generation error:', error);
      return {
        pdfData: new Uint8Array(),
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        filename: '',
      };
    }
  }
}

export class LLMChatService {
  async sendMessage(message: string, sessionId: string, context?: string): Promise<ChatResult> {
    try {
      const result = await invoke<ChatCommandResult>('send_chat_message', {
        sessionId,
        message,
        context: context || null,
      });

      return {
        response: result.response,
        success: result.success,
        errorMessage: result.errorMessage || '',
        sessionId: result.sessionId,
        actionPlan: result.actionPlan,
      };
    } catch (error) {
      console.error('Chat error:', error);
      return {
        response: '',
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        sessionId,
      };
    }
  }

  async getChatHistory(sessionId: string, limit: number = 50): Promise<ChatHistoryResponse> {
    try {
      const result = await invoke<ChatHistoryCommandResult>('get_chat_history', {
        sessionId,
        limit,
      });

      const messages: ChatMessage[] = (result.messages || []).map((msg) => ({
        id: msg.id,
        role: (msg.role as ChatMessage['role']) || 'assistant',
        content: msg.content,
        timestamp: msg.timestamp ? new Date(msg.timestamp).getTime() : Date.now(),
        sessionId,
      }));

      return {
        success: result.success,
        messages,
        errorMessage: '',
      };
    } catch (error) {
      console.error('Chat history error:', error);
      return {
        messages: [],
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async getSessionContext(sessionId: string): Promise<SessionContextResponse> {
    try {
      const result = await invoke<SessionContextCommandResult>('get_session_context', {
        sessionId,
      });

      return {
        success: result.success,
        errorMessage: result.errorMessage,
        session: result.session,
      };
    } catch (error) {
      console.error('Get session context error:', error);
      return {
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async clearHistory(sessionId: string): Promise<{ success: boolean; errorMessage?: string }> {
    try {
      await invoke('clear_chat_history', { sessionId });
      return { success: true };
    } catch (error) {
      console.error('Clear history error:', error);
      return {
        success: false,
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}