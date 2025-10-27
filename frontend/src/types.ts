export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  sessionId: string;
  actionPlan?: ActionPlan;
}

export interface ActionPlan {
  action_type: string;
  status: string;
  parameters?: Record<string, any>;
  missing_params?: string[];
  clarification_question?: string;
  confirmation_message?: string;
  requires_user_input: boolean;
}

export interface VideoInfo {
  filename: string;
  size: number;
  duration?: number;
}

export interface TranscriptionResult {
  transcription: string;
  success: boolean;
  errorMessage?: string;
  sessionId: string;
}

export interface PDFResult {
  pdfData: Uint8Array;
  success: boolean;
  errorMessage?: string;
  filename: string;
}

export interface ChatResult {
  response: string;
  success: boolean;
  errorMessage?: string;
  sessionId: string;
  actionPlan?: ActionPlan;
}

export interface ChatHistoryResponse {
  success: boolean;
  messages: ChatMessage[];
  errorMessage?: string;
}

export interface SessionContextResponse {
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
}