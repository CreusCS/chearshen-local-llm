export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  sessionId: string;
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

export interface SummaryResult {
  summary: string;
  success: boolean;
  errorMessage?: string;
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
}