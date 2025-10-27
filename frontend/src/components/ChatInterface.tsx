import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { LLMChatService, VideoAnalysisService } from '../services/grpcClient';
import { v4 as uuidv4 } from 'uuid';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: ChatMessage) => void;
  sessionId: string;
  currentVideo: string | null;
  currentTranscription: string | null;
  onClearHistory: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  sessionId,
  currentVideo,
  currentTranscription,
  onClearHistory
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatService = new LLMChatService();
  const videoService = new VideoAnalysisService();

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: uuidv4(),
      role: 'user',
      content: inputMessage,
      timestamp: Date.now(),
      sessionId
    };

    onSendMessage(userMessage);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Check for special commands
      if (inputMessage.toLowerCase().includes('summarize') && currentTranscription) {
        await handleSummarizeCommand(inputMessage);
      } else if (inputMessage.toLowerCase().includes('generate pdf') && currentTranscription) {
        await handleGeneratePDFCommand(inputMessage);
      } else {
        // Regular chat
        const result = await chatService.sendMessage(
          inputMessage,
          sessionId,
          currentTranscription || undefined
        );

        if (result.success) {
          const assistantMessage: ChatMessage = {
            id: uuidv4(),
            role: 'assistant',
            content: result.response,
            timestamp: Date.now(),
            sessionId
          };
          onSendMessage(assistantMessage);
        } else {
          throw new Error(result.errorMessage || 'Failed to get response');
        }
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
        sessionId
      };
      onSendMessage(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSummarizeCommand = async (message: string) => {
    if (!currentTranscription) {
      throw new Error('No transcription available to summarize');
    }

    const result = await videoService.summarizeTranscription(currentTranscription, sessionId);
    
    if (result.success) {
      const summaryMessage: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: `Here's a summary of the video transcription:\n\n${result.summary}`,
        timestamp: Date.now(),
        sessionId
      };
      onSendMessage(summaryMessage);
    } else {
      throw new Error(result.errorMessage || 'Failed to generate summary');
    }
  };

  const handleGeneratePDFCommand = async (message: string) => {
    if (!currentTranscription) {
      throw new Error('No content available to generate PDF');
    }

    // First get a summary
    const summaryResult = await videoService.summarizeTranscription(currentTranscription, sessionId);
    
    if (!summaryResult.success) {
      throw new Error(summaryResult.errorMessage || 'Failed to generate summary for PDF');
    }

    // Then generate PDF
    const pdfResult = await videoService.generatePDF(
      summaryResult.summary,
      currentVideo || 'Video Summary',
      sessionId
    );

    if (pdfResult.success) {
      const pdfMessage: ChatMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: `PDF generated successfully! The document "${pdfResult.filename}" has been created with the video summary.`,
        timestamp: Date.now(),
        sessionId
      };
      onSendMessage(pdfMessage);
    } else {
      throw new Error(pdfResult.errorMessage || 'Failed to generate PDF');
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h3>💬 AI Assistant</h3>
        <div className="chat-actions">
          {currentVideo && (
            <span className="current-video">📹 {currentVideo}</span>
          )}
          <button onClick={onClearHistory} className="clear-btn">
            🗑️ Clear History
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h4>👋 Welcome to AI Video Analyzer!</h4>
            <p>I can help you with:</p>
            <ul>
              <li>📹 Transcribing uploaded videos</li>
              <li>📝 Summarizing transcriptions</li>
              <li>📄 Generating PDF reports</li>
              <li>💡 Answering questions about the content</li>
            </ul>
            <p>Try commands like:</p>
            <div className="example-commands">
              <span>"Summarize the video"</span>
              <span>"Generate PDF"</span>
              <span>"What was discussed in the video?"</span>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="message-header">
                <span className="role">
                  {message.role === 'user' ? '👤' : message.role === 'assistant' ? '🤖' : '🔧'}
                  {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
                </span>
                <span className="timestamp">{formatTimestamp(message.timestamp)}</span>
              </div>
              <div className="message-content">
                {message.content}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="message-header">
              <span className="role">🤖 Assistant</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span>●</span>
                <span>●</span>
                <span>●</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <div className="input-container">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about the video or chat with me..."
            rows={2}
            disabled={isLoading}
          />
          <button 
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="send-btn"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
        </div>
        
        {currentVideo && (
          <div className="quick-actions">
            <button 
              onClick={() => setInputMessage('Summarize the video')}
              disabled={isLoading}
              className="quick-btn"
            >
              📝 Summarize
            </button>
            <button 
              onClick={() => setInputMessage('Generate PDF')}
              disabled={isLoading}
              className="quick-btn"
            >
              📄 Generate PDF
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;