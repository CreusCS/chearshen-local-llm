import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { LLMChatService } from '../services/grpcClient';
import { v4 as uuidv4 } from 'uuid';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: ChatMessage) => void;
  sessionId: string;
  currentVideo: string | null;
  currentTranscription: string | null;
  onClearHistory: () => void;
  onNewSession: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  sessionId,
  currentVideo,
  currentTranscription,
  onClearHistory,
  onNewSession
}) => {
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatService = new LLMChatService();

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
      const isFirstMessage = messages.length === 0;

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
          sessionId,
          actionPlan: result.actionPlan
        };
        onSendMessage(assistantMessage);
      } else {
        let errorMsg = result.errorMessage || 'Failed to get response';
        if (isFirstMessage && errorMsg.includes('timeout')) {
          errorMsg += '\n\nNote: The first message may take 10-30 seconds while the AI model loads. Please try again.';
        }
        throw new Error(errorMsg);
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

  const handleQuickReply = (reply: string) => {
    setInputMessage(reply);
    // Auto-send after a brief delay
    setTimeout(() => {
      const sendBtn = document.querySelector('.send-btn') as HTMLButtonElement;
      if (sendBtn) sendBtn.click();
    }, 100);
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
          <button onClick={onNewSession} className="clear-btn" style={{ marginLeft: '5px' }}>
            ✨ New Session
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
            <div className="info-note" style={{ marginTop: '20px', padding: '10px', background: '#f0f8ff', borderRadius: '5px', fontSize: '0.9em' }}>
              ⏱️ <strong>First-time note:</strong> Your first message may take 10-30 seconds while the AI model loads. Subsequent messages will be faster (2-5 seconds)!
            </div>
            <div className="info-note" style={{ marginTop: '10px', padding: '10px', background: '#f0fff0', borderRadius: '5px', fontSize: '0.9em' }}>
              💾 <strong>Persistent sessions:</strong> Your chat history is automatically saved. Close and reopen the app to see your messages restored!
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
              
              {/* Action confirmation buttons */}
              {message.role === 'assistant' && message.actionPlan?.requires_user_input && (
                <div className="action-buttons">
                  {message.actionPlan.status === 'requires_confirmation' && (
                    <>
                      <button 
                        className="confirm-btn"
                        onClick={() => handleQuickReply('Yes, proceed')}
                        disabled={isLoading}
                      >
                        ✓ Yes, Proceed
                      </button>
                      <button 
                        className="cancel-btn"
                        onClick={() => handleQuickReply('No, cancel')}
                        disabled={isLoading}
                      >
                        ✗ Cancel
                      </button>
                    </>
                  )}
                  
                  {message.actionPlan.status === 'needs_clarification' && 
                   message.actionPlan.missing_params && 
                   message.actionPlan.missing_params.length > 0 && (
                    <div className="clarification-hint">
                      💡 Needs: {message.actionPlan.missing_params.join(', ')}
                    </div>
                  )}
                </div>
              )}
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
              {messages.length === 1 && (
                <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#666' }}>
                  First message may take 10-30 seconds while loading AI model...
                </div>
              )}
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
          <div className="quick-actions" />
        )}
      </div>
    </div>
  );
};

export default ChatInterface;