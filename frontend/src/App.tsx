import { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';
import ChatInterface from './components/ChatInterface';
import { ChatMessage } from './types';
import { v4 as uuidv4 } from 'uuid';
import { LLMChatService } from './services/grpcClient';
import './App.css';

function App() {
  // Get or create persistent session ID
  const [sessionId] = useState(() => {
    const stored = localStorage.getItem('current_session_id');
    if (stored) {
      return stored;
    }
    const newId = uuidv4();
    localStorage.setItem('current_session_id', newId);
    return newId;
  });
  
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentVideo, setCurrentVideo] = useState<string | null>(null);
  const [currentTranscription, setCurrentTranscription] = useState<string | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [hasLoadedInitialHistory, setHasLoadedInitialHistory] = useState(false);

  useEffect(() => {
    // Load chat history and session context on startup from backend
    if (!hasLoadedInitialHistory) {
      loadChatHistoryFromBackend();
      loadSessionContext();
    }
  }, [sessionId]);

  const loadSessionContext = async () => {
    try {
      const chatService = new LLMChatService();
      const result = await chatService.getSessionContext(sessionId);
      
      if (result.success && result.session) {
        // Restore video and transcription state if available
        if (result.session.video_filename) {
          setCurrentVideo(result.session.video_filename);
          console.log(`Restored video: ${result.session.video_filename}`);
        }
        if (result.session.transcription) {
          setCurrentTranscription(result.session.transcription);
          console.log('Restored transcription from session');
        }
      }
    } catch (error) {
      console.error('Failed to load session context:', error);
      // This is non-critical, session may be new
    }
  };

  const loadChatHistoryFromBackend = async () => {
    setIsLoadingHistory(true);
    try {
      const chatService = new LLMChatService();
      const result = await chatService.getChatHistory(sessionId);
      
      if (result.success && result.messages && result.messages.length > 0) {
        setChatHistory(result.messages);
        saveChatHistory(result.messages);
        console.log(`Loaded ${result.messages.length} messages from backend`);
      } else {
        console.log('No previous chat history found, starting fresh');
      }
      setHasLoadedInitialHistory(true);
    } catch (error) {
      console.error('Failed to load chat history from backend:', error);
      // Try fallback to localStorage
      loadChatHistoryFromLocalStorage();
      setHasLoadedInitialHistory(true);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const loadChatHistoryFromLocalStorage = () => {
    try {
      const stored = localStorage.getItem(`chat_history_${sessionId}`);
      if (stored) {
        const messages = JSON.parse(stored);
        setChatHistory(messages);
        console.log('Loaded chat history from localStorage');
      }
      setHasLoadedInitialHistory(true);
    } catch (error) {
      console.error('Failed to load chat history from localStorage:', error);
      setHasLoadedInitialHistory(true);
    }
  };

  const saveChatHistory = (messages: ChatMessage[]) => {
    try {
      localStorage.setItem(`chat_history_${sessionId}`, JSON.stringify(messages));
    } catch (error) {
      console.error('Failed to save chat history:', error);
    }
  };

  const addMessage = (message: ChatMessage) => {
    setChatHistory(prevHistory => {
      const newHistory = [...prevHistory, message];
      saveChatHistory(newHistory);
      return newHistory;
    });
  };

  const handleVideoProcessed = (filename: string, transcription: string) => {
    setCurrentVideo(filename);
    setCurrentTranscription(transcription);
    
    addMessage({
      id: uuidv4(),
      role: 'system',
      content: `Video "${filename}" processed successfully. Transcription completed.`,
      timestamp: Date.now(),
      sessionId
    });
  };

  const clearHistory = async () => {
    try {
      // Clear backend history
      const chatService = new LLMChatService();
      await chatService.clearHistory(sessionId);
      
      // Clear frontend state
      setChatHistory([]);
      
      // Clear localStorage
      localStorage.removeItem(`chat_history_${sessionId}`);
      
      console.log('Chat history cleared from backend and frontend');
    } catch (error) {
      console.error('Failed to clear chat history:', error);
      // Still clear frontend even if backend fails
      setChatHistory([]);
      localStorage.removeItem(`chat_history_${sessionId}`);
    }
  };

  const startNewSession = () => {
    // Create new session
    const newSessionId = uuidv4();
    localStorage.setItem('current_session_id', newSessionId);
    
    // Reload the app to start fresh
    window.location.reload();
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎥 AI Video Analyzer</h1>
        <p>Upload videos, get transcriptions, summaries, and chat with AI - all locally!</p>
        {!isLoadingHistory && chatHistory.length > 0 && (
          <div style={{ fontSize: '0.9em', color: '#666', marginTop: '5px' }}>
            Session: {sessionId.substring(0, 8)}... | {chatHistory.length} messages restored
          </div>
        )}
      </header>

      <main className="app-main">
        <div className="upload-section">
          <VideoUpload 
            onVideoProcessed={handleVideoProcessed}
            sessionId={sessionId}
          />
        </div>

        <div className="chat-section">
          <ChatInterface
            messages={chatHistory}
            onSendMessage={addMessage}
            sessionId={sessionId}
            currentVideo={currentVideo}
            currentTranscription={currentTranscription}
            onClearHistory={clearHistory}
            onNewSession={startNewSession}
          />
        </div>
      </main>
    </div>
  );
}

export default App;