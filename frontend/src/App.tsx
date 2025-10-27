import React, { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';
import ChatInterface from './components/ChatInterface';
import { ChatMessage } from './types';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

function App() {
  const [sessionId] = useState(() => uuidv4());
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [currentVideo, setCurrentVideo] = useState<string | null>(null);
  const [currentTranscription, setCurrentTranscription] = useState<string | null>(null);

  useEffect(() => {
    // Load chat history on startup
    loadChatHistory();
  }, []);

  const loadChatHistory = async () => {
    try {
      const stored = localStorage.getItem(`chat_history_${sessionId}`);
      if (stored) {
        setChatHistory(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
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
    const newHistory = [...chatHistory, message];
    setChatHistory(newHistory);
    saveChatHistory(newHistory);
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

  const clearHistory = () => {
    setChatHistory([]);
    localStorage.removeItem(`chat_history_${sessionId}`);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎥 AI Video Analyzer</h1>
        <p>Upload videos, get transcriptions, summaries, and chat with AI - all locally!</p>
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
          />
        </div>
      </main>
    </div>
  );
}

export default App;