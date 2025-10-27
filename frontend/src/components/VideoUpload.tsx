import React, { useState, useRef } from 'react';
import { VideoAnalysisService } from '../services/grpcClient';
import { VideoInfo } from '../types';

interface VideoUploadProps {
  onVideoProcessed: (filename: string, transcription: string) => void;
  sessionId: string;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoProcessed, sessionId }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('video/') || !file.name.endsWith('.mp4')) {
      alert('Please select a valid MP4 video file.');
      return;
    }

    // Validate file size (max 100MB for local processing)
    if (file.size > 100 * 1024 * 1024) {
      alert('File size must be less than 100MB.');
      return;
    }

    setVideoInfo({
      filename: file.name,
      size: file.size
    });

    await processVideo(file);
  };

  const processVideo = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Read file as bytes
      const arrayBuffer = await file.arrayBuffer();
      const videoData = new Uint8Array(arrayBuffer);

      setUploadProgress(30);

      // Call transcription service
      const videoService = new VideoAnalysisService();
      const result = await videoService.transcribeVideo(videoData, file.name);

      setUploadProgress(80);

      if (result.success) {
        setUploadProgress(100);
        onVideoProcessed(file.name, result.transcription);
      } else {
        throw new Error(result.errorMessage || 'Transcription failed');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      alert(`Error processing video: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && fileInputRef.current) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInputRef.current.files = dt.files;
      fileInputRef.current.dispatchEvent(new Event('change', { bubbles: true }));
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  return (
    <div className="video-upload">
      <div
        className={`upload-zone ${isUploading ? 'uploading' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          disabled={isUploading}
        />
        
        {!isUploading ? (
          <div className="upload-content">
            <div className="upload-icon">📹</div>
            <h3>Upload Video File</h3>
            <p>Click here or drag and drop your MP4 video file</p>
            <small>Maximum file size: 100MB</small>
          </div>
        ) : (
          <div className="upload-progress">
            <div className="upload-icon">⏳</div>
            <h3>Processing Video...</h3>
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <p>{uploadProgress}% Complete</p>
          </div>
        )}
      </div>

      {videoInfo && !isUploading && (
        <div className="video-info">
          <h4>📄 File Information</h4>
          <p><strong>Name:</strong> {videoInfo.filename}</p>
          <p><strong>Size:</strong> {(videoInfo.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;