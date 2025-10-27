import React, { useState } from 'react';
import { open } from '@tauri-apps/api/dialog';
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

  const handleSelectVideo = async () => {
    if (isUploading) {
      return;
    }

    const selected = await open({
      title: 'Select MP4 Video',
      multiple: false,
      filters: [{ name: 'Video', extensions: ['mp4'] }],
    });

    if (!selected) {
      return;
    }

    const filePath = Array.isArray(selected) ? selected[0] : selected;

    if (!filePath.toLowerCase().endsWith('.mp4')) {
      alert('Please select an MP4 video file.');
      return;
    }

    try {
      const filename = filePath.split(/[\/\\]/).pop() || 'video.mp4';
      setVideoInfo({ filename, size: 0 });
      await processVideo(filePath, filename);
    } catch (error) {
      console.error('Failed to read file info:', error);
      alert('Unable to read file information.');
    }
  };

  const processVideo = async (filePath: string, filename: string) => {
    setIsUploading(true);
    setUploadProgress(10);

    try {
      const videoService = new VideoAnalysisService();
      const result = await videoService.transcribeVideoFromPath(filePath, sessionId);

      setUploadProgress(80);

      if (result.success) {
        setUploadProgress(100);
        onVideoProcessed(filename, result.transcription);
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

  return (
    <div className="video-upload">
      <div
        className={`upload-zone ${isUploading ? 'uploading' : ''}`}
        onClick={handleSelectVideo}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            handleSelectVideo();
          }
        }}
      >
        {!isUploading ? (
          <div className="upload-content">
            <div className="upload-icon">📹</div>
            <h3>Select Video File</h3>
            <p>Click to choose an MP4 video from your computer</p>
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
          {videoInfo.size > 0 && (
            <p><strong>Size:</strong> {(videoInfo.size / 1024 / 1024).toFixed(2)} MB</p>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoUpload;