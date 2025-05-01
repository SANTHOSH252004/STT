import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Add this import at the top of your file
// You'll need to run: npm install recordrtc
import RecordRTC from 'recordrtc';

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';

function App() {
  const [tab, setTab] = useState('tts');
  const [script, setScript] = useState('');
  const [voice] = useState('shimmer');
  const [audioURL, setAudioURL] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(5); // Default 5 seconds
  const recordingTimerRef = useRef(null);
  const micStreamRef = useRef(null);
  const websocketRef = useRef(null);
  const audioRef = useRef(null);
  const [isTranscribingFile, setIsTranscribingFile] = useState(false);
  const recorderRef = useRef(null);
  const transcriptChunksRef = useRef([]);

  // Clean up WebSocket on component unmount
  useEffect(() => {
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      stopRecording();
    };
  }, []);

  const handleSynthesize = async () => {
    if (!script.trim()) return;
    
    try {
      setIsLoading(true);
      const response = await axios.post(`${API_BASE_URL}/synthesize`, {
        script,
        voice
      });
      setAudioURL(`${API_BASE_URL}${response.data.audio_url}`);
      setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play();
        }
      }, 100);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTranscribe = async () => {
    if (!audioFile) return;
    
    try {
      setIsTranscribing(true);
      setTranscript(''); // Clear previous transcript
      
      const formData = new FormData();
      formData.append('file', audioFile);
      
      const response = await axios.post(`${API_BASE_URL}/transcribe`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setTranscript(response.data.text);
    } catch (error) {
      console.error('Error transcribing file:', error);
      setTranscript('Error transcribing file: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsTranscribing(false);
    }
  };

  const setupWebSocket = () => {
    if (websocketRef.current) {
      websocketRef.current.close();
    }

    websocketRef.current = new WebSocket(`${WS_BASE_URL}/ws/transcribe`);
    transcriptChunksRef.current = [];
    
    websocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        console.error('Transcription error:', data.error);
        setRecordingError(data.error);
        stopRecording();
      } else if (data.text) {
        transcriptChunksRef.current.push(data.text.trim());
        setTranscript(transcriptChunksRef.current.join(' '));
      }
    };

    websocketRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setRecordingError('Error connecting to transcription service. Make sure the backend server is running.');
      stopRecording();
    };

    return new Promise((resolve, reject) => {
      websocketRef.current.onopen = () => resolve();
      websocketRef.current.onerror = (error) => reject(new Error('WebSocket connection failed'));
    });
  };

  const startRecording = async () => {
    try {
      setRecordingError('');
      setTranscript('');
      transcriptChunksRef.current = [];
      
      // Setup WebSocket first
      await setupWebSocket();
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        } 
      });
      micStreamRef.current = stream;
      
      // Create recorder with WAV format
      const recorder = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        recorderType: RecordRTC.StereoAudioRecorder,
        numberOfAudioChannels: 1,
        desiredSampRate: 16000,
        timeSlice: 1000, // Get data every 1 second
        ondataavailable: (blob) => {
          // Send data to WebSocket
          if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then(buffer => {
              websocketRef.current.send(buffer);
            });
          }
        }
      });
      
      // Start recording
      recorder.startRecording();
      recorderRef.current = recorder;
      
      // Set recording state
      setIsRecording(true);
      setIsTranscribing(true);
      
      // Optional: set maximum recording duration
      recordingTimerRef.current = setTimeout(() => {
        if (isRecording) {
          stopRecording();
        }
      }, recordingDuration * 1000); // Convert to milliseconds
      
    } catch (error) {
      console.error('Error recording:', error);
      // Handle specific errors to provide better feedback
      if (error.name === 'NotAllowedError') {
        setRecordingError('Microphone access denied. Please allow microphone access to record.');
      } else if (error.name === 'NotFoundError') {
        setRecordingError('No microphone found. Please connect a microphone and try again.');
      } else {
        setRecordingError(`Error recording: ${error.message || 'Unknown error'}`);
      }
      // Make sure recording state is reset
      setIsRecording(false);
      setIsTranscribing(false);
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && isRecording) {
      recorderRef.current.stopRecording(() => {
        // Get the recorded blob
        const blob = recorderRef.current.getBlob();
        console.log('Recording stopped, blob size:', blob.size);
        
        // Clean up
        recorderRef.current = null;
      });
      
      setIsRecording(false);
      setIsTranscribing(false);
      
      // Clean up microphone stream
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach(track => track.stop());
        micStreamRef.current = null;
      }
      
      // Close WebSocket
      if (websocketRef.current) {
        // Keep the connection open to receive any final transcriptions
        setTimeout(() => {
          if (websocketRef.current) {
            websocketRef.current.close();
            websocketRef.current = null;
          }
        }, 2000);
      }
      
      // Clear timer
      if (recordingTimerRef.current) {
        clearTimeout(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }
  };

  const clearAudioFile = (e) => {
    e.stopPropagation();
    setAudioFile(null);
    setTranscript('');
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('audio/')) {
        setRecordingError('Please select an audio file');
        return;
      }
      setAudioFile(file);
      setTranscript(''); // Clear previous transcript
    }
  };

  return (
    <div className="app">
      <div className="header">
        <button className={tab === 'tts' ? 'tab active' : 'tab'} onClick={() => setTab('tts')}>TEXT-TO-SPEECH</button>
        <button className={tab === 'stt' ? 'tab active' : 'tab'} onClick={() => setTab('stt')}>SPEECH-TO-TEXT</button>
      </div>

      {tab === 'tts' && (
        <>
          <h1 className="title">Text-to-Speech Converter</h1>
          <p className="subtitle">Convert your text into speech</p>

          <div className="card">
            <label>Script</label>
            <p className="input-description">Enter the text you want to convert to speech</p>
            <textarea value={script} onChange={(e) => setScript(e.target.value)} placeholder="Enter your script..." />

            <div className="actions">
              <button 
                className={`generate ${isLoading && script.trim() ? 'loading' : ''}`} 
                onClick={handleSynthesize} 
                disabled={isLoading || !script.trim()}
              >
                {isLoading && script.trim() ? <span className="spinner">⌛</span> : 'GENERATE'}
              </button>
            </div>

            {audioURL && (
              <div className="player">
                <audio ref={audioRef} controls src={audioURL}></audio>
              </div>
            )}
          </div>
        </>
      )}

      {tab === 'stt' && (
        <div className="card">
          <h1 className="title">Speech-to-Text Converter</h1>
          <p className="subtitle">Convert your speech into text</p>

          <div className="live-recording">
            <div className="recording-controls">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`record-button ${isRecording ? 'recording' : ''}`}
                disabled={isTranscribingFile}
              >
                {isRecording ? 'STOP RECORDING' : 'START RECORDING'}
                {isRecording && <span className="recording-indicator"></span>}
              </button>
              
              
            </div>
            
            {recordingError && <p className="error-message">{recordingError}</p>}
            
            <div className="recording-status">
              {isRecording && (
                <div className="status-text">
                  <span className="listening-text">Listening and transcribing...</span>
                </div>
              )}
            </div>
          </div>

          <div className="upload-section">
            <p className="section-title">Or upload an audio file</p>
            
            <label className="custom-file-upload">
              <input
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                onClick={(e) => audioFile && (e.target.value = '')}
              />
              <span>
                {audioFile ? `Selected: ${audioFile.name}` : '🎵 Upload Audio File'}
              </span>
              {audioFile && (
                <button 
                  className="file-remove-btn" 
                  onClick={clearAudioFile}
                  title="Remove file"
                >
                  🗑️
                </button>
              )}
            </label>

            <button 
              onClick={handleTranscribe} 
              className="transcribe-button"
              disabled={!audioFile || isRecording || isTranscribing}
            >
              {isTranscribing ? <span className="spinner">⌛</span> : 'TRANSCRIBE FILE'}
            </button>

            <div className="transcript-container">
              <label>Transcription:</label>
              <textarea
                readOnly
                value={transcript}
                placeholder="Transcription will appear here..."
                className="transcript-textarea"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
