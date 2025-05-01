# Live Transcription App

This application provides real-time speech-to-text transcription using the faster-whisper model. It consists of a React frontend and a FastAPI backend.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher

## Setup

### Backend Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
cd backend
python main.py
```

The backend server will start at http://localhost:8000

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## Usage

1. Open the application in your web browser
2. Click the "Start Recording" button to begin recording audio
3. Speak into your microphone
4. The transcription will appear in real-time in the text box
5. Click "Stop Recording" when you're done

## Technical Details

- Frontend: React with TypeScript
- Backend: FastAPI
- Speech-to-Text: faster-whisper model
- Real-time communication: WebSocket

## Notes

- Make sure to allow microphone access in your browser when prompted
- The transcription quality may vary depending on:
  - Background noise
  - Microphone quality
  - Speaking clarity
  - Network connection speed 
