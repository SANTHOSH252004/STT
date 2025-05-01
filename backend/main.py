from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from whisper_utils import transcribe_audio, record_audio
from faster_whisper_utils import transcribe_chunk
from tts_utils import synthesize_speech
from pydub import AudioSegment
import os
import io
import tempfile
import subprocess
import logging
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create temp_audio directory if it doesn't exist
os.makedirs("temp_audio", exist_ok=True)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/temp_audio", StaticFiles(directory="temp_audio"), name="temp_audio")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware with updated origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Favicon route
@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = "static/favicon.ico"
    # Create a default favicon if it doesn't exist
    if not os.path.exists(favicon_path):
        # Create static directory if needed
        os.makedirs("static", exist_ok=True)
        # Create an empty favicon file
        with open(favicon_path, "wb") as f:
            # Simple 16x16 transparent ICO file (minimal valid ICO format)
            f.write(bytes.fromhex("00000100010010100000010020006804000016000000"))
    
    return FileResponse(favicon_path)

def convert_audio_to_wav(audio_data, input_format="webm"):
    """Convert audio data to WAV format using pydub with enhanced error handling"""
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input audio to temporary file
            input_path = os.path.join(temp_dir, f"input.{input_format}")
            output_path = os.path.join(temp_dir, "output.wav")
            
            with open(input_path, "wb") as f:
                f.write(audio_data)
            
            try:
                # Try using pydub first
                audio = AudioSegment.from_file(input_path)
                # Convert to mono and set sample rate
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                
                with open(output_path, "rb") as f:
                    return f.read()
            except Exception as pydub_error:
                logger.warning(f"Pydub conversion failed: {str(pydub_error)}")
                
                # Fallback to direct ffmpeg conversion
                try:
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", input_path,
                        "-acodec", "pcm_s16le",
                        "-ac", "1",
                        "-ar", "16000",
                        output_path
                    ]
                    
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    if process.returncode != 0:
                        raise Exception(f"FFmpeg error: {process.stderr}")
                    
                    with open(output_path, "rb") as f:
                        return f.read()
                        
                except Exception as ffmpeg_error:
                    raise Exception(f"Audio conversion failed. FFmpeg error: {str(ffmpeg_error)}")
                
    except Exception as e:
        raise Exception(f"Audio conversion error: {str(e)}")

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    temp_audio = io.BytesIO()
    buffer_size = 0
    chunk_threshold = 24000  # Adjusted buffer size for better real-time performance
    last_transcription_time = time.time()
    transcription_interval = 0.5  # Process at least every 0.5 seconds
    
    logger.info("WebSocket connection opened for transcription")
    
    try:
        while True:
            try:
                # Set a timeout for receiving data
                audio_chunk = await websocket.receive_bytes()
                
                # Skip empty chunks
                if not audio_chunk or len(audio_chunk) < 100:
                    logger.warning(f"Received small audio chunk ({len(audio_chunk) if audio_chunk else 0} bytes), skipping")
                    continue
                    
                chunk_size = len(audio_chunk)
                logger.info(f"Received audio chunk: {chunk_size} bytes")
                
                # Check if it's a complete WAV file with header
                if chunk_size >= 44 and audio_chunk[0:4] == b'RIFF':
                    # Process it directly as a WAV file
                    try:
                        logger.info(f"Processing complete WAV chunk: {chunk_size} bytes")
                        first_bytes = ' '.join(f'{b:02x}' for b in audio_chunk[:16])
                        logger.info(f"First 16 bytes: {first_bytes}")
                        
                        # Use faster-whisper for transcription
                        transcript = transcribe_chunk(audio_chunk, "wav")
                        
                        if transcript and transcript.strip():
                            await websocket.send_json({"text": transcript})
                            logger.info(f"Transcribed: {transcript[:30]}...")
                        else:
                            logger.info("No transcription result or silent audio")
                        
                        last_transcription_time = time.time()
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Transcription error: {error_msg}")
                        await websocket.send_json({"error": error_msg})
                    
                    # Reset the buffer after processing a complete WAV
                    temp_audio = io.BytesIO()
                    buffer_size = 0
                    continue
                
                # If not a WAV file, log format information for debugging
                if chunk_size >= 4:
                    header_bytes = ' '.join(f'{b:02x}' for b in audio_chunk[:16])
                    logger.info(f"Non-WAV chunk header: {header_bytes}")
                
                # Handle raw audio chunks by appending to buffer
                temp_audio.write(audio_chunk)
                buffer_size += chunk_size
                
                current_time = time.time()
                time_since_last = current_time - last_transcription_time
                
                # Process when we have enough data or enough time has passed
                if buffer_size >= chunk_threshold or time_since_last >= transcription_interval:
                    if buffer_size > 0:  # Only process if we have data
                        logger.info(f"Processing audio buffer: {buffer_size} bytes, {time_since_last:.2f}s since last processing")
                        
                        temp_audio.seek(0)
                        audio_data = temp_audio.getvalue()
                        
                        # Reset buffer right away to avoid delays
                        new_buffer = io.BytesIO()
                        buffer_size = 0
                        temp_audio.close()
                        temp_audio = new_buffer
                        
                        try:
                            # Check if we have enough data to process
                            if len(audio_data) < 100:
                                logger.warning(f"Audio buffer too small: {len(audio_data)} bytes")
                                continue
                            
                            # Use faster-whisper for transcription
                            transcript = transcribe_chunk(audio_data, "wav")
                            
                            if transcript and transcript.strip():
                                await websocket.send_json({"text": transcript})
                                logger.info(f"Transcribed: {transcript[:30]}...")
                            else:
                                logger.info("No transcription result or silent audio")
                            
                            last_transcription_time = time.time()
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Transcription error: {error_msg}")
                            await websocket.send_json({"error": error_msg})
            
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
                
            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {str(chunk_error)}")
                try:
                    await websocket.send_json({"error": f"Chunk processing error: {str(chunk_error)}"})
                except:
                    break  # If we can't send, client is probably disconnected
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"WebSocket error: {error_msg}")
        try:
            await websocket.send_json({"error": error_msg})
        except:
            pass
    finally:
        temp_audio.close()
        logger.info("WebSocket connection closed")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(None), filename: str = Form(None)):
    try:
        if file:
            # Handle uploaded file
            content = await file.read()
            audio_filename = file.filename
        elif filename:
            # Handle recorded audio file
            if not os.path.exists(filename):
                raise HTTPException(status_code=404, detail="Audio file not found")
            with open(filename, 'rb') as f:
                content = f.read()
            audio_filename = os.path.basename(filename)
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")
            
        transcript = transcribe_audio(content, audio_filename)
        return {"text": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize(payload: dict):
    try:
        script = payload.get("script")
        voice = payload.get("voice", "shimmer")
        emotion = payload.get("emotion")
        format = "audio-16khz-128kbitrate-mono-mp3"
        
        audio_bytes = synthesize_speech(script, voice, format, emotion)
        
        temp_dir = "temp_audio"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        temp_file = os.path.join(temp_dir, f"{hash(script)}.mp3")
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
            
        return {"audio_url": f"/temp_audio/{os.path.basename(temp_file)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record")
async def record(payload: dict):
    try:
        duration = payload.get("duration", 5)
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise HTTPException(status_code=400, detail="Invalid duration. Must be a positive number.")
        
        # Record audio
        filename = record_audio(duration=duration)
        
        if not filename or not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Failed to record audio")
            
        return {
            "success": True,
            "filename": filename,
            "message": f"Audio recorded successfully for {duration} seconds"
        }
    except Exception as e:
        logger.error(f"Recording error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API is running"}