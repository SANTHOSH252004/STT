import requests
import os
import tempfile
import logging
import json
import sounddevice as sd
import soundfile as sf
import uuid
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder

logger = logging.getLogger(__name__)

WHISPER_API_KEY = "Q09ujpX9DvwW3FhNd7lp4Fas1PbzvaqtphtmfoseuX5KZSO3FcJOJQQJ99BDACHYHv6XJ3w3AAAAACOGwE5I"
WHISPER_RESOURCE = "santh-m9ds5x0q-eastus2"
WHISPER_DEPLOYMENT = "whisper"
WHISPER_ENDPOINT = f"https://{WHISPER_RESOURCE}.openai.azure.com/openai/deployments/{WHISPER_DEPLOYMENT}/audio/transcriptions?api-version=2023-09-01-preview"

def record_audio(duration=50, samplerate=16000):
    """
    Record audio using sounddevice
    
    Args:
        duration: Recording duration in seconds
        samplerate: Sampling rate in Hz
        
    Returns:
        str: Path to the recorded audio file
    """
    try:
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        
        # Generate unique filename
        filename = f"recordings/recording_{uuid.uuid4().hex}.wav"
        logger.info(f"Starting recording for {duration} seconds...")
        
        # Record audio
        audio = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32'
        )
        
        # Wait for recording to complete
        sd.wait()
        logger.info("Recording completed")
        
        # Normalize audio to prevent clipping
        audio = np.clip(audio, -1, 1)
        
        # Convert to int16 for WAV format
        audio = np.int16(audio * 32767)
        
        # Save the audio file
        sf.write(filename, audio, samplerate)
        logger.info(f"Audio saved to: {filename}")
        
        # Verify the file was created
        if not os.path.exists(filename):
            raise Exception("Failed to save audio file")
            
        # Verify file size
        file_size = os.path.getsize(filename)
        if file_size < 1000:  # Minimum reasonable size for a WAV file
            raise Exception(f"Audio file too small: {file_size} bytes")
            
        logger.info(f"Audio file size: {file_size} bytes")
        return filename
        
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        raise

def transcribe_recorded_audio(filename):
    """
    Transcribe a recorded audio file
    
    Args:
        filename: Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        with open(filename, 'rb') as f:
            audio_bytes = f.read()
            return transcribe_audio(audio_bytes, os.path.basename(filename))
    except Exception as e:
        logger.error(f"Error transcribing recorded audio: {str(e)}")
        raise

def validate_audio_file(file_bytes):
    """Validate the audio file before sending to Whisper"""
    if len(file_bytes) < 500:
        logger.error("Audio file is too small (less than 500 bytes)")
        return False
    if len(file_bytes) > 25 * 1024 * 1024:  # 25MB limit
        logger.error("Audio file is too large (more than 25MB)")
        return False
    return True

def transcribe_audio(file_bytes, filename):
    """
    Transcribe audio using Azure OpenAI Whisper service
    
    Args:
        file_bytes: The audio data as bytes
        filename: The name of the file
        
    Returns:
        str: The transcribed text
    """
    if not validate_audio_file(file_bytes):
        return ""
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        headers = {
            "api-key": WHISPER_API_KEY
        }
        
        # Create multipart form data with proper boundary
        multipart_data = MultipartEncoder(
            fields={
                'file': (filename, file_bytes, 'audio/wav'),
                'model': (None, 'whisper-1'),
                'language': (None, 'en'),
                'response_format': (None, 'json')
            }
        )
        
        headers['Content-Type'] = multipart_data.content_type
        
        logger.info(f"Sending request to Whisper API: {WHISPER_ENDPOINT}")
        logger.info(f"File size: {len(file_bytes)} bytes")
        
        try:
            response = requests.post(
                WHISPER_ENDPOINT,
                headers=headers,
                data=multipart_data,
                timeout=30
            )
            
            logger.info(f"Response status code: {response.status_code}")
            if not response.ok:
                logger.error(f"Response headers: {response.headers}")
                logger.error(f"Response content: {response.text}")
            
        except requests.exceptions.Timeout:
            logger.error("Whisper API timeout - processing took too long")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Whisper API request failed: {str(e)}")
            return ""

        if response.ok:
            try:
                response_data = response.json()
                transcript = response_data.get("text", "").strip()
                if not transcript:
                    logger.warning("Received empty transcript from Whisper API")
                    logger.warning(f"Full response: {json.dumps(response_data, indent=2)}")
                return transcript
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Whisper API response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
                return ""
        else:
            error_message = f"Whisper API Error {response.status_code}: {response.text}"
            logger.error(error_message)
            raise Exception(error_message)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {str(e)}")