import sounddevice as sd
import numpy as np
import queue
import threading
import io
import os
import tempfile
import subprocess
from faster_whisper import WhisperModel
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

# Settings
samplerate = 16000
block_duration = 0.5
chunk_duration = 2
channels = 1

frames_per_block = int(block_duration * samplerate)
frames_per_chunk = int(chunk_duration * samplerate)

audio_queue = queue.Queue()
audio_buffer = []

# Initialize whisper model
MODEL_SIZE = "small.en" 
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# Function to handle audio stream
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels, blocksize=frames_per_block, callback=audio_callback):
        print("Recording...")
        while True:
            sd.sleep(100)
           
def transcribe():
    global audio_buffer
    while True:
        try:
            block = audio_queue.get()
            audio_buffer.append(block)

            total_frames = sum(len(b) for b in audio_buffer)
            if total_frames >= frames_per_chunk:
                audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
                audio_buffer = []

                audio_data = audio_data.flatten().astype(np.float32)

                # Transcription 
                segments, _ = model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=1,
                )

                for segment in segments:
                    print(f"[{segment.text}]")
        except Exception as e:
            print(f"Error in transcribe: {e}")

def transcribe_chunk(audio_bytes, input_format="webm"):
    """
    Transcribe an audio chunk using faster-whisper
    
    Args:
        audio_bytes: Audio data as bytes
        input_format: Format of input audio
        
    Returns:
        str: Transcribed text
    """
    if len(audio_bytes) < 100:  # Skip very small chunks
        return ""
    
    try:
        # Process audio bytes
        samples, sample_rate = process_audio_bytes(audio_bytes, input_format)
        
        # Transcribe
        logger.info(f"Transcribing audio chunk: {len(audio_bytes)} bytes, {len(samples) / sample_rate:.2f} seconds")
        segments, info = model.transcribe(
            samples, 
            language="en", 
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect text from segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)
        
        return " ".join(text_parts).strip()
        
    except Exception as e:
        logger.error(f"Error transcribing chunk: {str(e)}")
        raise

def process_audio_bytes(audio_bytes, input_format="wav"):
    """
    Process audio bytes to format suitable for faster-whisper
    
    Args:
        audio_bytes: Audio data as bytes
        input_format: Format of input audio
        
    Returns:
        np.ndarray: Audio data as numpy array
        int: Sample rate
    """
    # If the data is too small, it's likely not valid audio
    if len(audio_bytes) < 100:
        raise ValueError(f"Audio data too small ({len(audio_bytes)} bytes), likely not valid audio")
    
    try:
        # Log the first few bytes for debugging
        first_bytes = ' '.join(f'{b:02x}' for b in audio_bytes[:16])
        logger.info(f"First 16 bytes: {first_bytes}")
        
        # Check if it's a WAV file with header (should start with "RIFF")
        if input_format.lower() == "wav" and len(audio_bytes) >= 44 and audio_bytes[0:4] == b'RIFF':
            logger.info("Processing WAV data with header")
            
            # Parse WAV header
            channels = int.from_bytes(audio_bytes[22:24], byteorder='little')
            sample_rate = int.from_bytes(audio_bytes[24:28], byteorder='little')
            bits_per_sample = int.from_bytes(audio_bytes[34:36], byteorder='little')
            
            logger.info(f"WAV header info: channels={channels}, sample_rate={sample_rate}, bits_per_sample={bits_per_sample}")
            
            try:
                # Find the "data" chunk
                data_pos = audio_bytes.find(b'data')
                if data_pos > 0:
                    # The data chunk has a 4-byte size after the "data" marker
                    data_size = int.from_bytes(audio_bytes[data_pos+4:data_pos+8], byteorder='little')
                    data_start = data_pos + 8  # Skip the "data" marker and size
                    
                    logger.info(f"Found data chunk at position {data_pos}, size {data_size} bytes")
                    
                    # Extract audio data
                    if bits_per_sample == 16:
                        # Convert bytes to 16-bit integers
                        data = np.frombuffer(audio_bytes[data_start:data_start+data_size], dtype=np.int16)
                        # Normalize to float32 in range [-1, 1]
                        samples = data.astype(np.float32) / 32768.0
                        return samples, sample_rate
                    else:
                        logger.warning(f"Unsupported bits per sample: {bits_per_sample}, trying alternative...")
                        
                else:
                    # Simple extraction - skip the 44-byte header
                    logger.info("Could not find 'data' chunk, using standard 44-byte header")
                    if bits_per_sample == 16:
                        # Convert bytes to 16-bit integers
                        data = np.frombuffer(audio_bytes[44:], dtype=np.int16)
                        # Normalize to float32 in range [-1, 1]
                        samples = data.astype(np.float32) / 32768.0
                        return samples, sample_rate
                    
            except Exception as wav_error:
                logger.warning(f"Error extracting WAV data: {wav_error}")
        
        # Direct in-memory conversion using pydub
        import io
        from pydub import AudioSegment
        
        logger.info(f"Processing audio using pydub: format={input_format}, data size={len(audio_bytes)}")
        
        # Use in-memory conversion
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=input_format)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        
        # Convert to raw PCM data
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        # Normalize to [-1, 1]
        samples = samples / (1 << (8 * audio_segment.sample_width - 1))
        
        return samples, audio_segment.frame_rate
        
    except Exception as direct_error:
        # If direct conversion fails, try with temporary files
        logger.warning(f"Direct audio conversion failed: {direct_error}. Trying with temporary files...")
        
        # Create temporary directory and files manually
        temp_dir = tempfile.mkdtemp()
        try:
            # Determine the correct file extension
            ext = "wav" if input_format.lower() == "wav" else input_format
            input_path = os.path.join(temp_dir, f"input.{ext}")
            output_path = os.path.join(temp_dir, "output.wav")
            
            # Save the input bytes to a file
            with open(input_path, "wb") as f:
                f.write(audio_bytes)
            
            try:
                # Try using FFmpeg directly with more verbose output
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", input_path,
                    "-acodec", "pcm_s16le",
                    "-ac", "1",
                    "-ar", "16000",
                    "-v", "warning",  # Be more verbose
                    output_path
                ]
                
                process = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True
                )
                
                if process.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {process.stderr}")
                    raise RuntimeError(f"FFmpeg conversion failed: {process.stderr}")
                
                # Load the WAV file directly into numpy array
                with open(output_path, 'rb') as f:
                    wav_data = f.read()
                    # Skip the WAV header (44 bytes)
                    pcm_data = wav_data[44:]
                    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                return samples, 16000
                
            except Exception as e:
                logger.error(f"Failed to process audio: {e}")
                
                # Last resort: create a silent audio segment
                logger.warning("Creating silent audio as fallback")
                samples = np.zeros(16000, dtype=np.float32)
                return samples, 16000
        finally:
            # Clean up the temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

# Only start the threads if this file is run directly, not when imported
if __name__ == "__main__":
    threading.Thread(target=recorder, daemon=True).start()
    transcribe()
            





