#!/usr/bin/env python3
"""
Simple test script for faster-whisper real-time transcription
"""
import time
import threading
import traceback
import faster_whisper_utils

print("Starting real-time transcription test...")
print("Speak into your microphone. Press Ctrl+C to stop.")
print("-" * 50)

try:
    # Start the recorder thread
    recorder_thread = threading.Thread(
        target=faster_whisper_utils.recorder,
        daemon=True
    )
    recorder_thread.start()
    print("Recorder thread started successfully")
    
    # Call transcribe directly
    print("Starting transcription...")
    faster_whisper_utils.transcribe()
except KeyboardInterrupt:
    print("\nStopping transcription test.")
except Exception as e:
    print(f"\nError occurred: {e}")
    print("Traceback:")
    traceback.print_exc()
    print("\nPlease make sure you have a microphone connected and all dependencies installed.")
    print("You may need to run: pip install -r requirements.txt") 