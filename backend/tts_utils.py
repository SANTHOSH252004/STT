import requests
from typing import Optional

TTS_API_KEY = "1ql77YZAtFKDTrFdUBVv4zqBPc8HEOtBDSBpr5CzpusRJ6tRs6i5JQQJ99BDACHrzpqXJ3w3AAAAACOGi0an"
TTS_ENDPOINT_URL = "https://santh-m9jgfhsu-northcentralus.openai.azure.com/openai/deployments/tts-hd/audio/speech?api-version=2025-03-01-preview"

def get_ssml_with_emotion(text: str, emotion: Optional[str] = None) -> str:
    """Generate SSML with emotion markers if specified"""
    if not emotion:
        return text
        
    # Strip emoji from emotion text if present
    emotion = ''.join(c for c in emotion if not (0x1F300 <= ord(c) <= 0x1F9FF))
    emotion = emotion.strip().lower()
    
    style_map = {
        'joy': 'cheerful',
        'anger': 'angry',
        'awe': 'excited',
        'disgust': 'unfriendly',
        'savoring': 'friendly',
        'hushed': 'whispering'
    }
    
    style = style_map.get(emotion, '')
    
    if style:
        return f'<speak><voice effect="style={style}">{text}</voice></speak>'
    return text

def synthesize_speech(text: str, voice: str = "shimmer", format: str = "audio-16khz-128kbitrate-mono-mp3", emotion: Optional[str] = None) -> bytes:
    """
    Synthesize speech from text using Azure TTS API.
    
    Args:
        text: The text to synthesize
        voice: Voice ID to use
        format: Audio format to return
        emotion: Optional emotion to apply
        
    Returns:
        bytes: Audio data
        
    Raises:
        Exception: If API call fails
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": TTS_API_KEY
        }
        
        # Process text for SSML if emotion is specified
        processed_text = get_ssml_with_emotion(text, emotion)
        
        payload = {
            "input": processed_text,
            "voice": voice,
            "output_format": format
        }
        
        response = requests.post(TTS_ENDPOINT_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS API Error {response.status_code}: {response.text}")
            
    except Exception as e:
        raise Exception(f"Failed to synthesize speech: {str(e)}")