import pyttsx3
import os
from typing import Optional, List, Dict
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    A class for converting text to speech using pyttsx3.
    
    Attributes:
        engine (pyttsx3.Engine): Text-to-speech engine instance
    """
    
    def __init__(self):
        """
        Initialize the TextToSpeech processor with default settings.
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    def speak(self, text: str) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text (str): Text to be spoken
            
        Raises:
            Exception: If there's an error during speech synthesis
        """
        try:
            logger.info(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            raise

    def set_voice(self, language: str = "vi") -> Optional[str]:
        """
        Set the voice based on the specified language.
        
        Args:
            language (str): Language code for the desired voice (default: "vi")
            
        Returns:
            Optional[str]: ID of the selected voice if found, None otherwise
        """
        try:
            voices = self.engine.getProperty('voices')
            
            for voice in voices:
                if language in voice.languages[0].decode():
                    self.engine.setProperty('voice', voice.id)
                    return voice.id
            
            logger.warning(f"No voice found for language: {language}")
            return None
        except Exception as e:
            logger.error(f"Error in set_voice: {e}")
            return None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices with their properties.
        
        Returns:
            List[Dict[str, Any]]: List of voice properties
        """
        voices = self.engine.getProperty('voices')
        return [{
            'id': voice.id,
            'name': voice.name,
            'languages': voice.languages,
            'gender': voice.gender
        } for voice in voices]
