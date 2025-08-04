import speech_recognition as sr
from typing import Optional
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    A class for handling speech-to-text conversion using Google Speech Recognition.
    
    Attributes:
        recognizer (sr.Recognizer): Speech recognition engine instance
    """
    
    def __init__(self):
        """
        Initialize the SpeechToText processor with default settings.
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000

    def listen_for_wake_word(self) -> bool:
        """
        Listen for the wake word to activate the assistant.
        
        Returns:
            bool: True if wake word is detected, False otherwise
        """
        try:
            with sr.Microphone(sample_rate=settings.SAMPLE_RATE) as source:
                logger.info("Listening for wake word...")
                audio = self.recognizer.listen(source, timeout=5)
                
                try:
                    text = self.recognizer.recognize_google(audio, language="vi-VN")
                    text = text.lower()
                    detected = "hey assistant" in text
                    logger.info(f"Wake word detected: {detected}")
                    return detected
                except sr.UnknownValueError:
                    logger.debug("Could not understand audio")
                    return False
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    return False
        except Exception as e:
            logger.error(f"Error in listen_for_wake_word: {e}")
            return False

    def listen_for_query(self) -> Optional[str]:
        """
        Listen for and transcribe user query.
        
        Returns:
            Optional[str]: Transcribed query text if successful, None otherwise
        """
        try:
            with sr.Microphone(sample_rate=settings.SAMPLE_RATE) as source:
                logger.info("Listening for query...")
                audio = self.recognizer.listen(source, timeout=10)
                
                try:
                    query = self.recognizer.recognize_google(audio, language="vi-VN")
                    logger.info(f"Recognized query: {query}")
                    return query
                except sr.UnknownValueError:
                    logger.error("Google Speech Recognition could not understand audio")
                    return None
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service: {e}")
                    return None
        except Exception as e:
            logger.error(f"Error in listen_for_query: {e}")
            return None
