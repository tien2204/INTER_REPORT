import speech_recognition as sr
import numpy as np
from faster_whisper import WhisperModel
import sounddevice as sd
import queue
from typing import Optional

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.wake_words = ["hey assistant", "hey computer"]
        self.model = WhisperModel("small", device="cpu")

    def listen_for_wake_word(self) -> Optional[str]:
        """Listen for wake word and return True if detected"""
        with sr.Microphone() as source:
            print("Listening for wake word...")
            audio = self.recognizer.listen(source)
            
            try:
                text = self.recognizer.recognize_google(audio)
                text = text.lower()
                print(f"Heard: {text}")
                
                if any(word in text for word in self.wake_words):
                    return text
                elif "stop" in text or "goodbye" in text:
                    return "stop"
                return None
            
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    def listen_for_query(self) -> Optional[str]:
        """Listen for user query and return the text"""
        with sr.Microphone() as source:
            print("Listening for query...")
            audio = self.recognizer.listen(source)
            
            try:
                # First try Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print(f"Query: {text}")
                return text
            
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None
