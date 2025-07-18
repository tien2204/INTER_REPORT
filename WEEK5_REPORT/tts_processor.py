import pyttsx3
import os
from typing import Optional

class TTSProcessor:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

    def speak(self, text: str) -> None:
        """Speak the given text"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error speaking: {str(e)}")

    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file from text"""
        try:
            # Create temp directory if it doesn't exist
            if not os.path.exists("temp"):
                os.makedirs("temp")

            # Generate audio file
            temp_file = os.path.join("temp", f"output_{int(time.time())}.wav")
            self.engine.save_to_file(text, temp_file)
            self.engine.runAndWait()
            return temp_file
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None
