from TTS.api import TTS
import sounddevice as sd
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self, language="en"):
        # Kiểm tra nếu có GPU
        self.use_gpu = torch.cuda.is_available()
        logger.info(f"Using GPU: {self.use_gpu}")
        self.set_voice(language)

    def speak(self, text: str) -> None:
        try:
            logger.info(f"Speaking: {text}")
            # Generate waveform
            wav = self.tts.tts(text)
            # Play audio
            sd.play(wav, samplerate=22050)
            sd.wait()
        except Exception as e:
            logger.error(f"Error in speak: {e}")
            raise

    def set_voice(self, language: str = "vi") -> None:
        try:
            logger.info(f"Setting voice to language: {language}")
            if language == "vi":
                model = "tts_models/vi/vivos/glow-tts"
            elif language == "en":
                model = "tts_models/en/ljspeech/tacotron2-DDC"
            else:
                raise ValueError(f"Unsupported language: {language}")
            self.tts = TTS(model_name=model, progress_bar=False, gpu=self.use_gpu)
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            raise

    def get_available_voices(self):
        return ["en", "vi"]

