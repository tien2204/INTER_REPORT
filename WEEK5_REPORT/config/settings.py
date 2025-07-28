from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_DIR: Path = DATA_DIR / "input"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    HISTORY_DIR: Path = DATA_DIR / "history"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Voice Processing
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # RAG Settings
    TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
