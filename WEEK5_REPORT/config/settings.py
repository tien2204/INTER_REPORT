from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str

    # Server Configuration
    SERVER_HOST: str
    SERVER_PORT: int
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INPUT_DIR: Path = DATA_DIR / "input"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    HISTORY_DIR: Path = DATA_DIR / "history"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/voice_agent.log"

    # Document Processing
    MAX_DOCUMENT_SIZE: int = 10_000_000
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Voice Processing
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    # History
    MAX_HISTORY_SIZE: int = 100

    # Model Configuration
    EMBEDDING_MODEL: str
    RERANKER_MODEL: str

    # Document Embeddings
    CHROMA_PERSIST_DIRECTORY: str
    CHROMA_COLLECTION_NAME: str

    # OCR
    OCR_LANGUAGE: str = "vi"

    # GraphRAG
    GRAPH_SIMILARITY_THRESHOLD: float = 0.3
    MAX_RELATED_DOCS: int = 5

    tts_voice_id: str = 'vi'
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
