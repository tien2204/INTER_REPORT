from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from .extractor import DocumentExtractor
from ..utils.history_manager import HistoryManager
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    A class for processing documents by extracting text, creating embeddings,
    and storing them in a vector database.
    
    Attributes:
        extractor (DocumentExtractor): Instance for extracting text from documents
        embeddings (OpenAIEmbeddings): Instance for creating text embeddings
        history_manager (HistoryManager): Instance for managing conversation history
        text_splitter (RecursiveCharacterTextSplitter): Instance for splitting text into chunks
    """
    
    def __init__(self):
        """
        Initialize the DocumentProcessor with necessary components.
        """
        self.extractor = DocumentExtractor()
        self.embeddings = OpenAIEmbeddings()
        self.history_manager = HistoryManager()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def process_documents(self) -> None:
        """
        Process all documents in the input directory.
        
        This method extracts text from documents, splits them into chunks,
        creates embeddings, and stores them in ChromaDB.
        
        Raises:
            Exception: If there's an error during document processing
        """
        try:
            documents = self.extractor.extract_all_documents()
            texts = self.text_splitter.split_documents(documents)
            
            # Create embeddings and store in ChromaDB
            db = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                persist_directory=str(settings.EMBEDDINGS_DIR)
            )
            db.persist()
            
            logger.info(f"Processed {len(texts)} chunks of text")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def get_document_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieve all document chunks from the vector store.
        
        Returns:
            List[Dict[str, Any]]: List of document chunks with their embeddings
            
        Raises:
            Exception: If there's an error retrieving document chunks
        """
        try:
            db = Chroma(
                persist_directory=str(settings.EMBEDDINGS_DIR),
                embedding_function=self.embeddings
            )
            return db.get()['documents']
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
