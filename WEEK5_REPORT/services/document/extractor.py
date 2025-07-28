import os
from typing import List, Dict, Optional
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """
    A class for extracting text from various document formats.
    
    Attributes:
        supported_extensions (Dict[str, callable]): Mapping of file extensions to extraction methods
    """
    
    def __init__(self):
        """
        Initialize the DocumentExtractor with supported file types and their extraction methods.
        """
        self.supported_extensions = {
            '.pdf': self._extract_from_pdf,
            '.doc': self._extract_from_doc,
            '.docx': self._extract_from_doc,
            '.xls': self._extract_from_excel,
            '.xlsx': self._extract_from_excel
        }

    def extract_all_documents(self) -> List[str]:
        """
        Extract text from all documents in the input directory.
        
        Returns:
            List[str]: List of extracted text from documents
        """
        documents = []
        for file_name in os.listdir(settings.INPUT_DIR):
            file_path = settings.INPUT_DIR / file_name
            if os.path.isfile(file_path):
                try:
                    text = self._extract_text(file_path)
                    if text:
                        documents.append(text)
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
        return documents

    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from different file types based on their extension.
        
        Args:
            file_path (Path): Path to the file to extract text from
            
        Returns:
            str: Extracted text from the file
        """
        file_extension = file_path.suffix.lower()
        extractor = self.supported_extensions.get(file_extension)
        if extractor:
            return extractor(file_path)
        return ""

    def _extract_from_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from PDF
        """
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_from_doc(self, file_path: Path) -> str:
        """
        Extract text from Word document.
        
        Args:
            file_path (Path): Path to the Word document
            
        Returns:
            str: Extracted text from Word document
        """
        doc = Document(str(file_path))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_from_excel(self, file_path: Path) -> str:
        """
        Extract text from Excel file.
        
        Args:
            file_path (Path): Path to the Excel file
            
        Returns:
            str: Extracted text from Excel file
        """
        df = pd.read_excel(str(file_path))
        text = ""
        for column in df.columns:
            text += f"\nColumn: {column}\n"
            text += df[column].astype(str).str.cat(sep="\n")
        return text
