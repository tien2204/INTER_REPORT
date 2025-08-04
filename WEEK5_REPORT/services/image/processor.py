import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import fitz  # pymupdf
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    A class for processing images and extracting text using OCR.
    
    Attributes:
        embeddings (OpenAIEmbeddings): Instance for creating text embeddings
        supported_extensions (List[str]): List of supported image file extensions
    """
    
    def __init__(self):
        """
        Initialize the ImageProcessor with necessary components.
        """
        self.embeddings = OpenAIEmbeddings()
        self.supported_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.pdf']

    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from an image using pytesseract OCR.
        
        Args:
            image_path (Path): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            text = pytesseract.image_to_string(Image.open(str(image_path)))
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def extract_text_from_pdf_images(self, pdf_path: Path) -> List[str]:
        """
        Extract text from images embedded in a PDF document.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            List[str]: List of extracted text from each image in the PDF
        """
        try:
            pdf_document = fitz.open(str(pdf_path))
            text_results = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save to buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Extract text
                text = self.extract_text_from_image(img_buffer)
                if text:
                    text_results.append(f"Page {page_num + 1}: {text}")
            
            return text_results
        except Exception as e:
            logger.error(f"Error processing PDF images: {str(e)}")
            return []

    def process_images(self) -> List[str]:
        """
        Process all images in the input directory.
        
        Returns:
            List[str]: List of extracted text from all images
        """
        texts = []
        for file_name in os.listdir(settings.INPUT_DIR):
            file_path = settings.INPUT_DIR / file_name
            if os.path.isfile(file_path) and file_path.suffix.lower() in self.supported_extensions:
                try:
                    if file_path.suffix.lower() == '.pdf':
                        texts.extend(self.extract_text_from_pdf_images(file_path))
                    else:
                        text = self.extract_text_from_image(file_path)
                        if text:
                            texts.append(text)
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
        return texts

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()  # Remove leading/trailing whitespace
        return text
