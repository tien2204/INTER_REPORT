import os
import json
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

class DocumentProcessor:
    def __init__(self, documents_path: str, database_path: str):
        self.documents_path = documents_path
        self.database_path = database_path
        self.embeddings = OpenAIEmbeddings()

    def extract_text(self, file_path: str) -> str:
        """Extract text from different file types"""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension in ['.doc', '.docx']:
            return self._extract_from_doc(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return self._extract_from_excel(file_path)
        else:
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _extract_from_doc(self, file_path: str) -> str:
        """Extract text from Word document"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        df = pd.read_excel(file_path)
        text = ""
        for column in df.columns:
            text += f"\nColumn: {column}\n"
            text += df[column].astype(str).str.cat(sep="\n")
        return text

    def process_documents(self):
        """Process all documents in the directory"""
        documents = []
        for file_name in os.listdir(self.documents_path):
            file_path = os.path.join(self.documents_path, file_name)
            if os.path.isfile(file_path):
                text = self.extract_text(file_path)
                if text:
                    documents.append(text)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings and store in ChromaDB
        db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=self.database_path
        )
        db.persist()
        print(f"Processed {len(texts)} chunks of text")
