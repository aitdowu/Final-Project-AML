import os
import logging
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract text from PDFs. Uses PyPDF2 and OCR."""
    
    def __init__(self):
        """Initialize with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR (for scanned PDFs)."""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""
            
            for i, image in enumerate(images):
                # Use OCR to extract text from image
                page_text = pytesseract.image_to_string(image)
                text += f"Page {i+1}:\n{page_text}\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text with OCR from {pdf_path}: {e}")
            return ""
    
    def merge_captions(self, text: str, pdf_name: str, captions_dir: str = None) -> str:
        """Merge caption files if they exist (for diagrams)."""
        # Use absolute path for captions directory
        if captions_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            captions_dir = os.path.join(project_root, "data", "captions")
        elif not os.path.isabs(captions_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            captions_dir = os.path.join(project_root, captions_dir)
        
        # Try both naming conventions: {pdf_name}_captions.txt and {pdf_name}.txt
        caption_file = os.path.join(captions_dir, f"{pdf_name}_captions.txt")
        if not os.path.exists(caption_file):
            caption_file = os.path.join(captions_dir, f"{pdf_name}.txt")
        
        if os.path.exists(caption_file):
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    captions = f.read()
                text += f"\n\n[Captions for {pdf_name}]:\n{captions}"
                logger.info(f"Merged captions for {pdf_name}")
            except Exception as e:
                logger.warning(f"Could not merge captions for {pdf_name}: {e}")
        
        return text
    
    def process_pdf(self, pdf_path: str, use_ocr: bool = True, captions_dir: str = None) -> List[Dict[str, Any]]:
        """Process a PDF and return chunked text."""
        pdf_name = Path(pdf_path).stem
        
        # Extract text using PyPDF2
        direct_text = self.extract_text_from_pdf(pdf_path)
        
        # Extract text using OCR if requested
        ocr_text = ""
        if use_ocr:
            ocr_text = self.extract_text_with_ocr(pdf_path)
        
        # Combine both extraction methods
        combined_text = direct_text
        if ocr_text and ocr_text.strip():
            combined_text += f"\n\n[OCR Text]:\n{ocr_text}"
        
        # Merge captions if available
        combined_text = self.merge_captions(combined_text, pdf_name, captions_dir)
        
        # Chunk the text
        chunks = self.text_splitter.split_text(combined_text)
        
        # Create metadata for each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only include non-empty chunks
                processed_chunks.append({
                    'text': chunk,
                    'source': pdf_name,
                    'chunk_id': i,
                    'pdf_path': pdf_path,
                    'extraction_method': 'combined' if ocr_text else 'direct'
                })
        
        logger.info(f"Processed {pdf_name}: {len(processed_chunks)} chunks")
        return processed_chunks

