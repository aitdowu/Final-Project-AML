"""
PDF Ingestion Script for RAG Course Notes Chatbot

This script processes PDF files from the data/ directory, extracts text using both
PyPDF2 and OCR (via pdf2image + pytesseract), chunks the text, computes embeddings,
and stores them in a FAISS vector database.

Author: RAG Course Notes Chatbot Project
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle

# PDF processing
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# Vector storage
import faiss

# Utilities
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction using both direct text extraction and OCR."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
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
        """
        Extract text from PDF using OCR (pdf2image + pytesseract).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
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
    
    def merge_captions(self, text: str, pdf_name: str, captions_dir: str = "data/captions") -> str:
        """
        Merge text captions for diagrams if available.
        
        Args:
            text: Original extracted text
            pdf_name: Name of the PDF file (without extension)
            captions_dir: Directory containing caption files
            
        Returns:
            Text with merged captions
        """
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
    
    def process_pdf(self, pdf_path: str, use_ocr: bool = True) -> List[Dict[str, Any]]:
        """
        Process a single PDF file and return chunked text with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            use_ocr: Whether to use OCR in addition to direct text extraction
            
        Returns:
            List of dictionaries containing chunked text and metadata
        """
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
        combined_text = self.merge_captions(combined_text, pdf_name)
        
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


class EmbeddingGenerator:
    """Handles text embedding generation using open-source models."""
    
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        # TODO: Choose appropriate open-source embedding model
        # Current choice: all-MiniLM-L6-v2 (384 dimensions, good performance)
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings


class VectorStore:
    """Handles FAISS vector database operations."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings and metadata to the vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing similarity scores and metadata
        """
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):  # Valid index
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save(self, db_path: str):
        """
        Save the vector store to disk.
        
        Args:
            db_path: Path to save the database
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{db_path}.index")
        
        # Save metadata
        with open(f"{db_path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved vector store to {db_path}")
    
    def load(self, db_path: str):
        """
        Load the vector store from disk.
        
        Args:
            db_path: Path to load the database from
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{db_path}.index")
        
        # Load metadata
        with open(f"{db_path}.metadata", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded vector store from {db_path}")


def main():
    """Main function to process all PDFs and create the vector database."""
    
    # Configuration
    data_dir = "data"
    db_path = "db/vector_store"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found. Please create it and add PDF files.")
        return
    
    # Initialize components
    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Find all PDF files
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files found in '{data_dir}' directory.")
        logger.info("Please add PDF files to the data/ directory and run again.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process all PDFs
    all_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = pdf_processor.process_pdf(str(pdf_file))
        all_chunks.extend(chunks)
    
    if not all_chunks:
        logger.error("No text chunks were extracted from the PDFs.")
        return
    
    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    
    # Generate embeddings
    texts = [chunk['text'] for chunk in all_chunks]
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Add to vector store
    vector_store.add_embeddings(embeddings, all_chunks)
    
    # Save vector store
    vector_store.save(db_path)
    
    # Save processing summary
    summary = {
        'total_pdfs': len(pdf_files),
        'total_chunks': len(all_chunks),
        'embedding_model': embedding_generator.model_name,
        'embedding_dimension': embeddings.shape[1],
        'pdf_files': [str(f) for f in pdf_files]
    }
    
    with open(f"{db_path}.summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("PDF ingestion completed successfully!")
    logger.info(f"Processed {summary['total_pdfs']} PDFs into {summary['total_chunks']} chunks")
    logger.info(f"Vector database saved to {db_path}")


if __name__ == "__main__":
    main()

