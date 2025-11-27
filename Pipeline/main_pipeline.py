import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from tqdm import tqdm

from Models.embedding_model import EmbeddingGenerator
from Models.vector_store import VectorStore
from Utils.pdf_processor import PDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Process all PDFs and build the vector database."""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, "data")
    db_path = os.path.join(project_root, "db", "vector_store")
    
    # Check if data dir exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found. Add PDF files first.")
        return
    
    # Initialize everything
    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Find PDFs
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDF files in '{data_dir}'.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process PDFs
    all_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = pdf_processor.process_pdf(str(pdf_file))
        all_chunks.extend(chunks)
    
    if not all_chunks:
        logger.error("No chunks extracted.")
        return
    
    logger.info(f"Total chunks: {len(all_chunks)}")
    
    # Generate embeddings
    texts = [chunk['text'] for chunk in all_chunks]
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Add to vector store
    vector_store.add_embeddings(embeddings, all_chunks)
    
    # Save everything
    vector_store.save(db_path)
    
    # Save summary
    summary = {
        'total_pdfs': len(pdf_files),
        'total_chunks': len(all_chunks),
        'embedding_model': embedding_generator.model_name,
        'embedding_dimension': embedding_generator.get_dimension(),
        'pdf_files': [str(f) for f in pdf_files]
    }
    
    with open(f"{db_path}.summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Done!")
    logger.info(f"Processed {summary['total_pdfs']} PDFs into {summary['total_chunks']} chunks")


if __name__ == "__main__":
    main()

