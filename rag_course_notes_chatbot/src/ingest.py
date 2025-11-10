"""
PDF Ingestion Script for RAG Course Notes Chatbot

This script processes PDF files from the data/ directory, extracts text using both
PyPDF2 and OCR (via pdf2image + pytesseract), chunks the text, computes embeddings,
and stores them in a FAISS vector database.

Author: RAG Course Notes Chatbot Project
"""

import os

# Set HuggingFace cache directory to local project directory
# This avoids permission issues with ~/.cache/huggingface
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
local_cache = os.path.join(project_root, ".cache", "huggingface")
os.makedirs(local_cache, exist_ok=True)

# Set environment variables to use local cache
os.environ["HF_HOME"] = local_cache
os.environ["SENTENCE_TRANSFORMERS_HOME"] = local_cache
# Disable tokenizers parallelism warning when forking processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable xformers (optional dependency that often fails to install on macOS)
os.environ["DISABLE_XFORMERS"] = "1"

import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle
import warnings

# PDF processing
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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
    
    def merge_captions(self, text: str, pdf_name: str, captions_dir: str = None) -> str:
        """
        Merge text captions for diagrams if available.
        
        Args:
            text: Original extracted text
            pdf_name: Name of the PDF file (without extension)
            captions_dir: Directory containing caption files (if None, uses project root)
            
        Returns:
            Text with merged captions
        """
        # Use absolute path for captions directory
        if captions_dir is None:
            # Get project root (assuming this is called from main())
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            captions_dir = os.path.join(project_root, "data", "captions")
        elif not os.path.isabs(captions_dir):
            # If relative path, make it absolute relative to project root
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
        """
        Process a single PDF file and return chunked text with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            use_ocr: Whether to use OCR in addition to direct text extraction
            captions_dir: Directory containing caption files
            
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


class EmbeddingGenerator:
    """Handles text embedding generation using sentence-transformers or transformers."""
    
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        # Get the local cache path (set at module level)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        local_cache = os.path.join(project_root, ".cache", "huggingface")
        
        self.model_name = model_name
        # Detect Snowflake model - check if model name contains snowflake or arctic-embed
        self.is_snowflake_model = ("snowflake" in model_name.lower() or 
                                   "arctic-embed" in model_name.lower() or
                                   "arctic_embed" in model_name.lower())
        
        logger.info(f"Model detection: is_snowflake_model = {self.is_snowflake_model}")
        
        try:
            if self.is_snowflake_model:
                # Load Snowflake model using transformers
                # Note: xformers is optional and not required for the model to work
                logger.info("Loading Snowflake Arctic Embed model with transformers")
                
                # Suppress xformers/flash-attention warnings - these are optional optimizations
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    
                    # Load tokenizer first
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=local_cache,
                        trust_remote_code=True
                    )
                    
                    # Load config and modify it to disable memory_efficient_attention
                    # CRITICAL: The model's custom code requires xformers when use_memory_efficient_attention=True
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(
                        model_name,
                        cache_dir=local_cache,
                        trust_remote_code=True
                    )
                    
                    logger.info("Modifying config to disable memory_efficient_attention (no xformers needed)")
                    
                    # Force disable memory_efficient_attention using multiple methods
                    # The model code reads this from config, so we need to ensure it's False
                    original_value = getattr(config, 'use_memory_efficient_attention', None)
                    
                    # Set it to False in all possible ways
                    config.use_memory_efficient_attention = False
                    setattr(config, 'use_memory_efficient_attention', False)
                    
                    # Also modify the internal dict if it exists
                    if hasattr(config, '__dict__'):
                        config.__dict__['use_memory_efficient_attention'] = False
                    
                    # Modify the _name_or_path to ensure config is treated as modified
                    # Also try to update the config's to_dict representation
                    try:
                        config_dict = config.to_dict()
                        config_dict['use_memory_efficient_attention'] = False
                        # Create new config from modified dict
                        config = type(config).from_dict(config_dict)
                        config.use_memory_efficient_attention = False
                    except Exception as e:
                        logger.debug(f"Could not recreate config from dict: {e}")
                        # Continue with modified config object
                    
                    logger.info(f"  Changed use_memory_efficient_attention: {original_value} -> False")
                    
                    # Verify the change took effect
                    final_value = getattr(config, 'use_memory_efficient_attention', None)
                    if final_value is not False:
                        logger.warning(f"  WARNING: use_memory_efficient_attention is still {final_value}, may need xformers")
                    else:
                        logger.info(f"  Verified: use_memory_efficient_attention = {final_value}")
                    
                    # Load model with modified config
                    # The config parameter should override the default config from the model files
                    logger.info("Loading model with modified config (no xformers required)")
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_name,
                            config=config,  # Pass modified config to override defaults
                            cache_dir=local_cache,
                            add_pooling_layer=False,
                            trust_remote_code=True
                        )
                    except AssertionError as e:
                        if "xformers" in str(e).lower() or "please install xformers" in str(e):
                            # If it still fails, the config modification didn't work
                            # This might mean the model code is reading from a different source
                            logger.error("=" * 60)
                            logger.error("XFORMERS ERROR: Config modification didn't prevent xformers requirement")
                            logger.error("=" * 60)
                            logger.error("The Snowflake model's custom code is still requiring xformers.")
                            logger.error("")
                            logger.error("Possible solutions:")
                            logger.error("1. Install xformers (may not work on macOS):")
                            logger.error("   pip install xformers")
                            logger.error("")
                            logger.error("2. Use a different embedding model that doesn't require xformers")
                            logger.error("   Example: 'sentence-transformers/all-MiniLM-L6-v2'")
                            logger.error("")
                            logger.error("3. Check Hugging Face discussions for workarounds:")
                            logger.error("   https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/discussions")
                            logger.error("=" * 60)
                            raise RuntimeError(
                                "Snowflake model requires xformers, but xformers is not available. "
                                "Please install xformers or use a different embedding model."
                            ) from e
                        else:
                            raise
                
                self.model.eval()
                
                # Determine device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded on device: {self.device}")
                
                # Get embedding dimension by testing (using mean pooling)
                test_text = "test"
                test_tokens = self.tokenizer([test_text], padding=True, truncation=True, 
                                            return_tensors='pt', max_length=8192)
                test_tokens = {k: v.to(self.device) for k, v in test_tokens.items()}
                with torch.no_grad():
                    test_outputs = self.model(**test_tokens)
                    test_last_hidden = test_outputs[0]
                    # Use mean pooling like in generate_embeddings
                    test_attention_mask = test_tokens.get('attention_mask', None)
                    if test_attention_mask is not None:
                        test_mask_expanded = test_attention_mask.unsqueeze(-1).expand(test_last_hidden.size()).float()
                        test_sum = torch.sum(test_last_hidden * test_mask_expanded, dim=1)
                        test_sum_mask = torch.clamp(test_mask_expanded.sum(dim=1), min=1e-9)
                        test_output = test_sum / test_sum_mask
                    else:
                        test_output = test_last_hidden.mean(dim=1)
                self.embedding_dimension = test_output.shape[1]
                
            else:
                # Load standard sentence-transformers model
                # BUT: Check if this might be a Snowflake model that wasn't detected
                # (This can happen if model name format is unusual)
                if "snowflake" in model_name.lower() or "arctic" in model_name.lower():
                    # Should have been caught by detection, but double-check
                    logger.warning(f"Model name suggests Snowflake model but detection didn't catch it: {model_name}")
                    logger.warning("Attempting to load with transformers library instead...")
                    self.is_snowflake_model = True
                    self._load_snowflake_model(model_name, local_cache)
                else:
                    # Load standard sentence-transformers model
                    logger.info("Loading model with sentence-transformers")
                    try:
                        self.model = SentenceTransformer(
                            model_name, 
                            cache_folder=local_cache,
                            trust_remote_code=True  # Some models may need this
                        )
                        self.tokenizer = None
                        self.device = None
                        
                        # Get embedding dimension from the model
                        test_embedding = self.model.encode(["test"])
                        self.embedding_dimension = test_embedding.shape[1]
                    except Exception as st_error:
                        error_str = str(st_error).lower()
                        # Check if this is a Snowflake model that sentence-transformers can't handle
                        if "trust_remote_code" in error_str or "custom code" in error_str:
                            logger.warning(f"sentence-transformers cannot load {model_name} (requires custom code)")
                            logger.warning("This model likely requires transformers library. Attempting to load with transformers...")
                            # Check if it looks like a Snowflake model
                            if "snowflake" in model_name.lower() or "arctic" in model_name.lower():
                                self.is_snowflake_model = True
                                self._load_snowflake_model(model_name, local_cache)
                            else:
                                raise RuntimeError(
                                    f"Model {model_name} requires trust_remote_code=True but sentence-transformers cannot load it. "
                                    "This model may need to be loaded with transformers library directly."
                                ) from st_error
                        else:
                            raise
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            error_msg = str(e).lower()
            
            # Check if this looks like a Snowflake model that needs transformers
            if ("snowflake" in model_name.lower() or "arctic" in model_name.lower()) and \
               ("trust_remote_code" in error_msg or "custom code" in error_msg):
                logger.error("")
                logger.error("=" * 60)
                logger.error("MODEL LOADING ERROR - Snowflake Model Detected")
                logger.error("=" * 60)
                logger.error(f"The model {model_name} requires transformers library with trust_remote_code=True")
                logger.error("The code should automatically use transformers for Snowflake models.")
                logger.error("")
                logger.error("If you're seeing this, there may be an issue with model detection.")
                logger.error("Try restarting the notebook kernel to reload the updated code.")
                logger.error("=" * 60)
                raise RuntimeError(
                    f"Snowflake model {model_name} requires transformers library. "
                    "Please restart the notebook kernel and try again, or check that the model detection is working."
                ) from e
            elif self.is_snowflake_model:
                if "xformers" in error_msg or "flash" in error_msg:
                    logger.error("")
                    logger.error("=" * 60)
                    logger.error("XFORMERS ERROR DETECTED")
                    logger.error("=" * 60)
                    logger.error("xformers is NOT required for the Snowflake model to work!")
                    logger.error("=" * 60)
                logger.error("Make sure transformers is installed: pip install transformers")
                logger.error("Make sure torch is installed: pip install torch")
            else:
                logger.error("Make sure sentence-transformers is installed: pip install sentence-transformers")
            raise
    
    def _load_snowflake_model(self, model_name: str, local_cache: str):
        """Helper method to load Snowflake model with transformers."""
        logger.info("Loading Snowflake model with transformers library...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=local_cache,
            trust_remote_code=True
        )
        
        # Load and modify config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=local_cache,
            trust_remote_code=True
        )
        
        # Disable memory_efficient_attention
        config.use_memory_efficient_attention = False
        setattr(config, 'use_memory_efficient_attention', False)
        if hasattr(config, '__dict__'):
            config.__dict__['use_memory_efficient_attention'] = False
        
        try:
            config_dict = config.to_dict()
            config_dict['use_memory_efficient_attention'] = False
            config = type(config).from_dict(config_dict)
            config.use_memory_efficient_attention = False
        except Exception:
            pass  # Continue with modified config
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=local_cache,
            add_pooling_layer=False,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Get dimension
        test_text = "test"
        test_tokens = self.tokenizer([test_text], padding=True, truncation=True, 
                                    return_tensors='pt', max_length=8192)
        test_tokens = {k: v.to(self.device) for k, v in test_tokens.items()}
        with torch.no_grad():
            test_outputs = self.model(**test_tokens)
            test_last_hidden = test_outputs[0]
            test_attention_mask = test_tokens.get('attention_mask', None)
            if test_attention_mask is not None:
                test_mask_expanded = test_attention_mask.unsqueeze(-1).expand(test_last_hidden.size()).float()
                test_sum = torch.sum(test_last_hidden * test_mask_expanded, dim=1)
                test_sum_mask = torch.clamp(test_mask_expanded.sum(dim=1), min=1e-9)
                test_output = test_sum / test_sum_mask
            else:
                test_output = test_last_hidden.mean(dim=1)
        self.embedding_dimension = test_output.shape[1]
        logger.info(f"Snowflake model loaded successfully. Embedding dimension: {self.embedding_dimension}")
    
    def generate_embeddings(self, texts: List[str], is_query: bool = False, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            is_query: Whether these are query texts (adds "query: " prefix for Snowflake model)
            batch_size: Batch size for processing (for Snowflake model)
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if self.is_snowflake_model:
            # Snowflake model requires special handling
            if is_query:
                # Add query prefix for queries
                query_prefix = "query: "
                texts = [f"{query_prefix}{text}" for text in texts]
            
            # Process in batches for memory efficiency
            all_embeddings = []
            num_batches = (len(texts) + batch_size - 1) // batch_size
            
            if show_progress:
                from tqdm import tqdm
                batch_iter = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
            else:
                batch_iter = range(0, len(texts), batch_size)
            
            for i in batch_iter:
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize and generate embeddings
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                      return_tensors='pt', max_length=8192)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    # Get the last hidden state (token embeddings)
                    last_hidden_state = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
                    
                    # Use mean pooling over all tokens (better than first token only)
                    # Apply attention mask to exclude padding tokens
                    attention_mask = tokens.get('attention_mask', None)
                    if attention_mask is not None:
                        # Expand attention mask to match hidden state dimensions
                        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        # Sum embeddings, excluding padding
                        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                        # Sum attention mask to get number of non-padding tokens
                        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                        # Mean pooling
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                    else:
                        # If no attention mask, just use mean over sequence length
                        batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    all_embeddings.append(batch_embeddings)
            
            # Concatenate all batch embeddings
            embeddings = np.vstack(all_embeddings)
            # Note: FAISS will normalize embeddings when storing/searching (see add_embeddings/search methods)
            return embeddings
        else:
            # Standard sentence-transformers model
            embeddings = self.model.encode(texts, show_progress_bar=show_progress)
            return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dimension


class VectorStore:
    """Handles FAISS vector database operations."""
    
    def __init__(self, dimension: int = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings. If None, will be set when first embeddings are added.
        """
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
        # Initialize index if dimension is provided
        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings and metadata to the vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
        """
        # Auto-detect dimension if not set
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.dimension = embedding_dim
            self.index = faiss.IndexFlatIP(embedding_dim)
            logger.info(f"Initialized FAISS index with dimension: {embedding_dim}")
        
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
        if self.index is None:
            raise ValueError("Vector store index not initialized. Load a database or add embeddings first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        
        # Check dimension match before searching
        query_dim = query_embedding.shape[1]
        if query_dim != self.dimension:
            raise ValueError(
                f"Dimension mismatch: Query embedding has {query_dim} dimensions, "
                f"but database has {self.dimension} dimensions. "
                f"Make sure you're using the same embedding model that was used to create the database. "
                f"Check the database summary at db/vector_store.summary.json for the correct model name."
            )
        
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
        
        # Get dimension from the loaded index
        self.dimension = self.index.d
        
        # Load metadata
        with open(f"{db_path}.metadata", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded vector store from {db_path} (dimension: {self.dimension})")


def main():
    """Main function to process all PDFs and create the vector database."""
    
    # Get the project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration - use absolute paths
    data_dir = os.path.join(project_root, "data")
    db_path = os.path.join(project_root, "db", "vector_store")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found. Please create it and add PDF files.")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Project root: {project_root}")
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
        'embedding_dimension': embedding_generator.get_dimension(),
        'pdf_files': [str(f) for f in pdf_files]
    }
    
    with open(f"{db_path}.summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("PDF ingestion completed successfully!")
    logger.info(f"Processed {summary['total_pdfs']} PDFs into {summary['total_chunks']} chunks")
    logger.info(f"Vector database saved to {db_path}")


if __name__ == "__main__":
    main()

