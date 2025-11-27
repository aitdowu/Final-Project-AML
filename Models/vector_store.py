import os
import logging
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS vector store for embeddings."""
    
    def __init__(self, dimension: int = None):
        """Initialize. Dimension can be set later if not provided."""
        self.dimension = dimension
        self.index = None
        self.metadata = []
        
        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings and metadata."""
        # Auto-detect dimension if needed
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.dimension = embedding_dim
            self.index = faiss.IndexFlatIP(embedding_dim)
            logger.info(f"Initialized FAISS index with dimension: {embedding_dim}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        if self.index is None:
            raise ValueError("Index not initialized. Load database or add embeddings first.")
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Check dimensions match
        query_dim = query_embedding.shape[1]
        if query_dim != self.dimension:
            raise ValueError(
                f"Dimension mismatch: query has {query_dim}, db has {self.dimension}. "
                f"Use the same embedding model. Check db/vector_store.summary.json"
            )
        
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save(self, db_path: str):
        """Save to disk."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        faiss.write_index(self.index, f"{db_path}.index")
        
        with open(f"{db_path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved to {db_path}")
    
    def load(self, db_path: str):
        """Load from disk."""
        self.index = faiss.read_index(f"{db_path}.index")
        self.dimension = self.index.d
        
        with open(f"{db_path}.metadata", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded from {db_path} (dimension: {self.dimension})")

