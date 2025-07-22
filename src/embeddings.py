from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict
import logging
import pickle
from pathlib import Path

from utils.helpers import ensure_directory_exists

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wrapper for multilingual embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Trying fallback model...")
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Fallback model loaded. Dimension: {self.dimension}")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts to embeddings"""
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return np.array([])
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        if not text:
            return np.array([])
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return embedding[0]
        except Exception as e:
            logger.error(f"Single encoding failed: {e}")
            return np.array([])
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension if self.dimension else 384


class VectorStore:
    """Base class for vector storage"""
    
    def __init__(self, embedding_model: EmbeddingModel, save_path: str):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None
        self.save_path = Path(save_path)

    def is_loaded(self) -> bool:
        """Check if the store has any chunks loaded."""
        return len(self.chunks) > 0

    def exists(self) -> bool:
        """Check if the vector store file exists on disk."""
        return self.save_path.exists()

    def build(self, chunks: List[Dict]) -> bool:
        """Builds the vector store from chunks. Alias for add_chunks."""
        return self.add_chunks(chunks)
    
    def add_chunks(self, chunks: List[Dict]) -> bool:
        """Add document chunks to the store"""
        if not chunks:
            logger.warning("No chunks to add")
            return False
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode_texts(texts)
        
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return False
        
        # Store chunks and embeddings
        self.chunks = chunks
        self.embeddings = embeddings
        
        logger.info(f"Successfully added {len(chunks)} chunks")
        return True
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Dict]:
        """Search for similar chunks"""
        if not self.chunks or self.embeddings is None:
            logger.warning("No chunks in vector store")
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode_single(query)
        
        if query_embedding.size == 0:
            logger.error("Failed to encode query")
            return []
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k results above threshold
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices[:k]:
            if similarities[idx] >= threshold:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(similarities[idx])
                results.append(chunk)
            else:
                break
        
        logger.info(f"Found {len(results)} similar chunks for query")
        return results
    
    def save(self) -> bool:
        """Save vector store to disk"""
        try:
            ensure_directory_exists(str(self.save_path.parent))
            
            store_data = {
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "model_name": self.embedding_model.model_name,
                "dimension": self.embedding_model.dimension
            }
            
            with open(self.save_path, 'wb') as f:
                pickle.dump(store_data, f)
            
            logger.info(f"Vector store saved to: {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load(self) -> bool:
        """Load vector store from disk"""
        try:
            if not self.exists():
                logger.warning(f"Vector store file not found: {self.save_path}")
                return False
            
            with open(self.save_path, 'rb') as f:
                store_data = pickle.load(f)
            
            self.chunks = store_data["chunks"]
            self.embeddings = store_data["embeddings"]
            
            # Verify model compatibility
            if store_data["model_name"] != self.embedding_model.model_name:
                logger.warning(f"Model mismatch: stored={store_data['model_name']}, current={self.embedding_model.model_name}")
            
            logger.info(f"Loaded {len(self.chunks)} chunks from: {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        if not self.chunks:
            return {"status": "empty"}
        
        languages = [chunk["metadata"]["language"] for chunk in self.chunks]
        pages = [chunk["metadata"]["page"] for chunk in self.chunks]
        
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_model.dimension,
            "languages": list(set(languages)),
            "language_distribution": {lang: languages.count(lang) for lang in set(languages)},
            "page_range": f"{min(pages)}-{max(pages)}",
            "total_pages": len(set(pages)),
            "model_name": self.embedding_model.model_name
        }
