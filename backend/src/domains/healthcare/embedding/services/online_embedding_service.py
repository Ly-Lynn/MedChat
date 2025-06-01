"""Online embedding service for real-time user queries"""

import logging
from typing import List, Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class OnlineEmbeddingService:
    """Service for real-time embedding of user queries"""
    
    def __init__(self, model_name: str = "biobert"):
        """Initialize the online embedding service
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        self.cache: Dict[str, List[float]] = {}
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model"""
        try:
            # For now, we'll create a mock implementation
            logger.info(f"🔄 Initializing {self.model_name} model...")
            
            # Mock model initialization
            self.model = {"model_name": self.model_name, "dimensions": 768}
            
            logger.info(f"✅ {self.model_name} model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.model_name} model: {str(e)}")
            raise
    
    async def embed_user_query(self, query: str) -> List[float]:
        """Embed a user query for real-time search
        
        Args:
            query: User query text
            
        Returns:
            List of embedding values
        """
        try:
            # Check cache first
            if query in self.cache:
                logger.info(f"📋 Cache hit for query: {query[:50]}...")
                return self.cache[query]
            
            # Preprocess query
            processed_query = await self._preprocess_query(query)
            
            # Generate embedding (mock implementation)
            embedding = await self._generate_embedding(processed_query)
            
            # Cache the result
            self.cache[query] = embedding
            
            logger.info(f"✅ Generated embedding for query: {query[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error embedding query: {str(e)}")
            raise
    
    async def embed_multiple_queries(self, queries: List[str]) -> List[List[float]]:
        """Embed multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            List of embeddings
        """
        try:
            embeddings = []
            
            for query in queries:
                embedding = await self.embed_user_query(query)
                embeddings.append(embedding)
            
            logger.info(f"✅ Generated embeddings for {len(queries)} queries")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error embedding multiple queries: {str(e)}")
            raise
    
    async def _preprocess_query(self, query: str) -> str:
        """Preprocess user query for better embedding
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Basic preprocessing
        processed = query.strip().lower()
        
        # Medical abbreviation expansion (mock)
        abbreviations = {
            "dm": "diabetes mellitus",
            "htn": "hypertension",
            "mi": "myocardial infarction"
        }
        
        for abbr, full in abbreviations.items():
            processed = processed.replace(abbr, full)
        
        return processed
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Mock embedding generation
        # In real implementation, this would use BioBERT
        import hashlib
        import random
        
        # Use hash for consistent "embedding"
        hash_obj = hashlib.md5(text.encode())
        random.seed(int(hash_obj.hexdigest(), 16))
        
        # Generate 768-dimensional vector
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        
        return embedding
    
    def switch_model(self, model_name: str):
        """Switch to a different embedding model
        
        Args:
            model_name: Name of the new model
        """
        if model_name != self.model_name:
            logger.info(f"🔄 Switching model from {self.model_name} to {model_name}")
            self.model_name = model_name
            self.cache.clear()  # Clear cache when switching models
            self._init_model()
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        logger.info("🗑️ Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "model_name": self.model_name,
            "total_queries_cached": len(self.cache)
        } 