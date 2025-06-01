"""Offline embedding service for batch document processing"""

import logging
from typing import List, Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class OfflineEmbeddingService:
    """Service for batch processing and embedding of documents"""
    
    def __init__(self, model_name: str = "pubmedbert"):
        """Initialize the offline embedding service
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model = None
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"🔄 Initializing {self.model_name} model for offline processing...")
            
            # Mock model initialization
            self.model = {"model_name": self.model_name, "dimensions": 768}
            
            logger.info(f"✅ {self.model_name} model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.model_name} model: {str(e)}")
            raise
    
    async def process_document(self, document) -> object:
        """Process and embed a single document
        
        Args:
            document: Document object to process
            
        Returns:
            Processed document with embeddings
        """
        try:
            logger.info(f"🔄 Processing document: {getattr(document, 'title', 'Unknown')}")
            
            # Extract text content
            text_content = self._extract_text_content(document)
            
            # Split text into chunks
            chunks = await self._split_text(text_content)
            
            # Generate embeddings for chunks
            embeddings = []
            for chunk in chunks:
                embedding = await self._generate_embedding(chunk)
                embeddings.append(embedding)
            
            # Update document with embeddings
            document.embeddings = embeddings
            document.processed = True
            
            logger.info(f"✅ Processed document with {len(embeddings)} embeddings")
            return document
            
        except Exception as e:
            logger.error(f"❌ Error processing document: {str(e)}")
            raise
    
    def _extract_text_content(self, document) -> str:
        """Extract text content from document
        
        Args:
            document: Document object
            
        Returns:
            Extracted text content
        """
        content_parts = []
        
        # Add title
        if hasattr(document, 'title') and document.title:
            content_parts.append(document.title)
        
        # Add content
        if hasattr(document, 'content') and document.content:
            if isinstance(document.content, dict):
                # Handle structured content
                for key, value in document.content.items():
                    if value:
                        content_parts.append(f"{key}: {value}")
            else:
                # Handle plain text content
                content_parts.append(str(document.content))
        
        return "\n\n".join(content_parts)
    
    async def _split_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """Split text into chunks for embedding
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple text splitting (mock implementation)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        
        logger.info(f"📄 Split text into {len(chunks)} chunks")
        return chunks
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Mock embedding generation
        import hashlib
        import random
        
        # Use hash for consistent "embedding"
        hash_obj = hashlib.md5(text.encode())
        random.seed(int(hash_obj.hexdigest(), 16))
        
        # Generate 768-dimensional vector
        embedding = [random.uniform(-1, 1) for _ in range(768)]
        
        return embedding
    
    async def process_batch(self, documents: List[object]) -> List[object]:
        """Process multiple documents in batch
        
        Args:
            documents: List of document objects
            
        Returns:
            List of processed documents
        """
        try:
            logger.info(f"🔄 Processing batch of {len(documents)} documents")
            
            processed_docs = []
            for doc in documents:
                processed_doc = await self.process_document(doc)
                processed_docs.append(processed_doc)
            
            logger.info(f"✅ Processed batch of {len(processed_docs)} documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"❌ Error processing document batch: {str(e)}")
            raise
    
    def switch_model(self, model_name: str):
        """Switch to a different embedding model
        
        Args:
            model_name: Name of the new model
        """
        if model_name != self.model_name:
            logger.info(f"🔄 Switching model from {self.model_name} to {model_name}")
            self.model_name = model_name
            self._init_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model
        
        Returns:
            Model information
        """
        return {
            "model_name": self.model_name,
            "dimensions": 768,
            "max_sequence_length": 512,
            "model_type": "offline_processing"
        } 