from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings
from src.domains.healthcare.embedding.config.model_config import EmbeddingModelConfig

class BaseMedicalEmbedding(Embeddings, ABC):
    """Abstract base class for medical embedding models with LangChain integration"""
    
    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.dimensions = config.dimensions
        self.max_sequence_length = config.max_sequence_length
        self.device = config.device
        self.batch_size = config.batch_size
        
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
    
    @abstractmethod
    def _load_model(self):
        """Load the embedding model"""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for medical domain"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Truncate if too long
        if len(text) > self.max_sequence_length * 4:  # Rough estimate for tokens
            text = text[:self.max_sequence_length * 4]
            
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_sequence_length": self.max_sequence_length,
            "device": self.device,
            "batch_size": self.batch_size
        } 