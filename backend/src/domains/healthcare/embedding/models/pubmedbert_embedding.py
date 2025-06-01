from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from src.domains.healthcare.embedding.models.base_embedding import BaseMedicalEmbedding
from src.domains.healthcare.embedding.config.model_config import ModelConfigs

class PubMedBERTEmbedding(BaseMedicalEmbedding):
    """PubMedBERT embedding implementation with LangChain integration"""
    
    def __init__(self, device: str = "cpu"):
        config = ModelConfigs.PUBMEDBERT
        config.device = device
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load PubMedBERT model and tokenizer"""
        try:
            # Use LangChain's HuggingFaceEmbeddings for better integration
            self.langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_path,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Also load raw model for custom processing if needed
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModel.from_pretrained(self.config.model_path)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PubMedBERT model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using PubMedBERT"""
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Use LangChain's embedding method for consistency
        embeddings = self.langchain_embeddings.embed_documents(processed_texts)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using PubMedBERT"""
        processed_text = self.preprocess_text(text)
        
        # Use LangChain's embedding method
        embedding = self.langchain_embeddings.embed_query(processed_text)
        
        return embedding
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text specifically for PubMedBERT"""
        # Basic preprocessing
        text = super().preprocess_text(text)
        
        # PubMedBERT specific preprocessing
        # Keep original case as PubMedBERT is uncased but trained on abstracts
        text = text.lower()
        
        # Handle PubMed-specific formatting
        # Remove common PubMed artifacts
        text = text.replace("BACKGROUND:", "").replace("METHODS:", "")
        text = text.replace("RESULTS:", "").replace("CONCLUSIONS:", "")
        text = text.replace("OBJECTIVE:", "").replace("PURPOSE:", "")
        
        return text.strip() 