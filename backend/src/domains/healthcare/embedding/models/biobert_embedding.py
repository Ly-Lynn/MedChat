from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from src.domains.healthcare.embedding.models.base_embedding import BaseMedicalEmbedding
from src.domains.healthcare.embedding.config.model_config import ModelConfigs

class BioBERTEmbedding(BaseMedicalEmbedding):
    """BioBERT embedding implementation with LangChain integration"""
    
    def __init__(self, device: str = "cpu"):
        config = ModelConfigs.BIOBERT
        config.device = device
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load BioBERT model and tokenizer"""
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
            raise RuntimeError(f"Failed to load BioBERT model: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BioBERT"""
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Use LangChain's embedding method for consistency
        embeddings = self.langchain_embeddings.embed_documents(processed_texts)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using BioBERT"""
        processed_text = self.preprocess_text(text)
        
        # Use LangChain's embedding method
        embedding = self.langchain_embeddings.embed_query(processed_text)
        
        return embedding
    
    def _embed_with_raw_model(self, texts: List[str]) -> List[List[float]]:
        """Alternative embedding method using raw model (for custom processing)"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_sequence_length
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding or mean pooling
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings.append(embedding.tolist())
        
        return embeddings
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text specifically for medical BioBERT"""
        # Basic preprocessing
        text = super().preprocess_text(text)
        
        # Medical-specific preprocessing
        # Convert to lowercase for BioBERT
        text = text.lower()
        
        # Handle medical abbreviations and terms
        # Add more medical-specific preprocessing here if needed
        
        return text 