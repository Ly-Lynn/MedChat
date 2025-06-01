from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    model_name: str
    model_path: str
    dimensions: int
    max_sequence_length: int
    device: str = "cpu"
    batch_size: int = 32
    cache_enabled: bool = True
    
class ModelConfigs:
    """Configuration for all supported embedding models"""
    
    BIOBERT = EmbeddingModelConfig(
        model_name="biobert",
        model_path="dmis-lab/biobert-base-cased-v1.1",
        dimensions=768,
        max_sequence_length=512,
        device="cpu",
        batch_size=16,
        cache_enabled=True
    )
    
    PUBMEDBERT = EmbeddingModelConfig(
        model_name="pubmedbert", 
        model_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        dimensions=768,
        max_sequence_length=512,
        device="cpu",
        batch_size=16,
        cache_enabled=True
    )
    
    CLINICAL_BERT = EmbeddingModelConfig(
        model_name="clinicalbert",
        model_path="emilyalsentzer/Bio_ClinicalBERT",
        dimensions=768,
        max_sequence_length=512,
        device="cpu", 
        batch_size=16,
        cache_enabled=True
    )
    
    @classmethod
    def get_config(cls, model_name: str) -> EmbeddingModelConfig:
        """Get configuration for specific model"""
        configs = {
            "biobert": cls.BIOBERT,
            "pubmedbert": cls.PUBMEDBERT,
            "clinicalbert": cls.CLINICAL_BERT
        }
        
        if model_name not in configs:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return configs[model_name]
    
    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available model names"""
        return ["biobert", "pubmedbert", "clinicalbert"] 