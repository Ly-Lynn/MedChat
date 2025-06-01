"""Medical document model"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import Field, ConfigDict
from src.shared.schemas.base_schemas import BaseEntity

class MedicalDocument(BaseEntity):
    """Medical document model"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Diabetes Management Guidelines",
                "content": {
                    "abstract": "This document provides guidelines for diabetes management...",
                    "body": "Diabetes mellitus is a chronic condition..."
                },
                "source": "pubmed",
                "tags": ["diabetes", "management", "guidelines"],
                "metadata": {
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "publication_date": "2023-01-15",
                    "journal": "Medical Journal"
                }
            }
        }
    )
    
    title: str = Field(..., description="Document title")
    content: Dict[str, Any] = Field(default_factory=dict, description="Document content")
    source: str = Field(default="manual", description="Source of the document")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    
    # Embedding-related fields
    embeddings: Optional[List[List[float]]] = Field(default=None, description="Document embeddings")
    processed: bool = Field(default=False, description="Whether document has been processed")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 