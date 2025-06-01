from typing import Optional, List, Dict, Any
from pydantic import Field
from .base import BaseModel

class Document(BaseModel):
    title: str
    content: Dict[str, str]
    _id: str
    source: str  # pubmed, manual, etc.
    source_id: Optional[str] = None  # external ID
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    embeddings: Optional[List[float]] = None
    status: str = "active"  # active, archived, deleted