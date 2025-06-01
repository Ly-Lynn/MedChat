from typing import List, Optional, Dict, Any
from .base import CoreBaseModel
from datetime import datetime
from pydantic import Field
from .base import PyObjectId

class Message(CoreBaseModel):
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Conversation(CoreBaseModel):
    user_id: PyObjectId
    title: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"  # active, archived, deleted