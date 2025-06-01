from typing import Optional, List, Dict, Any
from pydantic import EmailStr, Field
from backend.src.shared.schemas.base_schemas import BaseEntity

class User(BaseEntity):
    """User domain model"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    roles: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict) 