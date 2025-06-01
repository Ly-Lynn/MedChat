from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TypeVar, Generic
from ..models.base import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseRepository(Generic[T], ABC):
    """Base repository interface"""
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> T:
        """Create new record"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get record by ID"""
        pass
    
    @abstractmethod
    async def get_by_filter(self, filter_dict: Dict[str, Any], 
                           limit: int = 100, skip: int = 0) -> List[T]:
        """Get records by filter"""
        pass
    
    @abstractmethod
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update record"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete record"""
        pass
    
    @abstractmethod
    async def count(self, filter_dict: Dict[str, Any] = None) -> int:
        """Count records"""
        pass