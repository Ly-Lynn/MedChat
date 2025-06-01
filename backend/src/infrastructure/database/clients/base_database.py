from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseDatabase(ABC):
    """Abstract base class for all database implementations"""
    
    def __init__(self):
        self._connection = None
        self._is_connected = False
    
    @abstractmethod
    def _connect(self) -> None:
        """Set up database connection"""
        pass
    
    @abstractmethod
    def get_collection(self, collection_name: str) -> Any:
        """Get collection/index by name"""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, schema: Any) -> Any:
        """Create new collection/index"""
        pass
    
    @abstractmethod
    def index_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        """Add document to collection"""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        """Search in collection"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete collection/index"""
        pass
    
    @abstractmethod
    def delete_document(self, collection_name: str, doc_id: str) -> None:
        """Delete document by ID"""
        pass
    
    @abstractmethod
    def update_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        """Update document by ID"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Disconnect from the database"""
        pass
    
    @abstractmethod
    def ping(self) -> bool:
        """Check database connection"""
        pass
    
    def is_connected(self) -> bool:
        """Check database connection status"""
        return self._is_connected
    
    def ensure_connection(self) -> None:
        """Ensure connection, reconnect if not connected"""
        if not self.is_connected():
            self._connect() 