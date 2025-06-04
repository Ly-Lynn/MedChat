from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Processor(ABC):
    """
    Base abstract class for all document processors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processor with configuration
        
        Args:
            config: Configuration dictionary for processor
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process the input data
        
        Args:
            data: Input data to process
            **kwargs: Additional parameters
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data before processing
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def pre_process(self, data: Any) -> Any:
        """
        Pre-processing steps before main processing
        
        Args:
            data: Input data
            
        Returns:
            Pre-processed data
        """
        return data
    
    def post_process(self, data: Any) -> Any:
        """
        Post-processing steps after main processing
        
        Args:
            data: Processed data
            
        Returns:
            Post-processed data
        """
        return data
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get processor metadata
        
        Returns:
            Metadata dictionary
        """
        return {
            "processor_name": self.__class__.__name__,
            "config": self.config,
            "created_at": datetime.utcnow().isoformat()
        }
    