import logging
from typing import Dict, Type, Optional
from .base_database import BaseDatabase
from .mongodb import MongoDB
from .milvus import MilvusDB
from .elastic import ElasticsearchDB

logger = logging.getLogger(__name__)

class DatabaseFactory:
    _database_registry: Dict[str, Type[BaseDatabase]] = {
        'mongodb': MongoDB,
        'milvus': MilvusDB,
        'elasticsearch': ElasticsearchDB
    }
    
    @classmethod
    def create_database(cls, db_type: str) -> Optional[BaseDatabase]:
        if db_type not in cls._database_registry:
            logger.error(f"Unknown database type: {db_type}")
            return None
        
        try:
            db_class = cls._database_registry[db_type]
            return db_class()
        except Exception as e:
            logger.error(f"Failed to create {db_type} database: {str(e)}")
            return None
    
    @classmethod
    def register_database(cls, db_type: str, db_class: Type[BaseDatabase]) -> None:
        """
        Register new database type
        
        Args:
            db_type: Database type
            db_class: Database class inheriting from BaseDatabase
        """
        if not issubclass(db_class, BaseDatabase):
            raise ValueError(f"Database class must inherit from BaseDatabase")
        
        cls._database_registry[db_type] = db_class
        logger.info(f"Registered new database type: {db_type}")
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available database types"""
        return list(cls._database_registry.keys())
    
    @classmethod
    def create_multiple(cls, db_types: list) -> Dict[str, BaseDatabase]:
        databases = {}
        for db_type in db_types:
            db_instance = cls.create_database(db_type)
            if db_instance:
                databases[db_type] = db_instance
        
        return databases 