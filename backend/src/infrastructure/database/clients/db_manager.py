import logging
from typing import Optional, Dict, Type
from .mongodb import MongoDBClient
from .milvus import MilvusClient
from .redis import RedisDB
from .base_database import BaseDatabase
from .elastic import ElasticClient
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_configs: Optional[Dict[str, Type[BaseDatabase]]] = None):
        self.databases: Dict[str, BaseDatabase] = {}
        
        if database_configs is None:
            database_configs = {
                'mongodb': MongoDBClient,
                'milvus': MilvusClient,
                # 'redis': RedisDB
                'elastic': ElasticClient
            }
        
        self._initialize_databases(database_configs)
    
    def _initialize_databases(self, configs: Dict[str, Type[BaseDatabase]]) -> None:
        for db_name, db_class in configs.items():
            try:
                self.databases[db_name] = db_class()
                logger.info(f"Initialized {db_name} database successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {db_name} database: {str(e)}")
    def get_mongodb(self):
        """Get MongoDB client"""
        return self.databases.get('mongodb')
    
    def get_milvus(self):
        """Get Milvus client"""
        return self.databases.get('milvus')
    
    def get_redis(self):
        """Get Redis client"""
        return self.databases.get('redis')
    def get_elastic(self):
        """Get Elastic client"""
        return self.databases.get('elastic')
    def close_connections(self) -> None:
        for db_name, db_instance in self.databases.items():
            try:
                db_instance.close()
                logger.info(f"Closed {db_name} connection")
            except Exception as e:
                logger.error(f"Error closing {db_name} connection: {str(e)}")
        
        logger.info("All database connections closed")