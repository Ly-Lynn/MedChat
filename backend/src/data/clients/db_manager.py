import logging
from typing import Dict, Type, List, Any, Optional
from ..clients.base_database import BaseDatabase
from .mongodb import MongoDB
from .milvus import MilvusDB
from .elastic import ElasticsearchDB

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, database_configs: Optional[Dict[str, Type[BaseDatabase]]] = None):
        self.databases: Dict[str, BaseDatabase] = {}
        
        # Default configuration
        if database_configs is None:
            database_configs = {
                'mongodb': MongoDB,
                'milvus': MilvusDB,
                'elasticsearch': ElasticsearchDB
            }
        
        self._initialize_databases(database_configs)
    
    def _initialize_databases(self, configs: Dict[str, Type[BaseDatabase]]) -> None:
        for db_name, db_class in configs.items():
            try:
                self.databases[db_name] = db_class()
                logger.info(f"Initialized {db_name} database successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {db_name} database: {str(e)}")
    
    def get_database(self, db_name: str) -> Optional[BaseDatabase]:
        return self.databases.get(db_name)
    
    def get_mongodb(self) -> Optional[MongoDB]:
        return self.databases.get('mongodb')
    
    def get_milvus(self) -> Optional[MilvusDB]:
        return self.databases.get('milvus')
    
    def get_elasticsearch(self) -> Optional[ElasticsearchDB]:
        return self.databases.get('elasticsearch')
    
    def add_database(self, db_name: str, db_instance: BaseDatabase) -> None:
        self.databases[db_name] = db_instance
        logger.info(f"Added {db_name} database to manager")
    
    def remove_database(self, db_name: str) -> None:
        if db_name in self.databases:
            self.databases[db_name].close()
            del self.databases[db_name]
            logger.info(f"Removed {db_name} database from manager")
    
    def get_available_databases(self) -> List[str]:
        return list(self.databases.keys())
    
    def check_connections(self) -> Dict[str, bool]:
        status = {}
        for db_name, db_instance in self.databases.items():
            try:
                status[db_name] = db_instance.ping()
            except Exception as e:
                logger.error(f"Error checking {db_name} connection: {str(e)}")
                status[db_name] = False
        return status
    
    def reconnect_all(self) -> None:
        for db_name, db_instance in self.databases.items():
            try:
                if not db_instance.is_connected():
                    db_instance._connect()
                    logger.info(f"Reconnected to {db_name} successfully")
            except Exception as e:
                logger.error(f"Failed to reconnect to {db_name}: {str(e)}")
    
    def close_connections(self) -> None:
        for db_name, db_instance in self.databases.items():
            try:
                db_instance.close()
                logger.info(f"Closed {db_name} connection")
            except Exception as e:
                logger.error(f"Error closing {db_name} connection: {str(e)}")
        
        logger.info("All database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connections()
    
    def __del__(self):
        try:
            self.close_connections()
        except:
            pass  # Ignore errors during cleanup 