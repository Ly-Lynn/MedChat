import logging
from src.config.settings import settings
from typing import Dict, Any, Optional
from .base_database import BaseDatabase
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

class ElasticClient(BaseDatabase):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ElasticClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        super().__init__()
        self.client = None
        self._connect()
    
    def _connect(self) -> None:
        try:
            if not settings.ELASTICSEARCH_URI:
                raise ValueError("Elasticsearch URI is not set in environment variables")
            
            self.client = Elasticsearch(
                settings.ELASTICSEARCH_URI,
                headers={
                    "Accept": "application/vnd.elasticsearch+json;compatible-with=8",
                    "Content-Type": "application/vnd.elasticsearch+json;compatible-with=8"
                }
            )
            
            self._is_connected = True
            logger.info("Connected to Elasticsearch successfully")
            
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
    
    def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        self.ensure_connection()
        try:
            if self.client.indices.exists(index=collection_name):
                response = self.client.search(
                    index=collection_name,
                    body={
                        "query": {"match_all": {}},
                        "size": 1000
                    }
                )
                return response
            else:
                logger.error(f"Index '{collection_name}' does not exist")
                return None
        except Exception as e:
            logger.error(f"Error getting index '{collection_name}': {str(e)}")
            return None
    
    def create_collection(self, collection_name: str, schema: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            if not self.client.indices.exists(index=collection_name):
                result = self.client.indices.create(index=collection_name, mappings=schema)
                logger.info(f"Created index '{collection_name}' successfully")
                return result
            else:
                logger.info(f"Index '{collection_name}' already exists")
                return None
        except Exception as e:
            logger.error(f"Error creating index '{collection_name}': {str(e)}")
            raise
    def update_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        self.ensure_connection()
        try:
            result = self.client.update(
                index=collection_name,
                document=document,
                id=doc_id
            )
            logger.info(f"Updated document '{doc_id}' successfully: {result['result']}")
            return result
        except Exception as e:
            logger.error(f"Error updating document '{doc_id}': {str(e)}")
            raise
    def index_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        self.ensure_connection()
        try:
            result = self.client.index(
                index=collection_name,
                document=document,
                id=doc_id
            )
            logger.info(f"Indexed document '{doc_id}' successfully: {result['result']}")
            return result
        except Exception as e:
            logger.error(f"Error indexing document '{doc_id}': {str(e)}")
            raise
    
    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            result = self.client.search(
                index=collection_name,
                query=query
            )
            return result
        except Exception as e:
            logger.error(f"Error searching in index '{collection_name}': {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        try:
            if self.client.indices.exists(index=collection_name):
                self.client.indices.delete(index=collection_name)
                logger.info(f"Deleted index '{collection_name}' successfully")
            else:
                logger.error(f"Index '{collection_name}' does not exist")
        except Exception as e:
            logger.error(f"Error deleting index '{collection_name}': {str(e)}")
            raise
    
    def delete_document(self, collection_name: str, doc_id: str) -> None:
        self.ensure_connection()
        try:
            result = self.client.delete(index=collection_name, id=doc_id)
            logger.info(f"Deleted document '{doc_id}' successfully from index '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}' from index '{collection_name}': {str(e)}")
            raise
    
    def ping(self) -> bool:
        try:
            if self.client:
                return self.client.ping()
            return False
        except Exception as e:
            logger.error(f"Elasticsearch ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        if self.client:
            self.client.close()
            self._is_connected = False
            logger.info("Elasticsearch connection closed") 