import logging
from pymilvus import (
    connections, 
    Collection,
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType
)
from src.config.settings import settings
from typing import List, Dict, Any, Optional
from src.data.clients.base_database import BaseDatabase

logger = logging.getLogger(__name__)

class MilvusClient(BaseDatabase):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        super().__init__()
        self._connect()
    
    def _connect(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            
            self._is_connected = True
            logger.info("Connected to Milvus successfully")
            
        except Exception as e:
            self._is_connected = False
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        self.ensure_connection()
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info(f"Collection '{collection_name}' loaded successfully")
            return collection
        else:
            logger.error(f"Collection '{collection_name}' does not exist")
            return None
    
    def create_collection(self, collection_name: str, schema: List[FieldSchema]) -> Collection:
        self.ensure_connection()
        if utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        collection_schema = CollectionSchema(schema, collection_name)
        collection = Collection(name=collection_name, schema=collection_schema)
        logger.info(f"Collection '{collection_name}' created successfully")
        return collection
    

    def update_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")        
            result = collection.update(
                expr=f"id == '{doc_id}'",
                data=document
            )
            logger.info(f"Document '{doc_id}' updated successfully in collection '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error updating document '{doc_id}': {str(e)}")
            raise
        
    def index_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if 'id' not in document:
                document['id'] = doc_id
            
            result = collection.insert([document])
            logger.info(f"Document indexed successfully in collection '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error indexing document '{doc_id}': {str(e)}")
            raise
    
    def search(self, collection_name: str, query: Dict[str, Any]) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
                
            search_params = query.get('search_params', {})
            data = query.get('data', [])
            anns_field = query.get('anns_field', 'embedding')
            param = query.get('param', {})
            limit = query.get('limit', 10)
            
            results = collection.search(
                data=data,
                anns_field=anns_field,
                param=param,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error searching in collection '{collection_name}': {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        try:
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
            collection.drop()
            logger.info(f"Collection '{collection_name}' dropped successfully")
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {str(e)}")
            raise
    
    def delete_document(self, collection_name: str, doc_id: str) -> None:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            expr = f"id == '{doc_id}'"
            result = collection.delete(expr)
            logger.info(f"Document '{doc_id}' deleted successfully from collection '{collection_name}'")
            return result
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {str(e)}")
            raise
    
    def ping(self) -> bool:
        try:
            utility.list_collections()
            return True
        except Exception as e:
            logger.error(f"Milvus ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        try:
            connections.disconnect("default")
            self._is_connected = False
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {str(e)}") 