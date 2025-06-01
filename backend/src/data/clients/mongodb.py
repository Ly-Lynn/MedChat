import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from src.config.settings import settings
from typing import List, Dict, Any, Optional
from pymongo.collection import Collection
from ..clients.base_database import BaseDatabase

logger = logging.getLogger(__name__)

class MongoDBClient(BaseDatabase):
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        super().__init__()
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self) -> None:
        try:
            if not settings.MONGODB_URI:
                raise ValueError("MongoDB URI is not set in environment variables")
            
            self.client = MongoClient(settings.MONGODB_URI)
            self.db = self.client[settings.DB_NAME]
            
            self.ping()
            self._is_connected = True
            logger.info(f"Connected to MongoDB successfully - Database: {settings.DB_NAME}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self._is_connected = False
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            self._is_connected = False
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            raise
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        self.ensure_connection()
        if self.db:
            return self.db[collection_name]
        return None
    
    def create_collection(self, collection_name: str, schema: Optional[Dict[str, Any]] = None) -> Collection:
        self.ensure_connection()
        if not self.db:
            raise RuntimeError("Database connection not established")

        if schema:
            collection = self.db.create_collection(
                collection_name, 
                validator={"$jsonSchema": schema}
            )
        else:
            collection = self.db[collection_name]
        
        logger.info(f"Collection '{collection_name}' created successfully")
        return collection
    def update_document(self, collection_name: str, document: Dict[str, Any], doc_id: str) -> Any:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            result = collection.update_one(
                {"_id": doc_id},
                {"$set": document}
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
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
        
            document_copy = document.copy()
            document_copy['_id'] = doc_id
            
            try:
                result = collection.insert_one(document_copy)
                logger.info(f"Document '{doc_id}' indexed successfully in collection '{collection_name}'")
                return result
            except Exception as e:
                logger.error(f"Failed to index document '{doc_id}': {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error indexing document '{doc_id}': {str(e)}")
            raise
    
    def search(self, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            results = list(collection.find(query))
            logger.info(f"Search completed in collection '{collection_name}', found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Search failed in collection '{collection_name}': {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> None:
        self.ensure_connection()
        if not self.db:
            raise RuntimeError("Database connection not established")
        
        try:
            if collection_name in self.db.list_collection_names():
                self.db.drop_collection(collection_name)
                logger.info(f"Collection '{collection_name}' dropped successfully")
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
        except Exception as e:
            logger.error(f"Failed to drop collection '{collection_name}': {str(e)}")
            raise
    
    def delete_document(self, collection_name: str, doc_id: str) -> None:
        self.ensure_connection()
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                raise ValueError(f"Collection '{collection_name}' does not exist")
        
            result = collection.delete_one({"_id": doc_id})
            if result.deleted_count > 0:
                logger.info(f"Document '{doc_id}' deleted successfully from collection '{collection_name}'")
            else:
                logger.warning(f"Document '{doc_id}' not found in collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete document '{doc_id}': {str(e)}")
            raise
    
    def ping(self) -> bool:
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error(f"MongoDB ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        if self.client:
            self.client.close()
            self._is_connected = False
            logger.info("MongoDB connection closed")