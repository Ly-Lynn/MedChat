from typing import Optional, List, Dict, Any
from bson import ObjectId
from src.data.clients.mongodb import MongoDBClient
from src.data.clients.milvus import MilvusClient
from src.data.clients.elastic import ElasticClient
from src.data.models.document import Document
from src.data.repositories.base import BaseRepository
import logging
from src.common.config import Config

class DocumentRepository(BaseRepository[Document]):
    def __init__(self):
        self.mongodb_client = MongoDBClient()
        self.milvus_client = MilvusClient()
        self.elastic_client = ElasticClient()
        self.mongodb_collection = self.mongodb_client.get_collection(Config.MONGO_COLLECTION_NAME)
        self.milvus_collection = self.milvus_client.get_collection(Config.MILVUS_COLLECTION_NAME)
        self.elastic_collection = self.elastic_client.get_collection(Config.ELASTIC_COLLECTION_NAME)
        self.logger = logging.getLogger(__name__)
    
    async def create(self, data: Dict[str, Any]) -> Document:
        """Create a new document"""
        milvus_data = data["milvus"]
        elastic_data = data["elastic"]
        mongo_data = data["mongo"]

        # Create document in MongoDB
        mongo_result = await self.mongodb_collection.insert_one(mongo_data)

        # Create document in Milvus
        milvus_result = await self.milvus_collection.insert_one(milvus_data)

        # Create document in Elasticsearch
        elastic_result = await self.elastic_collection.insert_one(elastic_data)
        
        _data = {
            "_id": data["_id"],
            "content": data["content"],
            "source": data["source"],
            "source_id": data["source_id"],
            "metadata": data["metadata"],
            "tags": data["tags"],
            "embeddings": data["embeddings"]
        }
        return Document(**_data)

    async def vector_search(self, query_embedding: List[float], 
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Search similar documents using vector similarity"""
        try:
            collection = self.milvus_client.get_collection("documents")
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding_vector",
                param=search_params,
                limit=top_k,
                expr=None,
                output_fields=["document_id"]
            )
            
            # Get full documents from MongoDB
            doc_ids = [hit.entity.get("document_id") for hit in results[0]]
            documents = await self.get_by_filter({"_id": {"$in": [ObjectId(id) for id in doc_ids]}})
            
            return documents
        except Exception as e:
            self.logger.error(f"Failed to perform vector search: {e}")
            return []