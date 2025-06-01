import asyncio
import logging
from typing import List, Dict, Any
from src.domains.healthcare.embedding.services.offline_embedding_service import OfflineEmbeddingService
from src.domains.healthcare.medical_records.models.document import MedicalDocument
from src.infrastructure.database.repositories.document import DocumentRepository
from backend.src.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingWorker:
    """Background worker for processing document embeddings"""
    
    def __init__(self, model_name: str = "pubmedbert"):
        self.embedding_service = OfflineEmbeddingService(model_name=model_name)
        self.document_repo = DocumentRepository()
        self.is_running = False
    
    async def start(self):
        """Start the embedding worker"""
        self.is_running = True
        logger.info("Embedding worker started")
        
        while self.is_running:
            try:
                # Check for unprocessed documents
                unprocessed_docs = await self.get_unprocessed_documents()
                
                if unprocessed_docs:
                    logger.info(f"Found {len(unprocessed_docs)} unprocessed documents")
                    await self.process_documents_batch(unprocessed_docs)
                else:
                    # No documents to process, wait before checking again
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
            except Exception as e:
                logger.error(f"Error in embedding worker: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def stop(self):
        """Stop the embedding worker"""
        self.is_running = False
        logger.info("Embedding worker stopped")
    
    async def get_unprocessed_documents(self) -> List[MedicalDocument]:
        """Get documents that need embedding processing"""
        try:
            # Query documents without embeddings
            documents = await self.document_repo.find_documents_without_embeddings(limit=50)
            return documents
        except Exception as e:
            logger.error(f"Error fetching unprocessed documents: {str(e)}")
            return []
    
    async def process_documents_batch(self, documents: List[MedicalDocument]):
        """Process a batch of documents"""
        try:
            logger.info(f"Processing batch of {len(documents)} documents")
            
            # Process documents with embedding service
            processed_docs = await self.embedding_service.process_documents_batch(documents)
            
            # Update documents in database
            for doc in processed_docs:
                if doc.embeddings:  # Only update if embeddings were generated
                    await self.document_repo.update_document_embeddings(
                        doc.id, doc.embeddings, doc.metadata
                    )
                    logger.debug(f"Updated embeddings for document: {doc.title}")
            
            logger.info(f"Successfully processed {len(processed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error processing document batch: {str(e)}")
    
    async def process_single_document(self, document_id: str) -> bool:
        """Process a single document by ID"""
        try:
            # Get document from database
            document = await self.document_repo.get_document_by_id(document_id)
            if not document:
                logger.warning(f"Document not found: {document_id}")
                return False
            
            # Process document
            processed_doc = await self.embedding_service.process_document(document)
            
            # Update in database
            if processed_doc.embeddings:
                await self.document_repo.update_document_embeddings(
                    processed_doc.id, processed_doc.embeddings, processed_doc.metadata
                )
                logger.info(f"Successfully processed document: {document.title}")
                return True
            else:
                logger.warning(f"Failed to generate embeddings for: {document.title}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return False
    
    async def reprocess_all_documents(self, force: bool = False):
        """Reprocess all documents (useful for model updates)"""
        try:
            if force:
                # Get all documents
                documents = await self.document_repo.get_all_documents()
            else:
                # Get only documents without embeddings
                documents = await self.document_repo.find_documents_without_embeddings()
            
            logger.info(f"Reprocessing {len(documents)} documents")
            
            # Process in batches
            batch_size = 20
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                await self.process_documents_batch(batch)
                
                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(5)
            
            logger.info("Finished reprocessing all documents")
            
        except Exception as e:
            logger.error(f"Error reprocessing documents: {str(e)}")
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "is_running": self.is_running,
            "embedding_service_stats": self.embedding_service.get_processing_stats()
        }

# Global worker instance
embedding_worker = EmbeddingWorker()

async def start_embedding_worker():
    """Start the embedding worker"""
    await embedding_worker.start()

async def stop_embedding_worker():
    """Stop the embedding worker"""
    await embedding_worker.stop() 