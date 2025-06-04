from typing import Any, Dict, List, Optional, Union, Callable
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# LangChain imports
from langchain.vectorstores import FAISS, Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# Local imports
from .scientific_paper_processor import ScientificPaperProcessor
from ..schemas.pubmed_schemas import PubMedArticle, CrawlResult
from ..parsers.xml_parser import XMLParser

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """
    Complete pipeline for processing scientific documents
    Handles the entire flow from raw documents to processed, embedded, and indexed data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize processing pipeline
        
        Args:
            config: Configuration dictionary containing:
                - processor_config: Configuration for ScientificPaperProcessor
                - vectorstore_type: Type of vector store ('faiss', 'chroma', 'memory')
                - vectorstore_config: Configuration for vector store
                - batch_size: Batch size for processing (default: 10)
                - max_workers: Max workers for parallel processing (default: 4)
                - enable_async: Enable async processing (default: True)
                - quality_threshold: Minimum quality score for documents (default: 3.0)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Pipeline configuration
        self.batch_size = self.config.get('batch_size', 10)
        self.max_workers = self.config.get('max_workers', 4)
        self.enable_async = self.config.get('enable_async', True)
        self.quality_threshold = self.config.get('quality_threshold', 3.0)
        
        # Initialize components
        self._init_processor()
        self._init_vectorstore()
        
        # Statistics
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None
        }
        
        self.logger.info(f"Initialized DocumentProcessingPipeline with config: {self.config}")
    
    def _init_processor(self):
        """Initialize the scientific paper processor"""
        processor_config = self.config.get('processor_config', {})
        self.processor = ScientificPaperProcessor(processor_config)
    
    def _init_vectorstore(self):
        """Initialize vector store based on configuration"""
        vectorstore_type = self.config.get('vectorstore_type', 'faiss')
        vectorstore_config = self.config.get('vectorstore_config', {})
        
        self.vectorstore = None
        self.vectorstore_type = vectorstore_type
        self.vectorstore_config = vectorstore_config
        
        self.logger.info(f"Vector store will be initialized on first use: {vectorstore_type}")
    
    def _create_vectorstore(self, documents: List[Dict[str, Any]]) -> Any:
        """Create vector store from processed documents"""
        if not documents or not self.processor.embeddings:
            return None
        
        try:
            # Extract text and embeddings
            texts = []
            embeddings = []
            metadatas = []
            
            for doc in documents:
                for i, chunk in enumerate(doc["chunks"]):
                    texts.append(chunk["content"])
                    metadatas.append(chunk["metadata"])
                    
                    # Use embeddings if available
                    if doc.get("embeddings") and i < len(doc["embeddings"]):
                        embeddings.append(doc["embeddings"][i])
            
            # Create vector store
            if self.vectorstore_type == 'faiss':
                if embeddings:
                    vectorstore = FAISS.from_embeddings(
                        text_embeddings=list(zip(texts, embeddings)),
                        embedding=self.processor.embeddings,
                        metadatas=metadatas
                    )
                else:
                    vectorstore = FAISS.from_texts(
                        texts=texts,
                        embedding=self.processor.embeddings,
                        metadatas=metadatas
                    )
            elif self.vectorstore_type == 'chroma':
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=self.processor.embeddings,
                    metadatas=metadatas,
                    **self.vectorstore_config
                )
            else:
                # Memory-based vector store (for testing)
                vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.processor.embeddings,
                    metadatas=metadatas
                )
            
            self.logger.info(f"Created {self.vectorstore_type} vector store with {len(texts)} documents")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    async def process_documents_async(
        self, 
        documents: List[Union[PubMedArticle, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process documents asynchronously
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            Processing results
        """
        self.processing_stats["start_time"] = datetime.utcnow()
        self.processing_stats["total_processed"] = len(documents)
        
        processed_documents = []
        errors = []
        
        try:
            # Process in batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                self.logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(documents)-1)//self.batch_size + 1}")
                
                # Process batch
                batch_results = await self._process_batch_async(batch, **kwargs)
                
                for result in batch_results:
                    if result.get("success"):
                        processed_documents.append(result["data"])
                        self.processing_stats["successful"] += 1
                    else:
                        errors.append(result.get("error", "Unknown error"))
                        self.processing_stats["failed"] += 1
            
            # Create vector store if documents were processed successfully
            vectorstore = None
            if processed_documents:
                vectorstore = self._create_vectorstore(processed_documents)
                self.vectorstore = vectorstore
            
            self.processing_stats["end_time"] = datetime.utcnow()
            processing_time = (self.processing_stats["end_time"] - self.processing_stats["start_time"]).total_seconds()
            
            results = {
                "success": True,
                "processed_documents": processed_documents,
                "vectorstore": vectorstore,
                "statistics": {
                    **self.processing_stats,
                    "processing_time_seconds": processing_time,
                    "documents_per_second": len(processed_documents) / processing_time if processing_time > 0 else 0
                },
                "errors": errors
            }
            
            self.logger.info(f"Pipeline completed: {self.processing_stats['successful']} successful, {self.processing_stats['failed']} failed")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_documents": processed_documents,
                "statistics": self.processing_stats,
                "errors": errors
            }
    
    def process_documents(
        self, 
        documents: List[Union[PubMedArticle, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process documents synchronously
        
        Args:
            documents: List of documents to process
            **kwargs: Additional parameters
            
        Returns:
            Processing results
        """
        if self.enable_async:
            return asyncio.run(self.process_documents_async(documents, **kwargs))
        else:
            return self._process_documents_sync(documents, **kwargs)
    
    def _process_documents_sync(
        self, 
        documents: List[Union[PubMedArticle, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous document processing"""
        self.processing_stats["start_time"] = datetime.utcnow()
        self.processing_stats["total_processed"] = len(documents)
        
        processed_documents = []
        errors = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Process in batches
                for i in range(0, len(documents), self.batch_size):
                    batch = documents[i:i + self.batch_size]
                    
                    # Submit batch processing tasks
                    futures = [
                        executor.submit(self._process_single_document, doc, **kwargs)
                        for doc in batch
                    ]
                    
                    # Collect results
                    for future in futures:
                        try:
                            result = future.result()
                            if result.get("success"):
                                processed_documents.append(result["data"])
                                self.processing_stats["successful"] += 1
                            else:
                                errors.append(result.get("error", "Unknown error"))
                                self.processing_stats["failed"] += 1
                        except Exception as e:
                            errors.append(str(e))
                            self.processing_stats["failed"] += 1
            
            # Create vector store
            vectorstore = None
            if processed_documents:
                vectorstore = self._create_vectorstore(processed_documents)
                self.vectorstore = vectorstore
            
            self.processing_stats["end_time"] = datetime.utcnow()
            processing_time = (self.processing_stats["end_time"] - self.processing_stats["start_time"]).total_seconds()
            
            return {
                "success": True,
                "processed_documents": processed_documents,
                "vectorstore": vectorstore,
                "statistics": {
                    **self.processing_stats,
                    "processing_time_seconds": processing_time,
                    "documents_per_second": len(processed_documents) / processing_time if processing_time > 0 else 0
                },
                "errors": errors
            }
            
        except Exception as e:
            self.logger.error(f"Sync processing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_documents": processed_documents,
                "statistics": self.processing_stats,
                "errors": errors
            }
    
    async def _process_batch_async(
        self, 
        batch: List[Union[PubMedArticle, Dict[str, Any]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents asynchronously"""
        tasks = [
            asyncio.create_task(self._process_single_document_async(doc, **kwargs))
            for doc in batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_single_document_async(
        self, 
        document: Union[PubMedArticle, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single document asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_single_document, document, **kwargs)
    
    def _process_single_document(
        self, 
        document: Union[PubMedArticle, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single document"""
        try:
            # Validate document
            if not self.processor.validate_input(document):
                return {
                    "success": False,
                    "error": "Document validation failed"
                }
            
            # Process document
            result = self.processor.process(document, **kwargs)
            
            # Check quality threshold
            quality_score = result.get("metadata", {}).get("quality_score", 0)
            if quality_score < self.quality_threshold:
                self.processing_stats["skipped"] += 1
                return {
                    "success": False,
                    "error": f"Document quality score {quality_score} below threshold {self.quality_threshold}"
                }
            
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a retriever from the processed vector store
        
        Args:
            search_kwargs: Search parameters for retriever
            
        Returns:
            LangChain retriever or None
        """
        if not self.vectorstore:
            self.logger.warning("No vector store available for retriever creation")
            return None
        
        try:
            search_kwargs = search_kwargs or {"k": 5}
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            
            self.logger.info(f"Created retriever with search_kwargs: {search_kwargs}")
            return retriever
            
        except Exception as e:
            self.logger.error(f"Error creating retriever: {str(e)}")
            return None
    
    def create_ensemble_retriever(
        self, 
        retrievers: List[Any],
        weights: Optional[List[float]] = None
    ) -> Any:
        """
        Create an ensemble retriever combining multiple retrievers
        
        Args:
            retrievers: List of retrievers to ensemble
            weights: Weights for each retriever
            
        Returns:
            Ensemble retriever
        """
        try:
            if not retrievers:
                return self.create_retriever()
            
            if weights and len(weights) != len(retrievers):
                weights = [1.0 / len(retrievers)] * len(retrievers)
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights or [1.0 / len(retrievers)] * len(retrievers)
            )
            
            self.logger.info(f"Created ensemble retriever with {len(retrievers)} retrievers")
            return ensemble_retriever
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble retriever: {str(e)}")
            return None
    
    def save_vectorstore(self, path: str) -> bool:
        """
        Save vector store to disk
        
        Args:
            path: Path to save vector store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            self.logger.warning("No vector store to save")
            return False
        
        try:
            if hasattr(self.vectorstore, 'save_local'):
                self.vectorstore.save_local(path)
                self.logger.info(f"Vector store saved to {path}")
                return True
            else:
                self.logger.warning("Vector store does not support local saving")
                return False
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vectorstore(self, path: str) -> bool:
        """
        Load vector store from disk
        
        Args:
            path: Path to load vector store from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vectorstore_type == 'faiss':
                self.vectorstore = FAISS.load_local(
                    path, 
                    self.processor.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                self.logger.warning(f"Loading not implemented for {self.vectorstore_type}")
                return False
            
            self.logger.info(f"Vector store loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        return {
            "config": self.config,
            "processor_initialized": self.processor is not None,
            "vectorstore_initialized": self.vectorstore is not None,
            "vectorstore_type": self.vectorstore_type,
            "processing_stats": self.processing_stats,
            "embeddings_model": self.processor.embedding_model_name if self.processor else None
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None
        }
        self.logger.info("Processing statistics reset") 