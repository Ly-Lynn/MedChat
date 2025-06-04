import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from src.domains.healthcare.data_ingestion.crawlers.pubmed_crawler import PubMedCrawler
from src.domains.healthcare.embedding.models.biobert_embedding import BioBERTEmbedding
from src.domains.healthcare.data_ingestion.schemas.pubmed_schemas import PubMedArticle
from src.config.settings import settings

logger = logging.getLogger(__name__)

class DataIngestionResult:
    """Result of data ingestion operation"""
    
    def __init__(self):
        self.job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.total_processed = 0
        self.processing_time = 0.0
        self.errors = []

class DataIngestionService:
    """Service for ingesting healthcare data"""
    
    def __init__(self):
        """Initialize data ingestion service"""
        self.crawler = PubMedCrawler()
        self.embedding_model = BioBERTEmbedding(
            device = settings.DEVICE
        )
        self.stats = {
            "total_documents": 0,
            "total_jobs": 0,
            "last_ingestion": None
        }
    def _get_embedding_data(self, articles: List[PubMedArticle]) -> List[Dict[str, Any]]:
        """Get embedding data from articles"""
        # pass
        for article in articles:
            embedding_data = {
                "id": article.pmid,
                "title": article.title,
                "abstract": article.abstract,
                "authors": article.authors,
                "sections": [f"{section.heading}: {section.text}" for section in article.sections]
            }
            self.embedding_model.embed_document(embedding_data)

            
    async def ingest_from_queries(self, queries: List[str], articles_per_query: int = 10, 
                                generate_embeddings: bool = True) -> DataIngestionResult:
        """Ingest data from search queries
        
        Args:
            queries: List of search queries
            articles_per_query: Number of articles per query
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Ingestion result
        """
        result = DataIngestionResult()
        start_time = datetime.now()
        
        try:
            logger.info(f"🔄 Starting data ingestion for queries: {queries}")
            
            total_processed = 0
            print("queries", queries)
            for query in queries:
                # crawling articles with crawler
                articles = self.crawler.crawl_by_query(query, articles_per_query)
                # embedding articles with embedding model
                # embedding_datas = self.get_embedding_data(articles)
                
                # embeddings = self.embedding_model.embed_documents(articles)

                processed_count = len(articles)
                total_processed += processed_count
                logger.info(f"✅ Processed {processed_count} articles for query: {query}")

                # DEBUG
                print("ARTICLES", articles[0])
                print("len(articles)", len(articles))
                break
                # # Mock processing
                # await asyncio.sleep(0.1)  # Simulate processing time
                # processed_count = min(articles_per_query, 5)  # Mock limited results
                # total_processed += processed_count
                
                # logger.info(f"✅ Processed {processed_count} articles for query: {query}")
            
            result.total_processed = total_processed
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.stats["total_documents"] += total_processed
            self.stats["total_jobs"] += 1
            self.stats["last_ingestion"] = datetime.now().isoformat()
            
            logger.info(f"✅ Data ingestion completed. Processed {total_processed} documents")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during data ingestion: {str(e)}")
            result.errors.append(str(e))
            raise
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_documents_ingested": self.stats["total_documents"],
            "total_ingestion_jobs": self.stats["total_jobs"],
            "last_ingestion_time": self.stats["last_ingestion"],
            "service_status": "active"
        }

class MockCrawler:
    """Mock crawler for testing"""
    
    def search_metadata(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for document metadata
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of document metadata
        """
        # Mock search results
        results = []
        for i in range(min(limit, 3)):  # Return max 3 mock results
            results.append({
                "title": f"Medical Research Paper {i+1} about {query}",
                "authors": ["Dr. Smith", "Dr. Johnson"],
                "abstract": f"This paper discusses {query} and related medical topics...",
                "doi": f"10.1234/example.{query}.{i+1}",
                "publication_date": "2023-01-15"
            })
        
        return results

def create_data_ingestion_service() -> DataIngestionService:
    """Factory function to create data ingestion service
    
    Returns:
        Data ingestion service instance
    """
    return DataIngestionService() 