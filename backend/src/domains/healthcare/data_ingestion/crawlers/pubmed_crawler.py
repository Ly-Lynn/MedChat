import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import json
from bs4 import BeautifulSoup
import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from backend.src.domains.healthcare.data_ingestion.schemas.pubmed_schemas import PubMedArticle, CrawlerConfig
from backend.src.domains.healthcare.data_ingestion.parsers.xml_parser import XMLParser
from src.config.settings import settings

logger = logging.getLogger(__name__)

class PubMedCrawler:
    """
    PubMed/EuropePMC crawler for medical literature data ingestion
    
    This crawler is part of the healthcare data ingestion subdomain and is responsible
    for fetching medical research papers from PubMed/EuropePMC APIs.
    """
    
    def __init__(self, retmode="xml", config: Optional[CrawlerConfig] = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.europepmc_base = "https://www.ebi.ac.uk/europepmc/webservices/rest"
        self.retmode = retmode
        self.config = config or CrawlerConfig()
        
        # Initialize XML parser
        self.xml_parser = XMLParser()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
        # Configure headers
        self.session.headers.update({
            'User-Agent': self.config.user_agent
        })

    def _make_request(self, url: str, params=None) -> requests.Response:
        """Make HTTP request with error handling and rate limiting"""
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=(self.config.connect_timeout, self.config.read_timeout)
            )
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error when accessing {url}: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout when accessing {url}: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error when accessing {url}: {str(e)}")
            raise

    def search_metadata(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for article metadata using EuropePMC API
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of article metadata dictionaries
        """
        search_limit = limit or self.config.default_limit
        url = f"{self.europepmc_base}/search"
        
        params = {
            "query": f"{query} AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": min(search_limit, 1000),  # EuropePMC max page size
            "resultType": "core"
        }
        
        try:
            response = self._make_request(url, params=params)
            data = json.loads(response.text)
            
            if 'resultList' not in data or 'result' not in data['resultList']:
                logger.warning(f"No results found for query: {query}")
                return []
                
            results = data['resultList']['result']
            processed_results = []
            
            for item in results[:search_limit]:
                try:
                    processed_item = self._process_metadata_item(item)
                    if processed_item:
                        processed_results.append(processed_item)
                except Exception as e:
                    logger.error(f"Failed to process metadata item: {str(e)}")
                    continue
                    
            logger.info(f"Successfully processed {len(processed_results)} articles for query: {query}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search metadata for query '{query}': {str(e)}")
            return []

    def _process_metadata_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single metadata item from EuropePMC response"""
        # Check if article has PMC ID (required for full text access)
        if 'pmcid' not in item or not item['pmcid']:
            return None
            
        return {
            'id': item.get('id'),
            'doi': item.get('doi'),
            'pmid': item.get('pmid'),
            'pmcid': item.get('pmcid'),
            'textId': [sub_item for sub_item in item['fullTextIdList']['fullTextId']] if 'fullTextIdList' in item else None,
            'title': item.get('title', '').strip(),
            'authors': item.get('authorString', '').split(', ') if 'authorString' in item else [],
            'journal': item.get('journalTitle', '').strip(),
            'pubYear': item.get('pubYear'),
            'journalVolume': item.get('journalVolume'),
            'pageInfo': item.get('pageInfo'),
            'abstractText': item.get('abstractText', '').strip(),
            'keywords': item.get('keywordList', {}).get('keyword', []) if 'keywordList' in item else []
        }

    def get_full_text_articles(self, metadata_list: List[Dict[str, Any]]) -> List[PubMedArticle]:
        """
        Fetch full text articles from EuropePMC
        
        Args:
            metadata_list: List of article metadata
            
        Returns:
            List of PubMedArticle objects
        """
        articles = []
        
        for metadata in tqdm(metadata_list, desc="Fetching full text articles"):
            if not metadata.get('pmcid'):
                logger.warning(f"No PMC ID found for article: {metadata.get('title', 'Unknown')}")
                continue
                
            try:
                article = self._fetch_single_article(metadata)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.error(f"Failed to fetch article {metadata.get('pmcid')}: {str(e)}")
                continue
                
        logger.info(f"Successfully fetched {len(articles)} full text articles")
        return articles

    def _fetch_single_article(self, metadata: Dict[str, Any]) -> Optional[PubMedArticle]:
        """Fetch a single article's full text"""
        pmcid = metadata['pmcid']
        url = f"{self.europepmc_base}/{pmcid}/fullTextXML"
        
        try:
            response = self._make_request(url)
            
            # Parse XML content
            parsed_content = self.xml_parser.parse_pubmed_xml(response.text, metadata)
            
            # Create PubMedArticle object
            article = PubMedArticle(**parsed_content)
            
            logger.debug(f"Successfully fetched article: {article.title}")
            return article
            
        except Exception as e:
            logger.error(f"Failed to fetch full text for {pmcid}: {str(e)}")
            return None

    def crawl_by_queries(self, queries: List[str], articles_per_query: int = 10) -> List[PubMedArticle]:
        """
        Crawl articles for multiple queries
        
        Args:
            queries: List of search queries
            articles_per_query: Maximum articles to fetch per query
            
        Returns:
            List of all fetched articles
        """
        all_articles = []
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            # Search for metadata
            metadata_list = self.search_metadata(query, limit=articles_per_query)
            
            if metadata_list:
                # Fetch full text articles
                articles = self.get_full_text_articles(metadata_list)
                all_articles.extend(articles)
                
                logger.info(f"Query '{query}': fetched {len(articles)} articles")
            else:
                logger.warning(f"No articles found for query: {query}")
        
        logger.info(f"Total articles crawled: {len(all_articles)}")
        return all_articles

    def save_articles_to_json(self, articles: List[PubMedArticle], output_dir: str = "data/pubmed"):
        """Save articles to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for article in articles:
            filename = f"{article.pmcid}.json" if article.pmcid else f"{article.pmid}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(article.dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_dir}")

# Legacy compatibility function
def parsing(metadata, xml_string):
    """Legacy parsing function for backward compatibility"""
    parser = XMLParser()
    return parser.parse_pubmed_xml(xml_string, metadata) 