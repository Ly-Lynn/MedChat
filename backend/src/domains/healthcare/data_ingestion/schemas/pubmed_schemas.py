from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.shared.schemas.base_schemas import BaseEntity

class CrawlerConfig(BaseModel):
    """Configuration for PubMed crawler"""
    max_retries: int = 3
    backoff_factor: float = 1.0
    connect_timeout: int = 5
    read_timeout: int = 15
    default_limit: int = 100
    user_agent: str = "MedChat-Crawler/1.0 (Healthcare Research Platform)"

class ArticleSection(BaseModel):
    """Represents a section of an article"""
    heading: str = ""
    text: str = ""

class PubMedArticle(BaseEntity):
    """Schema for PubMed article data"""
    # Identifiers
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    
    # Basic metadata
    title: str
    abstract: str = ""
    journal: str = ""
    pub_year: Optional[str] = None
    journal_volume: Optional[str] = None
    page_info: Optional[str] = None
    
    # Content
    sections: List[ArticleSection] = Field(default_factory=list)
    
    # Authors and keywords
    authors: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Medical specific metadata
    medical_subject_headings: List[str] = Field(default_factory=list, alias="mesh_terms")
    publication_types: List[str] = Field(default_factory=list)
    
    # Data ingestion metadata
    source: str = "pubmed"
    crawl_date: datetime = Field(default_factory=datetime.utcnow)
    processing_status: str = "raw"  # raw, processed, embedded, indexed
    
    # Quality metrics
    text_length: Optional[int] = None
    section_count: Optional[int] = None
    citation_count: Optional[int] = None
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('sections', pre=True)
    def convert_sections(cls, v):
        if isinstance(v, list):
            result = []
            for section in v:
                if isinstance(section, dict):
                    result.append(ArticleSection(**section))
                else:
                    result.append(section)
            return result
        return v
    
    def get_full_text(self) -> str:
        """Get the full text content of the article"""
        parts = []
        
        if self.title:
            parts.append(self.title)
        
        if self.abstract:
            parts.append(self.abstract)
            
        for section in self.sections:
            if section.text:
                parts.append(section.text)
        
        return "\n\n".join(parts)
    
    def get_content_dict(self) -> Dict[str, str]:
        """Get content as dictionary for compatibility with existing models"""
        content = {}
        
        if self.abstract:
            content["abstract"] = self.abstract
            
        for i, section in enumerate(self.sections):
            if section.heading:
                content[section.heading.lower()] = section.text
            else:
                content[f"section_{i+1}"] = section.text
        
        return content

class CrawlJob(BaseEntity):
    """Schema for crawl job tracking"""
    queries: List[str]
    status: str = "pending"  # pending, running, completed, failed
    total_articles_requested: int
    total_articles_found: int = 0
    total_articles_processed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    config: CrawlerConfig = Field(default_factory=CrawlerConfig)

class CrawlResult(BaseModel):
    """Result of a crawl operation"""
    job_id: str
    articles: List[PubMedArticle]
    total_processed: int
    errors: List[str] = Field(default_factory=list)
    processing_time: Optional[float] = None 