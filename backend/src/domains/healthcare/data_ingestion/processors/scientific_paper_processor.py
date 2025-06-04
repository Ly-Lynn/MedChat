from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime
import re

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document as LangChainDocument

# Local imports
from .processor import Processor
from ..schemas.pubmed_schemas import PubMedArticle, ArticleSection
from ..parsers.xml_parser import XMLParser

logger = logging.getLogger(__name__)

class ScientificPaperProcessor(Processor):
    """
    Processor specialized for scientific papers using LangChain
    Handles chunking, embedding, and metadata extraction for medical/scientific documents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scientific paper processor
        
        Args:
            config: Configuration containing:
                - chunk_size: Size of text chunks (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 200)
                - embedding_model: Embedding model to use (default: 'sentence-transformers/all-MiniLM-L6-v2')
                - use_openai: Whether to use OpenAI embeddings (default: False)
                - preserve_sections: Whether to respect section boundaries (default: True)
                - extract_citations: Whether to extract citations (default: True)
                - min_chunk_size: Minimum chunk size (default: 100)
        """
        super().__init__(config)
        
        # Configuration
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.embedding_model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.use_openai = self.config.get('use_openai', False)
        self.preserve_sections = self.config.get('preserve_sections', True)
        self.extract_citations = self.config.get('extract_citations', True)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        
        # Initialize components
        self._init_text_splitter()
        self._init_embeddings()
        self.xml_parser = XMLParser()
        
        self.logger.info(f"Initialized ScientificPaperProcessor with config: {self.config}")
    
    def _init_text_splitter(self):
        """Initialize text splitter with scientific paper optimizations"""
        if self.preserve_sections:
            # Custom separators for scientific papers
            separators = [
                "\n\n\n",  # Major section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence breaks
                " ",       # Word breaks
                ""         # Character level
            ]
        else:
            separators = None
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False
        )
    
    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            if self.use_openai:
                self.embeddings = OpenAIEmbeddings()
                self.logger.info("Using OpenAI embeddings")
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},  # Can be changed to 'cuda' if GPU available
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger.info(f"Using HuggingFace embeddings: {self.embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {str(e)}")
            self.embeddings = None
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data
        
        Args:
            data: Input data (PubMedArticle or dict)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(data, PubMedArticle):
                return bool(data.title and (data.abstract or data.sections))
            elif isinstance(data, dict):
                return bool(data.get('title') and (data.get('abstract') or data.get('sections')))
            else:
                return False
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
    
    def process(self, data: Union[PubMedArticle, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Process scientific paper document
        
        Args:
            data: PubMedArticle or dict containing paper data
            **kwargs: Additional parameters
                - include_embeddings: Whether to generate embeddings (default: True)
                - extract_metadata: Whether to extract enhanced metadata (default: True)
                
        Returns:
            Processed document with chunks, embeddings, and metadata
        """
        include_embeddings = kwargs.get('include_embeddings', True)
        extract_metadata = kwargs.get('extract_metadata', True)
        
        try:
            # Pre-process
            processed_data = self.pre_process(data)
            
            # Convert to PubMedArticle if needed
            if isinstance(processed_data, dict):
                article = PubMedArticle(**processed_data)
            else:
                article = processed_data
            
            # Extract and structure content
            structured_content = self._structure_content(article)
            
            # Create chunks
            chunks = self._create_chunks(structured_content, article)
            
            # Generate embeddings if requested
            embeddings = None
            if include_embeddings and self.embeddings:
                embeddings = self._generate_embeddings(chunks)
            
            # Extract enhanced metadata
            metadata = {}
            if extract_metadata:
                metadata = self._extract_enhanced_metadata(article)
            
            # Compile results
            result = {
                "article_id": article.pmid or article.id,
                "title": article.title,
                "abstract": article.abstract,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "embeddings": embeddings,
                "metadata": metadata,
                "processing_info": {
                    "processor": self.__class__.__name__,
                    "processed_at": datetime.utcnow().isoformat(),
                    "config": self.config,
                    "total_text_length": sum(len(chunk["content"]) for chunk in chunks)
                }
            }
            
            # Post-process
            result = self.post_process(result)
            
            self.logger.info(f"Successfully processed article {article.pmid}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing article: {str(e)}")
            raise
    
    def _structure_content(self, article: PubMedArticle) -> List[Dict[str, str]]:
        """
        Structure article content into logical sections
        
        Args:
            article: PubMedArticle instance
            
        Returns:
            List of structured content sections
        """
        sections = []
        
        # Title section
        if article.title:
            sections.append({
                "type": "title",
                "heading": "Title",
                "content": article.title,
                "section_order": 0
            })
        
        # Abstract section
        if article.abstract:
            sections.append({
                "type": "abstract", 
                "heading": "Abstract",
                "content": article.abstract,
                "section_order": 1
            })
        
        # Main content sections
        for i, section in enumerate(article.sections):
            sections.append({
                "type": "body",
                "heading": section.heading or f"Section {i+1}",
                "content": section.text,
                "section_order": i + 2
            })
        
        return sections
    
    def _create_chunks(self, structured_content: List[Dict[str, str]], article: PubMedArticle) -> List[Dict[str, Any]]:
        """
        Create text chunks with metadata
        
        Args:
            structured_content: Structured content sections
            article: Original article
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        chunk_id = 0
        
        for section in structured_content:
            if not section["content"] or len(section["content"]) < self.min_chunk_size:
                continue
                
            # Split section content
            if self.preserve_sections:
                # Keep sections intact if small enough
                if len(section["content"]) <= self.chunk_size:
                    section_chunks = [section["content"]]
                else:
                    section_chunks = self._split_text_preserving_structure(section["content"])
            else:
                # Use standard text splitter
                documents = [LangChainDocument(page_content=section["content"])]
                split_docs = self.text_splitter.split_documents(documents)
                section_chunks = [doc.page_content for doc in split_docs]
            
            # Create chunk metadata
            for i, chunk_content in enumerate(section_chunks):
                if len(chunk_content.strip()) < self.min_chunk_size:
                    continue
                    
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "article_id": article.pmid or str(article.id),
                    "section_type": section["type"],
                    "section_heading": section["heading"],
                    "section_order": section["section_order"],
                    "chunk_order": i,
                    "chunk_size": len(chunk_content),
                    "article_title": article.title,
                    "journal": article.journal,
                    "pub_year": article.pub_year,
                    "authors": article.authors[:3] if article.authors else [],  # First 3 authors
                    "doi": article.doi,
                    "pmcid": article.pmcid,
                    "mesh_terms": article.medical_subject_headings[:5] if article.medical_subject_headings else [],
                    "created_at": datetime.utcnow().isoformat()
                }
                
                # Extract citations if enabled
                if self.extract_citations:
                    citations = self._extract_citations(chunk_content)
                    chunk_metadata["citations"] = citations
                
                chunks.append({
                    "content": chunk_content.strip(),
                    "metadata": chunk_metadata
                })
                
                chunk_id += 1
        
        return chunks
    
    def _split_text_preserving_structure(self, text: str) -> List[str]:
        """
        Split text while preserving scientific paper structure
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def _extract_enhanced_metadata(self, article: PubMedArticle) -> Dict[str, Any]:
        """
        Extract enhanced metadata from article
        
        Args:
            article: PubMedArticle instance
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = {
            "article_type": self._determine_article_type(article),
            "research_field": self._determine_research_field(article),
            "methodology": self._extract_methodology(article),
            "findings_keywords": self._extract_findings_keywords(article),
            "statistical_methods": self._extract_statistical_methods(article),
            "study_design": self._extract_study_design(article),
            "quality_score": self._calculate_quality_score(article)
        }
        
        return metadata
    
    def _determine_article_type(self, article: PubMedArticle) -> str:
        """Determine article type based on content"""
        title_lower = article.title.lower()
        abstract_lower = article.abstract.lower() if article.abstract else ""
        
        if any(keyword in title_lower for keyword in ["review", "systematic review", "meta-analysis"]):
            return "review"
        elif any(keyword in title_lower or keyword in abstract_lower for keyword in ["clinical trial", "randomized", "rct"]):
            return "clinical_trial"
        elif any(keyword in title_lower or keyword in abstract_lower for keyword in ["case report", "case study"]):
            return "case_report"
        else:
            return "research_article"
    
    def _determine_research_field(self, article: PubMedArticle) -> str:
        """Determine research field based on keywords and MeSH terms"""
        mesh_terms = [term.lower() for term in article.medical_subject_headings]
        keywords = [keyword.lower() for keyword in article.keywords]
        all_terms = mesh_terms + keywords
        
        field_keywords = {
            "cardiology": ["heart", "cardiac", "cardio", "myocardial", "coronary"],
            "oncology": ["cancer", "tumor", "malignancy", "chemotherapy", "oncology"],
            "neurology": ["brain", "neural", "neuron", "cognitive", "neurological"],
            "infectious_disease": ["infection", "bacteria", "virus", "antimicrobial", "pathogen"],
            "immunology": ["immune", "antibody", "vaccination", "immunization", "allergy"]
        }
        
        for field, keywords_list in field_keywords.items():
            if any(keyword in " ".join(all_terms) for keyword in keywords_list):
                return field
        
        return "general_medicine"
    
    def _extract_methodology(self, article: PubMedArticle) -> List[str]:
        """Extract methodology keywords from article"""
        full_text = article.get_full_text().lower()
        
        methodology_terms = [
            "randomized controlled trial", "cohort study", "case-control study",
            "cross-sectional study", "meta-analysis", "systematic review",
            "qualitative study", "quantitative study", "experimental study"
        ]
        
        found_methods = []
        for method in methodology_terms:
            if method in full_text:
                found_methods.append(method)
        
        return found_methods
    
    def _extract_findings_keywords(self, article: PubMedArticle) -> List[str]:
        """Extract key findings and results keywords"""
        abstract = article.abstract.lower() if article.abstract else ""
        
        findings_patterns = [
            r'significant\w*\s+\w+',
            r'effective\w*\s+\w+',
            r'improvement\s+in\s+\w+',
            r'reduction\s+in\s+\w+',
            r'increase\s+in\s+\w+'
        ]
        
        findings = []
        for pattern in findings_patterns:
            matches = re.findall(pattern, abstract)
            findings.extend(matches)
        
        return findings[:10]  # Limit to top 10 findings
    
    def _extract_statistical_methods(self, article: PubMedArticle) -> List[str]:
        """Extract statistical methods mentioned in the article"""
        full_text = article.get_full_text().lower()
        
        statistical_terms = [
            "t-test", "chi-square", "anova", "regression", "correlation",
            "p-value", "confidence interval", "odds ratio", "hazard ratio",
            "mann-whitney", "wilcoxon", "fisher's exact test"
        ]
        
        found_stats = []
        for term in statistical_terms:
            if term in full_text:
                found_stats.append(term)
        
        return found_stats
    
    def _extract_study_design(self, article: PubMedArticle) -> str:
        """Extract study design information"""
        full_text = article.get_full_text().lower()
        
        design_keywords = {
            "randomized_controlled_trial": ["randomized controlled trial", "rct", "randomized trial"],
            "cohort_study": ["cohort study", "longitudinal study", "prospective study"],
            "case_control": ["case-control study", "case control"],
            "cross_sectional": ["cross-sectional", "cross sectional", "survey study"],
            "systematic_review": ["systematic review", "meta-analysis"],
            "case_series": ["case series", "case report"]
        }
        
        for design, keywords in design_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                return design
        
        return "unknown"
    
    def _calculate_quality_score(self, article: PubMedArticle) -> float:
        """Calculate a simple quality score for the article"""
        score = 0.0
        
        # Check for abstract
        if article.abstract and len(article.abstract) > 100:
            score += 2.0
        
        # Check for full text sections
        if article.sections:
            score += min(len(article.sections) * 0.5, 3.0)
        
        # Check for authors
        if article.authors:
            score += min(len(article.authors) * 0.2, 1.0)
        
        # Check for identifiers
        if article.doi:
            score += 1.0
        if article.pmcid:
            score += 0.5
        
        # Check for MeSH terms
        if article.medical_subject_headings:
            score += min(len(article.medical_subject_headings) * 0.1, 1.0)
        
        # Check for journal info
        if article.journal:
            score += 0.5
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation patterns from text"""
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, Year) format
            r'\[\d+\]',              # [1] format
            r'\(\d+\)',              # (1) format
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    def pre_process(self, data: Any) -> Any:
        """Pre-processing for scientific papers"""
        if isinstance(data, dict):
            # Clean and normalize text fields
            for field in ['title', 'abstract']:
                if field in data and data[field]:
                    data[field] = self._clean_text(data[field])
            
            # Clean sections
            if 'sections' in data:
                for section in data['sections']:
                    if 'text' in section:
                        section['text'] = self._clean_text(section['text'])
        
        return data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep scientific notation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\+\=\<\>\/\%\°]', '', text)
        
        return text.strip()
    
    def post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-processing for results"""
        # Add processing statistics
        if "chunks" in result:
            chunk_sizes = [len(chunk["content"]) for chunk in result["chunks"]]
            result["processing_info"]["chunk_statistics"] = {
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "total_chunks": len(chunk_sizes)
            }
        
        return result 