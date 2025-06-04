from bs4 import BeautifulSoup
import logging
from typing import Dict, Any, List, Optional
from src.domains.healthcare.data_ingestion.schemas.pubmed_schemas import ArticleSection

logger = logging.getLogger(__name__)

class XMLParser:
    """Parser for PubMed/PMC XML content"""
    
    def parse_pubmed_xml(self, xml_string: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse PubMed XML content and extract structured information
        
        Args:
            xml_string: Raw XML content from PubMed/PMC
            metadata: Existing metadata from API response
            
        Returns:
            Structured article data
        """
        try:
            soup = BeautifulSoup(xml_string, "xml")
            result = {
                # Use metadata as base
                "title": metadata.get('title', ''),
                "abstract": metadata.get('abstractText', ''),
                "authors": metadata.get('authors', []),
                "journal": metadata.get('journal', ''),
                "doi": metadata.get('doi'),
                "pmid": metadata.get('pmid'),
                "pmcid": metadata.get('pmcid'),
                "pub_year": metadata.get('pubYear'),
                "journal_volume": metadata.get('journalVolume'),
                "page_info": metadata.get('pageInfo'),
                "keywords": metadata.get('keywords', []),
                "sections": []
            }
            
            # Override with XML content if available and more complete
            self._extract_title(soup, result)
            self._extract_abstract(soup, result)
            self._extract_sections(soup, result)
            self._extract_authors(soup, result)
            self._extract_journal_info(soup, result)
            self._extract_identifiers(soup, result)
            self._extract_mesh_terms(soup, result)
            
            # Calculate metrics
            result['text_length'] = self._calculate_text_length(result)
            result['section_count'] = len(result['sections'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing XML: {str(e)}")
            # Return metadata as fallback
            return {
                "title": metadata.get('title', ''),
                "abstract": metadata.get('abstractText', ''),
                "authors": metadata.get('authors', []),
                "journal": metadata.get('journal', ''),
                "doi": metadata.get('doi'),
                "pmid": metadata.get('pmid'),
                "pmcid": metadata.get('pmcid'),
                "pub_year": metadata.get('pubYear'),
                "sections": []
            }
    
    def _extract_title(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract article title"""
        title_tag = soup.find("article-title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            if title:
                result["title"] = title
    
    def _extract_abstract(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract abstract content"""
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            abstract = abstract_tag.get_text(" ", strip=True)
            if abstract and len(abstract) > len(result.get("abstract", "")):
                result["abstract"] = abstract
    
    def _extract_sections(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract article sections (body content)"""
        sections = []
        
        # Extract from body sections
        for sec in soup.find_all("sec"):
            heading = sec.find("title")
            heading_text = heading.get_text(strip=True) if heading else ""
            
            # Get all paragraphs in this section
            paragraphs = []
            for p in sec.find_all("p"):
                paragraph_text = p.get_text(strip=True)
                if paragraph_text:
                    paragraphs.append(paragraph_text)
            
            if paragraphs:
                section_text = "\n\n".join(paragraphs)
                sections.append({
                    "heading": heading_text,
                    "text": section_text
                })
        
        result["sections"] = sections
    
    def _extract_authors(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract author information"""
        authors = []
        
        # Try different author extraction methods
        for contrib in soup.find_all("contrib", attrs={"contrib-type": "author"}):
            name_tag = contrib.find("name")
            if name_tag:
                given = name_tag.find("given-names")
                surname = name_tag.find("surname")
                
                given_text = given.get_text(strip=True) if given else ""
                surname_text = surname.get_text(strip=True) if surname else ""
                
                full_name = f"{given_text} {surname_text}".strip()
                if full_name:
                    authors.append(full_name)
        
        # Use extracted authors if we found more than in metadata
        if len(authors) > len(result.get("authors", [])):
            result["authors"] = authors
    
    def _extract_journal_info(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract journal information"""
        journal_tag = soup.find("journal-title")
        if journal_tag:
            journal = journal_tag.get_text(strip=True)
            if journal:
                result["journal"] = journal
        
        # Extract volume, issue, etc.
        volume_tag = soup.find("volume")
        if volume_tag:
            result["journal_volume"] = volume_tag.get_text(strip=True)
        
        # Extract page info
        fpage = soup.find("fpage")
        lpage = soup.find("lpage")
        if fpage and lpage:
            result["page_info"] = f"{fpage.get_text(strip=True)}-{lpage.get_text(strip=True)}"
        elif fpage:
            result["page_info"] = fpage.get_text(strip=True)
    
    def _extract_identifiers(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract article identifiers (DOI, PMID, PMCID)"""
        for article_id in soup.find_all("article-id"):
            id_type = article_id.get("pub-id-type")
            id_value = article_id.get_text(strip=True)
            
            if id_type == "doi" and id_value:
                result["doi"] = id_value
            elif id_type == "pmid" and id_value:
                result["pmid"] = id_value
            elif id_type == "pmc" and id_value:
                result["pmcid"] = id_value
        
        # Extract publication year
        year_tag = soup.find("pub-date")
        if year_tag:
            year = year_tag.find("year")
            if year:
                result["pub_year"] = year.get_text(strip=True)
    
    def _extract_mesh_terms(self, soup: BeautifulSoup, result: Dict[str, Any]):
        """Extract MeSH terms if available"""
        mesh_terms = []
        
        # Look for MeSH headings in various locations
        for kwd in soup.find_all("kwd"):
            term = kwd.get_text(strip=True)
            if term:
                mesh_terms.append(term)
        
        if mesh_terms:
            result["medical_subject_headings"] = mesh_terms
    
    def _calculate_text_length(self, result: Dict[str, Any]) -> int:
        """Calculate total text length"""
        total_length = 0
        
        if result.get("title"):
            total_length += len(result["title"])
        
        if result.get("abstract"):
            total_length += len(result["abstract"])
        
        for section in result.get("sections", []):
            if section.get("text"):
                total_length += len(section["text"])
        
        return total_length 