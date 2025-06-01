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
logger = logging.getLogger(__name__)

class PubMedCrawler:
    def __init__(self, retmode="xml"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.retmode = retmode
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
        # Configure headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _make_request(self, url: str, params=None) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            response = self.session.get(url, params=params, timeout=(5, 15))
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
    def parsing(self, metadata, xml_string):
        soup = BeautifulSoup(xml_string, "xml")

        result = {
            "title": metadata['title'],
            "abstract": "",
            "sections": [],
            "authors": metadata['authors'],
            "journal": metadata['journal'],
            "doi": metadata['doi'],
            "pmid": metadata['pmid'],
            "pmcid": metadata['pmcid'],
            "year": metadata['pubYear']
        }

        # Title
        title_tag = soup.find("article-title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)

        # Abstract
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            result["abstract"] = abstract_tag.get_text(" ", strip=True)

        # Sections (body text with headings)
        for sec in soup.find_all("sec"):
            heading = sec.find("title")
            heading_text = heading.get_text(strip=True) if heading else ""
            paragraphs = [p.get_text(strip=True) for p in sec.find_all("p")]
            if paragraphs:
                result["sections"].append({
                    "heading": heading_text,
                    "text": "\n\n".join(paragraphs)
                })

        # Authors
        for contrib in soup.find_all("contrib", attrs={"contrib-type": "author"}):
            name_tag = contrib.find("name")
            if name_tag:
                given = name_tag.find("given-names")
                surname = name_tag.find("surname")
                full_name = f"{given.get_text(strip=True) if given else ''} {surname.get_text(strip=True) if surname else ''}".strip()
                if full_name:
                    result["authors"].append(full_name)

        # Journal
        journal_tag = soup.find("journal-title")
        if journal_tag:
            result["journal"] = journal_tag.get_text(strip=True)

        # DOI, PMCID, PMID
        for article_id in soup.find_all("article-id"):
            id_type = article_id.get("pub-id-type")
            if id_type == "doi":
                result["doi"] = article_id.get_text(strip=True)
            elif id_type == "pmid":
                result["pmid"] = article_id.get_text(strip=True)
            elif id_type == "pmc":
                result["pmcid"] = article_id.get_text(strip=True)

        # Year
        year_tag = soup.find("pub-date")
        if year_tag:
            year = year_tag.find("year")
            if year:
                result["year"] = year.get_text(strip=True)

        return result

    def search_metadata(self, query: str) -> List[str]:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}+AND+OPEN_ACCESS:Y&format=json"
        try:
            response = self._make_request(url)
            root = json.loads(response.text)
            res = root['resultList']['result']
            final_res = []
            for item in res:
                try:
                    final_res.append({
                        'id': item['id'] if 'id' in item else None,
                        'doi': item['doi'] if 'doi' in item else None,
                        'pmid': item['pmid'] if 'pmid' in item else None,
                        'pmcid': item['pmcid'] if 'pmcid' in item else None,
                        'textId': [sub_item for sub_item in item['fullTextIdList']['fullTextId']] if 'fullTextIdList' in item else None,
                        'title': item['title'] if 'title' in item else None,
                        'authors': item['authorString'].split(', ') if 'authorString' in item else None,
                        'journal': item['journalTitle'] if 'journalTitle' in item else None,
                        'pubYear': item['pubYear'] if 'pubYear' in item else None,
                        'journalVolume': item['journalVolume'] if 'journalVolume' in item else None,
                        'pageInfo': item['pageInfo'] if 'pageInfo' in item else None,
                    })
                except Exception as e:
                    logger.error(f"Failed to parse item: {item} {str(e)}")
            return final_res
        except Exception as e:
            print("e", e)
            logger.error(f"\nFailed to search IDs for query '{query}': {str(e)}")
            return []

    def get_europepmc(self,pmc_objs:List[Dict]):
        res = []
        for pmc_obj in tqdm(pmc_objs):
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_obj['pmcid']}/fullTextXML"
            try:
                response = self._make_request(url)
                parsed =self.parsing(pmc_obj, response.text)
                self.save_to_json(parsed, f"{pmc_obj['pmcid']}.json")
                res.append(parsed)
            except Exception as e:
                logger.error(f"Failed to get EuropePMC for ID {pmc_obj['pmcid']}: {str(e)}")
        return len(res)
    
    def save_to_json(self, data, filename="pmc_article.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    queries = [
        "obesity",
        "heart disease",
        "breast cancer",
        "Alzheimer",
        "Parkinson",
        "COVID-19",
        "Leukemia",
        'Multiple sclerosis',
        'HIV/AIDS',
        'Fungal infections',
        'Lupus (SLE)'
    ]
    retmode = "xml"
    crawler = PubMedCrawler(retmode=retmode)

    for query in queries[:1]:
        print(f"Searching for '{query}'")
        pmc_ids = crawler.search_metadata(query)
        if pmc_ids:
            print(f"Found {len(pmc_ids)} PMC IDs for '{query}'")
            arr = crawler.get_europepmc(pmc_ids[:1])
            print(f"Saved {arr} articles to .json")
    print("✅ Done. Saved to .json")
