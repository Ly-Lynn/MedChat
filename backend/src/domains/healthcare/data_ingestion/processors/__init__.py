"""
Document Processing Module

This module provides classes for processing scientific papers and medical documents
using LangChain for RAG (Retrieval-Augmented Generation) applications.

Classes:
    - Processor: Base abstract class for all processors
    - ScientificPaperProcessor: Specialized processor for scientific papers
    - DocumentProcessingPipeline: Complete pipeline for document processing

Example usage:
    ```python
    from backend.src.domains.healthcare.data_ingestion.processors import (
        ScientificPaperProcessor,
        DocumentProcessingPipeline
    )
    
    # Basic processor usage
    processor = ScientificPaperProcessor({
        'chunk_size': 1000,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
    })
    
    # Full pipeline usage
    pipeline = DocumentProcessingPipeline({
        'processor_config': {
            'chunk_size': 1000,
            'use_openai': False
        },
        'vectorstore_type': 'faiss'
    })
    ```
"""

from .processor import Processor
from .scientific_paper_processor import ScientificPaperProcessor
from .document_processing_pipeline import DocumentProcessingPipeline

__all__ = [
    'Processor',
    'ScientificPaperProcessor', 
    'DocumentProcessingPipeline'
]

# Version info
__version__ = '1.0.0'
__author__ = 'MedChat Development Team'
