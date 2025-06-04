"""
Demo script showing how to use the Document Processing Pipeline
for scientific papers in LangChain-based RAG system
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Local imports
from .document_processing_pipeline import DocumentProcessingPipeline
from .scientific_paper_processor import ScientificPaperProcessor
from ..schemas.pubmed_schemas import PubMedArticle, ArticleSection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_articles() -> List[PubMedArticle]:
    """Create sample scientific articles for demonstration"""
    
    sample_articles = [
        PubMedArticle(
            pmid="12345678",
            title="Effects of Machine Learning in Medical Diagnosis: A Systematic Review",
            abstract="Background: Machine learning (ML) has emerged as a powerful tool in medical diagnosis. "
                     "This systematic review examines the current applications and effectiveness of ML in healthcare. "
                     "Methods: We searched PubMed, Embase, and Cochrane databases for studies published between 2015-2023. "
                     "Results: 150 studies were included. ML algorithms showed 85-95% accuracy in diagnostic tasks. "
                     "Conclusion: ML significantly improves diagnostic accuracy across multiple medical specialties.",
            journal="Journal of Medical Informatics",
            pub_year="2023",
            authors=["Nguyen Van A", "Tran Thi B", "Le Van C"],
            keywords=["machine learning", "medical diagnosis", "artificial intelligence", "healthcare"],
            medical_subject_headings=["Machine Learning", "Diagnosis", "Medical Informatics", "Artificial Intelligence"],
            sections=[
                ArticleSection(
                    heading="Introduction",
                    text="Machine learning has revolutionized many fields, and healthcare is no exception. "
                         "The application of ML algorithms in medical diagnosis has shown promising results "
                         "in improving accuracy and reducing diagnostic errors. This study aims to provide "
                         "a comprehensive review of current ML applications in medical diagnosis."
                ),
                ArticleSection(
                    heading="Methods",
                    text="We conducted a systematic literature review following PRISMA guidelines. "
                         "Electronic databases including PubMed, Embase, and Cochrane Library were searched. "
                         "Inclusion criteria: (1) studies published between 2015-2023, (2) focus on ML in diagnosis, "
                         "(3) peer-reviewed articles. Statistical analysis was performed using R software."
                ),
                ArticleSection(
                    heading="Results",
                    text="A total of 150 studies met our inclusion criteria. The most common ML algorithms "
                         "were support vector machines (35%), neural networks (28%), and random forests (20%). "
                         "Diagnostic accuracy ranged from 85% to 95% across different medical specialties. "
                         "Radiology showed the highest accuracy (95%), followed by pathology (92%)."
                ),
                ArticleSection(
                    heading="Discussion",
                    text="Our findings demonstrate the significant potential of ML in improving medical diagnosis. "
                         "The high accuracy rates observed across multiple specialties suggest that ML could "
                         "become a standard tool in clinical practice. However, challenges remain including "
                         "data quality, interpretability, and regulatory approval."
                )
            ],
            doi="10.1234/jmi.2023.001",
            source="pubmed"
        ),
        
        PubMedArticle(
            pmid="87654321",
            title="COVID-19 Vaccine Effectiveness: Real-World Evidence from Vietnam",
            abstract="Objective: To assess the real-world effectiveness of COVID-19 vaccines in preventing "
                     "severe disease and hospitalization in Vietnam. Design: Retrospective cohort study. "
                     "Setting: National surveillance data from January 2021 to December 2022. "
                     "Participants: 50 million vaccinated individuals. Main outcome measures: Vaccine "
                     "effectiveness against hospitalization and death. Results: Vaccine effectiveness "
                     "was 89% against hospitalization and 94% against death.",
            journal="Vietnamese Journal of Public Health",
            pub_year="2023",
            authors=["Pham Minh D", "Vo Thi E", "Hoang Van F"],
            keywords=["COVID-19", "vaccine effectiveness", "public health", "Vietnam"],
            medical_subject_headings=["COVID-19", "Vaccines", "Public Health", "Epidemiology"],
            sections=[
                ArticleSection(
                    heading="Background",
                    text="The COVID-19 pandemic has significantly impacted global health systems. "
                         "Vaccination has been the primary strategy for controlling the pandemic. "
                         "Vietnam implemented a national vaccination program in March 2021. "
                         "This study evaluates the real-world effectiveness of COVID-19 vaccines "
                         "in the Vietnamese population."
                ),
                ArticleSection(
                    heading="Methods",
                    text="We conducted a retrospective cohort study using national surveillance data. "
                         "The study period was from January 2021 to December 2022. Participants "
                         "included all individuals aged 18 years and older who received at least "
                         "one dose of COVID-19 vaccine. Vaccine effectiveness was calculated "
                         "using the test-negative case-control design."
                ),
                ArticleSection(
                    heading="Results",
                    text="During the study period, 50 million individuals received at least one dose "
                         "of COVID-19 vaccine. Overall vaccine effectiveness was 89% (95% CI: 87-91%) "
                         "against hospitalization and 94% (95% CI: 92-96%) against death. "
                         "Effectiveness varied by vaccine type: mRNA vaccines showed 92% effectiveness, "
                         "while viral vector vaccines showed 85% effectiveness."
                )
            ],
            doi="10.5678/vjph.2023.002",
            source="pubmed"
        )
    ]
    
    return sample_articles

def demo_scientific_paper_processor():
    """Demonstrate basic usage of ScientificPaperProcessor"""
    logger.info("=== Demo: ScientificPaperProcessor ===")
    
    # Configure processor
    config = {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'use_openai': False,  # Set to True if you have OpenAI API key
        'preserve_sections': True,
        'extract_citations': True,
        'min_chunk_size': 100
    }
    
    # Initialize processor
    processor = ScientificPaperProcessor(config)
    
    # Create sample article
    sample_articles = create_sample_articles()
    article = sample_articles[0]
    
    logger.info(f"Processing article: {article.title[:50]}...")
    
    # Process article
    try:
        result = processor.process(article)
        
        logger.info(f"Processing successful!")
        logger.info(f"Article ID: {result['article_id']}")
        logger.info(f"Number of chunks: {result['chunk_count']}")
        logger.info(f"Total text length: {result['processing_info']['total_text_length']}")
        
        # Display chunk information
        for i, chunk in enumerate(result['chunks'][:3]):  # Show first 3 chunks
            logger.info(f"\nChunk {i+1}:")
            logger.info(f"  Section: {chunk['metadata']['section_heading']}")
            logger.info(f"  Size: {chunk['metadata']['chunk_size']} characters")
            logger.info(f"  Content preview: {chunk['content'][:100]}...")
        
        # Display metadata
        if result['metadata']:
            logger.info(f"\nExtracted metadata:")
            logger.info(f"  Article type: {result['metadata']['article_type']}")
            logger.info(f"  Research field: {result['metadata']['research_field']}")
            logger.info(f"  Quality score: {result['metadata']['quality_score']:.2f}")
            logger.info(f"  Methodology: {result['metadata']['methodology']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}")
        return None

def demo_document_processing_pipeline():
    """Demonstrate usage of DocumentProcessingPipeline"""
    logger.info("\n=== Demo: DocumentProcessingPipeline ===")
    
    # Configure pipeline
    pipeline_config = {
        'processor_config': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_openai': False,
            'preserve_sections': True,
            'extract_citations': True
        },
        'vectorstore_type': 'faiss',
        'batch_size': 5,
        'max_workers': 2,
        'enable_async': True,
        'quality_threshold': 3.0
    }
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(pipeline_config)
    
    # Create sample articles
    sample_articles = create_sample_articles()
    
    logger.info(f"Processing {len(sample_articles)} articles...")
    
    # Process documents
    try:
        results = pipeline.process_documents(sample_articles)
        
        if results['success']:
            logger.info("Pipeline processing successful!")
            logger.info(f"Statistics: {results['statistics']}")
            
            # Display processed documents info
            processed_docs = results['processed_documents']
            logger.info(f"Processed {len(processed_docs)} documents")
            
            for i, doc in enumerate(processed_docs):
                logger.info(f"\nDocument {i+1}:")
                logger.info(f"  Title: {doc['title'][:50]}...")
                logger.info(f"  Chunks: {doc['chunk_count']}")
                logger.info(f"  Quality score: {doc['metadata']['quality_score']:.2f}")
            
            # Test retriever if vector store was created
            if results['vectorstore']:
                logger.info("\nTesting retriever...")
                retriever = pipeline.create_retriever({'k': 3})
                
                if retriever:
                    # Test query
                    test_query = "machine learning medical diagnosis"
                    relevant_docs = retriever.get_relevant_documents(test_query)
                    
                    logger.info(f"Found {len(relevant_docs)} relevant documents for query: '{test_query}'")
                    for i, doc in enumerate(relevant_docs):
                        logger.info(f"  Result {i+1}: {doc.page_content[:100]}...")
                        logger.info(f"  Metadata: {doc.metadata['section_heading']} - {doc.metadata['article_title'][:30]}...")
            
            return results
        else:
            logger.error(f"Pipeline processing failed: {results.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error in pipeline processing: {str(e)}")
        return None

async def demo_async_processing():
    """Demonstrate asynchronous processing"""
    logger.info("\n=== Demo: Async Processing ===")
    
    # Configure pipeline for async processing
    pipeline_config = {
        'processor_config': {
            'chunk_size': 800,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'use_openai': False
        },
        'vectorstore_type': 'faiss',
        'batch_size': 2,
        'enable_async': True
    }
    
    pipeline = DocumentProcessingPipeline(pipeline_config)
    sample_articles = create_sample_articles()
    
    start_time = datetime.utcnow()
    
    # Process asynchronously
    results = await pipeline.process_documents_async(sample_articles)
    
    end_time = datetime.utcnow()
    processing_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Async processing completed in {processing_time:.2f} seconds")
    logger.info(f"Results: {results['statistics']}")
    
    return results

def demo_configuration_options():
    """Demonstrate different configuration options"""
    logger.info("\n=== Demo: Configuration Options ===")
    
    # Test different configurations
    configs = [
        {
            'name': 'Small Chunks',
            'config': {
                'processor_config': {
                    'chunk_size': 500,
                    'chunk_overlap': 100,
                    'preserve_sections': False
                }
            }
        },
        {
            'name': 'Large Chunks',
            'config': {
                'processor_config': {
                    'chunk_size': 1500,
                    'chunk_overlap': 300,
                    'preserve_sections': True
                }
            }
        },
        {
            'name': 'High Quality Threshold',
            'config': {
                'processor_config': {'chunk_size': 1000},
                'quality_threshold': 8.0
            }
        }
    ]
    
    sample_articles = create_sample_articles()
    
    for test_config in configs:
        logger.info(f"\nTesting configuration: {test_config['name']}")
        
        pipeline = DocumentProcessingPipeline(test_config['config'])
        results = pipeline.process_documents(sample_articles)
        
        if results['success']:
            stats = results['statistics']
            logger.info(f"  Successful: {stats['successful']}")
            logger.info(f"  Failed: {stats['failed']}")
            logger.info(f"  Skipped: {stats['skipped']}")
            
            if results['processed_documents']:
                avg_chunks = sum(doc['chunk_count'] for doc in results['processed_documents']) / len(results['processed_documents'])
                logger.info(f"  Average chunks per document: {avg_chunks:.1f}")

def main():
    """Run all demos"""
    logger.info("Starting Document Processing Pipeline Demos")
    logger.info("=" * 50)
    
    # Demo 1: Basic processor usage
    demo_scientific_paper_processor()
    
    # Demo 2: Full pipeline usage
    demo_document_processing_pipeline()
    
    # Demo 3: Async processing
    asyncio.run(demo_async_processing())
    
    # Demo 4: Different configurations
    demo_configuration_options()
    
    logger.info("\n" + "=" * 50)
    logger.info("All demos completed!")

if __name__ == "__main__":
    main() 