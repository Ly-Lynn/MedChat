from flask import Blueprint, request, jsonify
import asyncio
from typing import Dict, Any, List
from src.domains.healthcare.data_ingestion.services.data_ingestion_service import create_data_ingestion_service
from src.shared.schemas.base_schemas import BaseResponse

data_ingestion_bp = Blueprint('data_ingestion', __name__)

# Initialize service (imported from domain layer)
data_ingestion_service = create_data_ingestion_service()

@data_ingestion_bp.route('/crawl', methods=['POST'])
def crawl_articles():
    """
    Crawl articles from PubMed based on search queries
    
    Request body:
    {
        "queries": ["diabetes", "heart disease"],
        "articles_per_query": 5,
        "generate_embeddings": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                "success": False,
                "message": "Queries list is required"
            }), 400
        
        queries = data['queries']
        articles_per_query = data.get('articles_per_query', 10)
        generate_embeddings = data.get('generate_embeddings', True)
        
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({
                "success": False,
                "message": "Queries must be a non-empty list"
            }), 400
        
        # Run data ingestion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            data_ingestion_service.ingest_from_queries(
                queries, articles_per_query, generate_embeddings
            )
        )
        
        return jsonify({
            "success": True,
            "message": "Data ingestion completed",
            "data": {
                "job_id": result.job_id,
                "total_processed": result.total_processed,
                "processing_time": result.processing_time,
                "errors": result.errors
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error during data ingestion: {str(e)}"
        }), 500

@data_ingestion_bp.route('/stats', methods=['GET'])
def get_ingestion_stats():
    """Get statistics about data ingestion"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats = loop.run_until_complete(data_ingestion_service.get_ingestion_stats())
        
        return jsonify({
            "success": True,
            "message": "Ingestion statistics retrieved",
            "data": stats
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error getting ingestion stats: {str(e)}"
        }), 500

@data_ingestion_bp.route('/test/search', methods=['POST'])
def test_search():
    """
    Test search functionality without full ingestion
    
    Request body:
    {
        "query": "diabetes",
        "limit": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "message": "Query is required"
            }), 400
        
        query = data['query']
        limit = data.get('limit', 5)
        
        # Test search metadata only
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metadata_list = loop.run_until_complete(
            asyncio.get_event_loop().run_in_executor(
                None,
                data_ingestion_service.crawler.search_metadata,
                query,
                limit
            )
        )
        
        return jsonify({
            "success": True,
            "message": "Search test completed",
            "data": {
                "query": query,
                "results_found": len(metadata_list),
                "sample_results": metadata_list[:3] if metadata_list else []
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error testing search: {str(e)}"
        }), 500 