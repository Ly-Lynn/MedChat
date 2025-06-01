from flask import Blueprint, request, jsonify
import asyncio
from typing import Dict, Any
from src.domains.healthcare.embedding.services.online_embedding_service import OnlineEmbeddingService
from src.domains.healthcare.embedding.services.offline_embedding_service import OfflineEmbeddingService
from src.shared.schemas.base_schemas import BaseResponse

embedding_bp = Blueprint('embedding', __name__)

# Initialize services (imported from domain layer)
online_service = OnlineEmbeddingService()
offline_service = OfflineEmbeddingService()

@embedding_bp.route('/query', methods=['POST'])
def embed_query():
    """
    Embed a user query for real-time search
    
    Request body:
    {
        "query": "What are the symptoms of diabetes?",
        "model": "biobert"  # optional
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
        model = data.get('model', 'biobert')
        
        # Switch model if needed
        if model != online_service.model_name:
            online_service.switch_model(model)
        
        # Generate embedding
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embedding = loop.run_until_complete(online_service.embed_user_query(query))
        
        return jsonify({
            "success": True,
            "message": "Query embedded successfully",
            "data": {
                "query": query,
                "embedding": embedding,
                "model": model,
                "dimensions": len(embedding)
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error embedding query: {str(e)}"
        }), 500

@embedding_bp.route('/batch', methods=['POST'])
def embed_batch():
    """
    Embed multiple queries in batch
    
    Request body:
    {
        "queries": ["query1", "query2", ...],
        "model": "biobert"  # optional
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
        model = data.get('model', 'biobert')
        
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({
                "success": False,
                "message": "Queries must be a non-empty list"
            }), 400
        
        # Switch model if needed
        if model != online_service.model_name:
            online_service.switch_model(model)
        
        # Generate embeddings
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        embeddings = loop.run_until_complete(online_service.embed_multiple_queries(queries))
        
        return jsonify({
            "success": True,
            "message": "Queries embedded successfully",
            "data": {
                "queries": queries,
                "embeddings": embeddings,
                "model": model,
                "count": len(embeddings)
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error embedding queries: {str(e)}"
        }), 500

@embedding_bp.route('/document', methods=['POST'])
def embed_document():
    """
    Process and embed a single document
    
    Request body:
    {
        "title": "Document title",
        "content": {"abstract": "...", "body": "..."},
        "source": "manual",
        "tags": ["tag1", "tag2"],
        "model": "pubmedbert"  # optional
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "Document data is required"
            }), 400
        
        # Create document object
        from src.domains.healthcare.medical_records.models.document import MedicalDocument
        document = MedicalDocument(**data)
        
        # Process document
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processed_doc = loop.run_until_complete(offline_service.process_document(document))
        
        return jsonify({
            "success": True,
            "message": "Document processed successfully",
            "data": {
                "document_id": str(processed_doc.id) if processed_doc.id else "generated",
                "title": processed_doc.title,
                "embedding_dimensions": len(processed_doc.embeddings) if processed_doc.embeddings else 0,
                "metadata": processed_doc.metadata
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing document: {str(e)}"
        }), 500

@embedding_bp.route('/models', methods=['GET'])
def list_models():
    """List available embedding models"""
    try:
        # Mock available models
        models = ["biobert", "pubmedbert", "clinicalbert"]
        model_info = {}
        
        for model_name in models:
            model_info[model_name] = {
                "model_path": f"path/to/{model_name}",
                "dimensions": 768,
                "max_sequence_length": 512
            }
        
        return jsonify({
            "success": True,
            "message": "Available models retrieved",
            "data": {
                "models": models,
                "model_info": model_info,
                "current_online_model": online_service.model_name,
                "current_offline_model": offline_service.model_name
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error listing models: {str(e)}"
        }), 500 