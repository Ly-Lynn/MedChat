from flask import Flask
from flask_cors import CORS
import logging
import sys
import asyncio
from src.config.settings import settings

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)

    # CORS configuration
    origins = [
        "http://localhost:5174",
        "http://localhost:3000"
    ]
    
    CORS(app, 
         resources={r"/*": {"origins": origins}},
         supports_credentials=True,
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
         expose_headers=["Content-Type", "Authorization"])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    app.config.from_object(settings)
    # Initialize database connections
    try:
        # Import and initialize database clients
        from src .infrastructure.database.clients.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        # Store database manager in app context
        app.db_manager = db_manager
        app.mongo_client = db_manager.get_mongodb()
        app.milvus_client = db_manager.get_milvus()
        app.redis_client = db_manager.get_redis()
        
        print(f"✅ Successfully initialized database connections")
    except Exception as e:
        print(f"❌ Failed to initialize database: {str(e)}")
        # Don't raise for now, continue without DB
    
    # Register API blueprints
    try:
        # Healthcare API routes
        from src.api.v1.healthcare.embedding import embedding_bp
        from src.api.v1.healthcare.data_ingestion import data_ingestion_bp
        
        # Register available blueprints
        app.register_blueprint(embedding_bp, url_prefix='/api/v1/healthcare/embedding')
        app.register_blueprint(data_ingestion_bp, url_prefix='/api/v1/healthcare/data-ingestion')
        
        print(f"✅ Successfully registered API blueprints")
    except ImportError as e:
        print(f"⚠️ Some API blueprints not available yet: {str(e)}")
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "development"
        }
    
    # API info endpoint
    @app.route('/api/v1/info')
    def api_info():
        """API information endpoint"""
        return {
            "api_version": "v1",
            "available_endpoints": {
                "healthcare": {
                    "embedding": "/api/v1/healthcare/embedding",
                    "data_ingestion": "/api/v1/healthcare/data-ingestion"
                }
            },
            "documentation": "/docs"
        }

    return app 