import os
from typing import Optional

class Settings:
    """Application settings"""
    
    def __init__(self):
        # Debug mode
        self.DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
        
        # Database settings
        self.MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.MONGO_DATABASE: str = os.getenv("MONGO_DATABASE", "medchat")
        self.DB_NAME: str = os.getenv("DB_NAME", "medchat")
        # Vector database
        self.MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
        
        # Cache
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Security
        self.SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
        
        # External APIs
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# Global settings instance
settings = Settings() 