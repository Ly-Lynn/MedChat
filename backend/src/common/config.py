import os
from dotenv import load_dotenv


load_dotenv()

class Config:
    MONGODB_URI = os.getenv('MONGODB_URI')
    DB_NAME = os.getenv('MONGO_DATABASE', 'AIC')
    DATA_DIR=os.getenv('DATA_DIR', 'data')
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.getenv('MILVUS_PORT', 19530)
    ELASTICSEARCH_URI=os.getenv('ELASTICSEARCH_URI', 'http://elasticsearch:9200')    
    # ZILLIZ_URI = os.getenv('ZILLIZ_URI')
    # ZILLIZ_API_KEY = os.getenv('ZILLIZ_API_KEY')
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = os.getenv('REDIS_PORT', 6379)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    DEBUG = True 
    
    DIMENSIONS = 768
    VECTOR_WEIGHT = 0.6
    FULL_TEXT_WEIGHT = 0.2
    UNIT_WEIGHT = 0.2

    