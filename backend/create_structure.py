import os

def create_directories():
    """Tạo cấu trúc thư mục cho domain-driven design"""
    
    directories = [
        # Config
        "src/config",
        
        # Shared
        "src/shared/middleware",
        "src/shared/utils", 
        "src/shared/exceptions",
        "src/shared/schemas",
        
        # Infrastructure
        "src/infrastructure/database/clients",
        "src/infrastructure/database/repositories", 
        "src/infrastructure/external_services",
        "src/infrastructure/messaging",
        
        # Healthcare Domain (NO API folders inside domains)
        "src/domains/healthcare/chatbot/models",
        "src/domains/healthcare/chatbot/services",
        "src/domains/healthcare/chatbot/repositories",
        "src/domains/healthcare/chatbot/schemas",
        
        "src/domains/healthcare/embedding/models",
        "src/domains/healthcare/embedding/services",
        "src/domains/healthcare/embedding/processors",
        "src/domains/healthcare/embedding/repositories",
        "src/domains/healthcare/embedding/schemas",
        "src/domains/healthcare/embedding/config",
        "src/domains/healthcare/embedding/utils",
        
        "src/domains/healthcare/medical_records/models",
        "src/domains/healthcare/medical_records/services", 
        "src/domains/healthcare/medical_records/repositories",
        "src/domains/healthcare/medical_records/schemas",
        
        # Data Ingestion Subdomain (NO API folder)
        "src/domains/healthcare/data_ingestion/crawlers",
        "src/domains/healthcare/data_ingestion/parsers",
        "src/domains/healthcare/data_ingestion/services",
        "src/domains/healthcare/data_ingestion/processors", 
        "src/domains/healthcare/data_ingestion/schemas",
        "src/domains/healthcare/data_ingestion/config",
        
        # User Management Domain (NO API folders)
        "src/domains/user_management/authentication/models",
        "src/domains/user_management/authentication/services",
        "src/domains/user_management/authentication/repositories", 
        "src/domains/user_management/authentication/schemas",
        
        "src/domains/user_management/profile/models",
        "src/domains/user_management/profile/services",
        "src/domains/user_management/profile/repositories",
        "src/domains/user_management/profile/schemas",
        
        # Workers
        "src/workers",
        
        # Centralized API Layer (ONLY place for API endpoints)
        "src/api/v1/healthcare",
        "src/api/v1/users",
        "src/api/v1/admin",
        "src/api/middleware",
        "src/api/utils",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Tạo file __init__.py cho mỗi thư mục Python
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("")
    
    print("✅ Đã tạo xong cấu trúc thư mục mới (cleaned architecture)!")

if __name__ == "__main__":
    create_directories() 