from typing import Optional, List, Dict, Any
from bson import ObjectId
from ..clients.mongodb_client import MongoDBClient
from ..models.user import User
from .base_repository import BaseRepository
import logging
from datetime import datetime

class UserRepository(BaseRepository[User]):
    def __init__(self, mongodb_client: MongoDBClient):
        self.client = mongodb_client
        self.collection = mongodb_client.get_collection("users")
        self.logger = logging.getLogger(__name__)
    
    async def create(self, data: Dict[str, Any]) -> User:
        """Create new user"""
        try:
            user_data = User(**data)
            result = await self.collection.insert_one(user_data.dict(by_alias=True))
            user_data.id = result.inserted_id
            self.logger.info(f"Created user: {user_data.id}")
            return user_data
        except Exception as e:
            self.logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_by_id(self, id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            result = await self.collection.find_one({"_id": ObjectId(id)})
            return User(**result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get user {id}: {e}")
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = await self.collection.find_one({"email": email})
            return User(**result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_by_filter(self, filter_dict: Dict[str, Any], 
                           limit: int = 100, skip: int = 0) -> List[User]:
        """Get users by filter"""
        try:
            cursor = self.collection.find(filter_dict).skip(skip).limit(limit)
            results = await cursor.to_list(length=limit)
            return [User(**result) for result in results]
        except Exception as e:
            self.logger.error(f"Failed to get users by filter: {e}")
            return []
    
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[User]:
        """Update user"""
        try:
            data['updated_at'] = datetime.utcnow()
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(id)},
                {"$set": data},
                return_document=True
            )
            return User(**result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to update user {id}: {e}")
            return None
    
    async def delete(self, id: str) -> bool:
        """Delete user (soft delete)"""
        try:
            result = await self.collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Failed to delete user {id}: {e}")
            return False
    
    async def count(self, filter_dict: Dict[str, Any] = None) -> int:
        """Count users"""
        try:
            return await self.collection.count_documents(filter_dict or {})
        except Exception as e:
            self.logger.error(f"Failed to count users: {e}")
            return 0