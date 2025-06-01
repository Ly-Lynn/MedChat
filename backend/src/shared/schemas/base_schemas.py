from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler

from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema()
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        return {"type": "string"}

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class BaseEntity(BaseModel):
    """Base schema for all domain entities"""
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()}
    )
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

class BaseRequest(BaseModel):
    """Base schema for request models"""
    pass

class BaseResponse(BaseModel):
    """Base schema for response models"""
    success: bool = True
    message: str = "Success"
    
class PaginatedResponse(BaseResponse):
    """Base schema for paginated responses"""
    total: int
    page: int 
    page_size: int
    total_pages: int 