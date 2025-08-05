from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DestinationDetails(BaseModel):
    description: Optional[str] = None
    highlight: Optional[str] = None
    services: Optional[str] = None
    cultureType: Optional[str] = None
    activities: Optional[str] = None
    fee: Optional[str] = None
    usefulInfo: Optional[str] = None
    tags: Optional[str] = None  
    openHour: Optional[str] = None
    contactInfo: Optional[str] = None

class IngestPayload(BaseModel):
    destinationId: str
    cityId: str
    info: str  
    slug: str
    name: str

class UpdatePayload(BaseModel):
    destinationId: str
    cityId: str
    info: str  
    slug: str
    name: str


class QuestionPayload(BaseModel):
    cityId: str
    query: str

class DeletePayload(BaseModel):
    destiationId: str
    cityId: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionPayload(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "llama-3.3-70b-versatile"  # Thay đổi model mặc định
    cityId: Optional[str] = None
    isUseKnowledge: Optional[bool] = True
