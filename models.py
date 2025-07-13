from pydantic import BaseModel
from typing import List, Optional

# Models for travel destination search
class DestinationSearchPayload(BaseModel):
    cityId: str  # Thay đổi từ citySlug thành cityId
    purpose: str
    limit: Optional[int] = 10

class DestinationResult(BaseModel):
    title: str
    slug: str
    tags: List[str]
    location: dict
    details: dict
    album: dict
    score: float

class DestinationSearchResponse(BaseModel):
    city: dict
    purpose: str
    generatedTags: List[str]
    destinations: List[DestinationResult]
    totalFound: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionPayload(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "deepseek-r1-distill-llama-70b"
    userId: str
    isUseKnowledge: Optional[bool] = False
    courseId: Optional[str] = None
    courseTitle: Optional[str] = None