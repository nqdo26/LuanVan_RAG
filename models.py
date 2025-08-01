from pydantic import BaseModel
from typing import List, Optional

class IngestPayload(BaseModel):
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
    model: Optional[str] = "llama-3.3-70b-versatile"
    cityId: Optional[str] = None
    isUseKnowledge: Optional[bool] = True
