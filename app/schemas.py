from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[float] = None


class Source(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str = Field(..., min_length=1)
    user_name: Optional[str] = None
    tone_preference: Optional[Literal["professional", "casual", "playful"]] = None
    image_url: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    suggestions: List[str] = []
    sources: List[Source] = []
    memory_summary: Optional[str] = None


class RagIngestRequest(BaseModel):
	folder: str = Field(..., description="Absolute or workspace path to a folder containing PDFs")


class RagAskRequest(BaseModel):
	question: str = Field(..., min_length=3)
	top_k: int = Field(5, ge=1, le=20)