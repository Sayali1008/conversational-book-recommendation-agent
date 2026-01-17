from typing import List, Optional
from pydantic import BaseModel, Field

class BookRecommendation(BaseModel):
    book_id: int
    title: str
    score: float
    source: str

class RecommendResponse(BaseModel):
    recommendations: List[BookRecommendation]
    strategy: str
    used_seeds: List[int] = []

class SwipeRequest(BaseModel):
    user_id: str
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|superlike)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class SwipeResponse(BaseModel):
    status: str
    next_recommendations: Optional[List[BookRecommendation]] = None