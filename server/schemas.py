from typing import Optional
from pydantic import BaseModel, Field

class BookRecommendation(BaseModel):
    book_id: int
    catalog_idx: int
    title: str
    authors: list[str]
    score: float
    source: str

class RecommendResponse(BaseModel):
    recommendations: list[BookRecommendation]
    strategy: str
    used_seeds: list[int] = []

class SwipeRequest(BaseModel):
    user_id: str
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|superlike)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class SwipeResponse(BaseModel):
    status: str
    next_recommendations: Optional[list[BookRecommendation]] = None