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

class SwipeRequest(BaseModel):
    """
    Swipe payload from frontend.
    
    action: "like", "dislike", or "superlike"
    confidence: optional; will be normalized by API to 1.0 (like) or 0.0 (dislike)
    k: number of replacement recommendations to return (defaults to 10)
    """
    user_id: str
    book_id: int
    action: str = Field(..., pattern="^(like|dislike|superlike)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    k: int = Field(default=10, ge=1, le=100)

class SwipeResponse(BaseModel):
    """
    Response to swipe request.
    
    status: "ok" on success
    next_recommendations: Prefetched recommendations for seamless UX
    """
    status: str
    next_recommendations: Optional[list[BookRecommendation]] = None

class BookDetails(BaseModel):
    """
    Full book details for modal/expanded view.
    
    Retrieved on-demand when user clicks on a book card.
    """
    book_id: int
    title: str
    authors: list[str]
    description: str
    genres: list[str]
    infolink: Optional[str] = None