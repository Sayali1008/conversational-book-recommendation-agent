from typing import Optional

from pydantic import BaseModel, Field


class BookRecommendation(BaseModel):
    book_id: int
    catalog_idx: int
    title: str
    authors: list[str]
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[BookRecommendation]
    strategy: str


class SwipeRequest(BaseModel):
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
    book_id: int
    title: str
    authors: list[str]
    description: str
    genres: list[str]
    infolink: Optional[str] = None


class SearchBooksResponse(BaseModel):
    books: list[BookDetails]


class CreateUserRequest(BaseModel):
    name: str
    user_id: str


class CreateUserResponse(BaseModel):
    user_id: str
    name: str
    created_at: str


class LoginRequest(BaseModel):
    user_id: str


class LoginResponse(BaseModel):
    user_id: str
    name: str
    first_login: bool = False

class Genre(BaseModel):
    genre_id: int
    name: str


class GenresResponse(BaseModel):
    genres: list[Genre]


class UserGenresRequest(BaseModel):
    user_id: str
    genre_ids: list[int]


class UserGenresResponse(BaseModel):
    user_id: str
    saved_genres: list[int]


class Author(BaseModel):
    author_id: int
    name: str


class AuthorsResponse(BaseModel):
    authors: list[Author]


class AddBookRequest(BaseModel):
    title: str
    authors: list[int]
    description: str
    genres: list[int]
    infolink: Optional[str] = None


class AddBookResponse(BaseModel):
    book_id: int
    title: str
    created_at: str


class CreateAuthorRequest(BaseModel):
    name: str


class CreateAuthorResponse(BaseModel):
    author_id: int
    name: str


class CreateGenreRequest(BaseModel):
    name: str


class CreateGenreResponse(BaseModel):
    genre_id: int
    name: str