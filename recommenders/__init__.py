"""
Recommendation system components.
Pure functions with TypedDicts.
"""

from .data_models import (
    IndexMappings,
    RecommendationContext,
    ModelConfig,
    RecommendationConfig,
)
from .collaborative import get_collaborative_recommendations
from .content_based import get_content_based_recommendations
from .handler import recommend_books

__all__ = [
    "IndexMappings",
    "RecommendationContext",
    "ModelConfig",
    "RecommendationConfig",
    "get_collaborative_recommendations",
    "get_content_based_recommendations",
    "recommend_books",
]
