"""
Recommendation system components.
Pure functions with TypedDicts.
"""

from .data_models import (
    IndexMappings,
    RecommendationContext,
    ModelConfig,
    EvalConfig,
)
from .collaborative import get_collaborative_recommendations
from .content_based import get_content_based_recommendations
from .orchestrator import recommend

__all__ = [
    "IndexMappings",
    "RecommendationContext",
    "ModelConfig",
    "EvalConfig",
    "get_collaborative_recommendations",
    "get_content_based_recommendations",
    "recommend",
]
