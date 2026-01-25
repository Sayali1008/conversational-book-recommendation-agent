"""
Content-based recommendations using semantic embeddings.
Used for cold-start scenarios and as fallback for warm users.
"""

from typing import Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import PATHS
from common.helpers import normalize_scores
from common.utils import setup_logging

from .data_models import RecommendationConfig, RecommendationContext

logger = setup_logging(__name__, PATHS["app_log_file"])


def get_content_based_recommendations(
    context: RecommendationContext,
    config: RecommendationConfig,
    k: int = 10,
    exclude_catalog_rows: Optional[Set[int]] = None,
    candidate_catalog_rows: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate content-based recommendations.

    Returns:
        catalog_rows: (k,) array of recommended catalog indices
        scores: (k,) array of recommendation scores
        sources: (k,) array ["embedding_only" for each result]
    """

    exclude_catalog_rows = exclude_catalog_rows or set()

    # Build or use provided user profile    
    if exclude_catalog_rows is not None:
        user_profile = context["catalog_embeddings"][list(exclude_catalog_rows)].mean(axis=0)
        logger.debug(f"[CB] Built profile from {len(exclude_catalog_rows)} seed items")
    else:
        user_profile = context["catalog_embeddings"].mean(axis=0)
        logger.debug(f"[CB] Built profile from catalog mean")

    # Determine search space
    if candidate_catalog_rows is None:
        candidate_catalog_rows = _get_cold_catalog_indices(context)

    # Filter out exclusions
    # exclude_catalog_rows will contain seed item calalog, already rated books (for warm users), and recommendations (for warm users)
    candidates = [c for c in candidate_catalog_rows if c not in exclude_catalog_rows]

    if not candidates:
        logger.debug(f"[CB] No candidates available after filtering")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

    logger.debug(f"[CB] Searching {len(candidates)} items from {len(candidate_catalog_rows)} total")

    # Compute similarity scores
    candidate_embeddings = context["catalog_embeddings"][candidates]
    scores = cosine_similarity(user_profile.reshape(1, -1), candidate_embeddings).flatten()

    logger.debug(f"[CB] Raw scores: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

    # Normalize
    scores = normalize_scores(scores, config["norm"], config["norm_metadata"])
    logger.debug(f"[CB] After {config['norm']} norm: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

    # Select top-k
    top_idx = np.argsort(scores)[::-1][:k]
    return (np.array(candidates)[top_idx], scores[top_idx], np.array(["embedding_only"] * len(top_idx), dtype=object))


def _get_cold_catalog_indices(context: RecommendationContext) -> np.ndarray:
    """Get indices of books NOT in the CF training set."""
    # These would be all the books for cold users
    n_catalog = context["catalog_embeddings"].shape[0]
    warm_catalog = set(context["index_mappings"]["book_cf_to_catalog_id"].values())
    return np.array([i for i in range(n_catalog) if i not in warm_catalog], dtype=int)
