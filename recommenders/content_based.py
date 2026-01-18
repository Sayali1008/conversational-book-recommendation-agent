"""
Content-based recommendations using semantic embeddings.
Used for cold-start scenarios and as fallback for warm users.
"""

from typing import Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import *
from common.helpers import normalize_scores
from common.utils import *

from .data_models import EvalConfig, RecommendationContext

logger = setup_logging(__name__, APP_LOG_FILE)


def get_content_based_recommendations(
    context: RecommendationContext,
    config: EvalConfig,
    k: int = 10,
    seed_catalog_indices: Optional[np.ndarray] = None,
    user_profile: Optional[np.ndarray] = None,
    exclude_indices: Optional[Set[int]] = None,
    candidate_catalog_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate content-based recommendations.

    Returns:
        catalog_indices: (k,) array of recommended catalog indices
        scores: (k,) array of recommendation scores
        sources: (k,) array ["embedding_only" for each result]
    """

    exclude_indices = exclude_indices or set()

    # Build or use provided user profile
    if user_profile is None:
        if seed_catalog_indices is not None:
            user_profile = context["catalog_embeddings"][seed_catalog_indices].mean(axis=0)
            logger.debug(f"[CB] Built profile from {len(seed_catalog_indices)} seed items")
        else:
            user_profile = context["catalog_embeddings"].mean(axis=0)
            logger.debug(f"[CB] Built profile from catalog mean")

    # Determine search space
    if candidate_catalog_indices is None:
        candidate_catalog_indices = _get_cold_catalog_indices(context)

    # Filter out exclusions
    # exclude_indices will contain seed item calalog, already rated books (for warm users), and recommendations (for warm users)
    candidates = [c for c in candidate_catalog_indices if c not in exclude_indices]

    if not candidates:
        logger.debug(f"[CB] No candidates available after filtering")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

    logger.debug(f"[CB] Searching {len(candidates)} items from {len(candidate_catalog_indices)} total")

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


# def _create_user_profile_from_history(
#     context: RecommendationContext,
#     user_idx: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     Create user embedding profile from interaction history or return catalog mean.

#     Args:
#         context: Immutable world state
#         user_idx: User index (if None, returns catalog mean)

#     Returns:
#         user_profile: (embedding_dim,) embedding vector
#     """
#     if user_idx is None:
#         return context["catalog_embeddings"].mean(axis=0)

#     mappings = context["index_mappings"]
#     user_row = context["train_matrix"][user_idx].toarray().flatten()
#     rated_cf_indices = np.where(user_row > 0)[0]

#     if len(rated_cf_indices) == 0:
#         # Cold user: return global average
#         logger.debug(f"[CB] No history for user, using catalog mean")
#         return context["catalog_embeddings"].mean(axis=0)

#     confidences = user_row[rated_cf_indices]

#     # Convert CF indices to catalog indices
#     rated_catalog_indices = np.array([mappings["cf_idx_to_catalog_id"][cf_idx] for cf_idx in rated_cf_indices])

#     rated_embeddings = context["catalog_embeddings"][rated_catalog_indices]

#     # Weighted average by confidence
#     user_profile = np.average(rated_embeddings, axis=0, weights=confidences)

#     return user_profile


def _get_cold_catalog_indices(context: RecommendationContext) -> np.ndarray:
    """Get indices of books NOT in the CF training set."""
    # These would be all the books for cold users
    n_catalog = context["catalog_embeddings"].shape[0]
    warm_catalog = set(context["index_mappings"]["cf_idx_to_catalog_id"].values())
    return np.array([i for i in range(n_catalog) if i not in warm_catalog], dtype=int)
