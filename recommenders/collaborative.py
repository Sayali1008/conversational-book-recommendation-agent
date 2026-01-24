"""
Collaborative filtering based recommendations.
Combines user-item interactions with content-based embeddings for hybrid recommendations.
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.helpers import normalize_scores

from .data_models import RecommendationConfig, RecommendationContext

logger = logging.getLogger(__name__)


def get_collaborative_recommendations(
    context: RecommendationContext,
    config: RecommendationConfig,
    user_idx: int,
    k: int = 10,
    exclude_catalog_rows: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate hybrid CF + content recommendations for warm user.

    Args:
        seed_catalog_indices: Optional recent swipes to boost user profile (e.g., from session state)
    """
    mappings = context["index_mappings"]
    n_cf_books = len(context["book_factors"])

    # Step 1: Compute collaborative filtering scores
    user_vec = context["user_factors"][user_idx]
    cf_scores = context["book_factors"].dot(user_vec)
    cf_scores = np.asarray(cf_scores).ravel()

    logger.debug(f"[CF] CF scores: min={cf_scores.min():.6f}, max={cf_scores.max():.6f}, mean={cf_scores.mean():.6f}")

    # Mask rated items
    if config["filter_rated"]:
        user_train_row = context["train_matrix"][user_idx].toarray().flatten()
        rated_cf_indices = np.where(user_train_row > 0)[0]
        logger.debug(f"[CF] User has {len(rated_cf_indices)} rated books")
        cf_scores[rated_cf_indices] = -np.inf

    # Step 2: Select candidate pool
    if config["candidate_pool_size"] is not None:
        candidate_cf_book_indices = np.argsort(cf_scores)[::-1][: config["candidate_pool_size"]]
    else:
        candidate_cf_book_indices = np.arange(n_cf_books)

    candidate_cf_book_indices = candidate_cf_book_indices[np.isfinite(cf_scores[candidate_cf_book_indices])]
    logger.debug(f"[CF] {len(candidate_cf_book_indices)} candidates selected")

    if len(candidate_cf_book_indices) == 0:
        logger.debug(f"[CF] No candidates available")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

    # Step 3: Compute embedding scores on candidates
    candidate_catalog_indices = np.array(
        [mappings["cf_idx_to_catalog_id"][cf_idx] for cf_idx in candidate_cf_book_indices]
    )
    candidate_embeddings = context["catalog_embeddings"][candidate_catalog_indices]

    logger.debug("[CF] Building user embedding profile for warm user")
    user_profile = _create_user_profile_from_history(
        context, user_idx, exclude_catalog_rows=exclude_catalog_rows, recency_boost=config["recency_boost"]
    )

    embedding_scores = cosine_similarity(user_profile.reshape(1, -1), candidate_embeddings).flatten()

    # Step 4: Blend and rank
    cf_scores_candidate = cf_scores[candidate_cf_book_indices]

    cf_scores_norm = normalize_scores(cf_scores_candidate, config["norm"], config["norm_metadata"])
    embedding_scores_norm = normalize_scores(embedding_scores, config["norm"], config["norm_metadata"])

    hybrid_scores = config["lambda_weight"] * cf_scores_norm + (1 - config["lambda_weight"]) * embedding_scores_norm

    sorted_idx = np.argsort(hybrid_scores)[::-1][:k]

    return (
        candidate_catalog_indices[sorted_idx],
        hybrid_scores[sorted_idx],
        np.array(["hybrid"] * len(sorted_idx), dtype=object),
    )


def _create_user_profile_from_history(context, user_idx, exclude_catalog_rows=None, recency_boost=2.0):
    """
    Create user's semantic profile from their rated books + recent session swipes.
    Combines historical training data with recent swipes (if provided), giving more
    weight to recent preferences to capture session-level intent.
    """
    mappings = context["index_mappings"]
    user_row = context["train_matrix"][user_idx].toarray().flatten()
    rated_cf_book_indexes = np.where(user_row > 0)[0]

    if len(rated_cf_book_indexes) == 0:
        return np.zeros(context["catalog_embeddings"].shape[1])

    # Convert CF indices to catalog indices
    rated_catalog_rows = np.array([mappings["cf_idx_to_catalog_id"][cf_idx] for cf_idx in rated_cf_book_indexes])
    rated_embeddings = context["catalog_embeddings"][rated_catalog_rows]

    # Weighted average by confidence
    confidences = user_row[rated_cf_book_indexes]

    # If recent swipes provided, combine historical + recent preferences
    if exclude_catalog_rows is not None and len(exclude_catalog_rows) > 0:
        logger.debug(f"[CF] Combining {len(rated_cf_book_indexes)} historical + {len(exclude_catalog_rows)} recent items")

        # Get recent swipe embeddings
        recent_embeddings = context["catalog_embeddings"][exclude_catalog_rows]

        # Combine embeddings and weights
        all_embeddings = np.vstack([rated_embeddings, recent_embeddings])

        # Give recent swipes higher weight (recency_boost)
        historical_weights = confidences  # Original training confidences
        recent_weights = np.full(len(exclude_catalog_rows), recency_boost)
        all_weights = np.concatenate([historical_weights, recent_weights])

        # Weighted average across all items
        user_profile = np.average(all_embeddings, axis=0, weights=all_weights)

        logger.debug(
            f"[CF] User profile: historical weight={historical_weights.sum():.2f}, recent weight={recent_weights.sum():.2f}"
        )
    else:
        # No recent swipes; use historical data only
        user_profile = np.average(rated_embeddings, axis=0, weights=confidences)

    return user_profile
