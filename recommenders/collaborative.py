"""
Collaborative filtering based recommendations.
Combines user-item interactions with content-based embeddings for hybrid recommendations.
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.helpers import normalize_scores

from .data_models import EvalConfig, RecommendationContext

logger = logging.getLogger(__name__)


def get_collaborative_recommendations(
    context: RecommendationContext,
    config: EvalConfig,
    user_idx: int,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate hybrid CF + content recommendations for warm user."""
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
    
    # CONTENT-BASED STARTS HERE

    # Step 2: Select candidate pool
    if config["candidate_pool_size"] is not None:
        candidate_cf_book_indices = np.argsort(cf_scores)[::-1][:config["candidate_pool_size"]]
    else:
        candidate_cf_book_indices = np.arange(n_cf_books)

    candidate_cf_book_indices = candidate_cf_book_indices[np.isfinite(cf_scores[candidate_cf_book_indices])]    
    logger.debug(f"[CF] {len(candidate_cf_book_indices)} candidates selected")
    
    if len(candidate_cf_book_indices) == 0:
        logger.debug(f"[CF] No candidates available")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)
    
    # Step 3: Compute embedding scores on candidates
    candidate_catalog_indices = np.array([
        mappings["cf_idx_to_catalog_id"][cf_idx] 
        for cf_idx in candidate_cf_book_indices
    ])
    candidate_embeddings = context["catalog_embeddings"][candidate_catalog_indices]
    
    logger.debug("[CF] Building user embedding profile for warm user")
    user_profile = _create_user_profile_from_history(context, user_idx)
    
    embedding_scores = cosine_similarity(
        user_profile.reshape(1, -1), 
        candidate_embeddings
    ).flatten()
    
    # Step 4: Blend and rank
    cf_scores_candidate = cf_scores[candidate_cf_book_indices]
    
    cf_scores_norm = normalize_scores(cf_scores_candidate, config["norm"], config["norm_metadata"])
    embedding_scores_norm = normalize_scores(embedding_scores, config["norm"], config["norm_metadata"])
    
    hybrid_scores = (
        config["lambda_weight"] * cf_scores_norm + 
        (1 - config["lambda_weight"]) * embedding_scores_norm
    )
    
    sorted_idx = np.argsort(hybrid_scores)[::-1][:k]
    
    return (
        candidate_catalog_indices[sorted_idx],
        hybrid_scores[sorted_idx],
        np.array(["hybrid"] * len(sorted_idx), dtype=object)
    )


def _create_user_profile_from_history(context: RecommendationContext, user_idx: int) -> np.ndarray:
    """
    Create user's semantic profile from their rated books.
    
    Returns:
        user_profile: (embedding_dim,) weighted average of rated book embeddings
    """
    mappings = context["index_mappings"]
    user_row = context["train_matrix"][user_idx].toarray().flatten()
    rated_cf_indices = np.where(user_row > 0)[0]
    
    if len(rated_cf_indices) == 0:
        # Fallback: return zero vector
        return np.zeros(context["catalog_embeddings"].shape[1])
    
    # Convert CF indices to catalog indices
    rated_catalog_indices = np.array([
        mappings["cf_idx_to_catalog_id"][cf_idx]
        for cf_idx in rated_cf_indices
    ])
    
    rated_embeddings = context["catalog_embeddings"][rated_catalog_indices]
    
    # Weighted average by confidence
    confidences = user_row[rated_cf_indices]
    user_profile = np.average(rated_embeddings, axis=0, weights=confidences)
    
    return user_profile

