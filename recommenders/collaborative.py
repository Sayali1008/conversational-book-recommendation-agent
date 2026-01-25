"""
Collaborative filtering based recommendations.
Combines user-item interactions with content-based embeddings for hybrid recommendations.
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import *
from common.helpers import normalize_scores
from common.utils import *

from .data_models import RecommendationConfig, RecommendationContext

logger = setup_logging(__name__, PATHS["app_log_file"], logging.DEBUG)


def get_collaborative_recommendations(
    context: RecommendationContext,
    config: RecommendationConfig,
    user_cf,
    k=10,
    seed_book_ids=None,
    swiped_books=None,
):
    """Generate hybrid CF + content recommendations for warm user."""

    mappings = context["index_mappings"]
    n_cf_books = len(context["book_factors"])

    # Compute collaborative filtering scores
    user_vec = context["user_factors"][user_cf]
    cf_scores = context["book_factors"].dot(user_vec)
    cf_scores = np.asarray(cf_scores).ravel()

    logger.debug(f"[CF] CF scores: min={cf_scores.min():.6f}, max={cf_scores.max():.6f}, mean={cf_scores.mean():.6f}")

    # Mask rated items
    if config["filter_rated"]:
        user_train_row = context["train_matrix"][user_cf].toarray().flatten()
        rated_cf_indices = np.where(user_train_row > 0)[0]
        logger.debug(f"[CF] User has {len(rated_cf_indices)} rated books")
        cf_scores[rated_cf_indices] = -np.inf

    # Select candidate pool
    if config["candidate_pool_size"] is not None:
        selected_book_cf_list = np.argsort(cf_scores)[::-1][: config["candidate_pool_size"]]
    else:
        selected_book_cf_list = np.arange(n_cf_books)

    # Remove invalid scores
    selected_book_cf_list = selected_book_cf_list[np.isfinite(cf_scores[selected_book_cf_list])]

    # Remove swiped books to be excluded from selected recommendations
    exclude_book_cf_set = set()
    if swiped_books:
        swiped_book_ids = [row["book_id"] for row in swiped_books]
        for b in swiped_book_ids:
            cf_id = mappings["book_id_to_cf"].get(b)
            if cf_id is not None:
                exclude_book_cf_set.add(cf_id)

    selected_book_cf_list = [cf_id for cf_id in selected_book_cf_list if cf_id not in exclude_book_cf_set]
    logger.debug(f"[CF] Removed {len(exclude_book_cf_set)} excluded items from candidate pool")
    logger.debug(f"[CF] {len(selected_book_cf_list)} candidates selected")

    if len(selected_book_cf_list) == 0:
        logger.debug(f"[CF] No candidates available")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

    # Compute embedding scores on candidates
    selected_catalog_ids = np.array([mappings["book_cf_to_catalog_id"][cf_idx] for cf_idx in selected_book_cf_list])
    selected_catalog_embeddings = context["catalog_embeddings"][selected_catalog_ids]

    logger.debug("[CF] Building user embedding profile for warm user")
    seed_catalog_ids = set()
    if seed_book_ids:
        for b in seed_book_ids:
            idx = mappings["book_id_to_catalog_id"].get(b)
            if idx is not None:
                seed_catalog_ids.add(idx)

    # Add liked books to seed_book_ids to be used for profile building
    liked_catalog_ids = set()
    if swiped_books:
        liked_book_ids = [row["book_id"] for row in swiped_books if row["action"] == "like"]
        for b in liked_book_ids:
            idx = mappings["book_id_to_catalog_id"].get(b)
            if idx is not None:
                liked_catalog_ids.add(idx)

    user_profile = _create_user_profile_from_history(
        context, user_cf, liked_catalog_ids=liked_catalog_ids, seed_catalog_ids=seed_catalog_ids
    )

    embedding_scores = cosine_similarity(user_profile.reshape(1, -1), selected_catalog_embeddings).flatten()

    # Step 4: Blend and rank
    cf_scores_candidate = cf_scores[selected_book_cf_list]

    cf_scores_norm = normalize_scores(cf_scores_candidate, config["norm"], config["norm_metadata"])
    embedding_scores_norm = normalize_scores(embedding_scores, config["norm"], config["norm_metadata"])

    hybrid_scores = config["lambda_weight"] * cf_scores_norm + (1 - config["lambda_weight"]) * embedding_scores_norm

    sorted_idx = np.argsort(hybrid_scores)[::-1][:k]

    return (
        selected_catalog_ids[sorted_idx],
        hybrid_scores[sorted_idx],
        np.array(["hybrid"] * len(sorted_idx), dtype=object),
    )


def _create_user_profile_from_history(
    context, user_idx, liked_catalog_ids, seed_catalog_ids, seed_boost=1.0, likeness_boost=1.0
):
    """Create user's semantic profile from their rated books + swipes."""
    mappings = context["index_mappings"]
    user_row = context["train_matrix"][user_idx].toarray().flatten()
    rated_cf_book_indexes = np.where(user_row > 0)[0]

    # Cold/new users: use seeds (preferred_catalog_ids) to avoid zero-vector profile
    if len(rated_cf_book_indexes) == 0:
        if liked_catalog_ids or seed_catalog_ids:
            liked_embeddings, seed_embeddings = [], []
            liked_weights, seed_weights = [], []
            if liked_catalog_ids and len(liked_catalog_ids) > 0:
                liked_embeddings = context["catalog_embeddings"][list(liked_catalog_ids)]
                liked_weights = np.full(len(liked_catalog_ids), likeness_boost)

            if seed_catalog_ids and len(seed_catalog_ids) > 0:
                seed_embeddings = context["catalog_embeddings"][list(seed_catalog_ids)]
                seed_weights = np.full(len(seed_catalog_ids), seed_boost)

            profile_embeddings = np.vstack([liked_embeddings, seed_embeddings])
            profile_weights = np.concatenate([liked_weights, seed_weights])
            user_profile = np.average(profile_embeddings, axis=0, weights=profile_weights)

            logger.debug(
                f"[CF] Cold user with {len(liked_catalog_ids)} liked books and {len(seed_catalog_ids)} seed books."
            )
            return user_profile

        logger.debug(f"[CF] Cold user with no seeds or likes, returning zero-vector profile")
        return np.zeros(context["catalog_embeddings"].shape[1])

    rated_catalog_embeddings = []
    confidences = []
    if len(rated_cf_book_indexes) > 0:
        # Weighted average by confidence
        confidences = user_row[rated_cf_book_indexes]

        # Convert CF indices to catalog indices
        rated_catalog_ids = np.array([mappings["book_cf_to_catalog_id"][cf_idx] for cf_idx in rated_cf_book_indexes])
        rated_catalog_embeddings = context["catalog_embeddings"][rated_catalog_ids]

    # Contains seed + swiped likes
    embedding_dim = context["catalog_embeddings"].shape[1]
    liked_embeddings = np.empty((0, embedding_dim))  # Shape (0, 384)
    if liked_catalog_ids and len(liked_catalog_ids) > 0:
        liked_embeddings = context["catalog_embeddings"][list(liked_catalog_ids)]

    # Seed books
    seed_embeddings = np.empty((0, embedding_dim))   # Shape (0, 384)
    if seed_catalog_ids and len(seed_catalog_ids) > 0:
        seed_embeddings = context["catalog_embeddings"][list(seed_catalog_ids)]

    logger.debug(
        f"[CF] Combining {len(rated_cf_book_indexes)} historical + {len(liked_catalog_ids) if liked_catalog_ids else 0} liked books + {len(seed_catalog_ids) if seed_catalog_ids else 0} + seed books"
    )

    # Combine embeddings and weights
    all_embeddings = np.vstack([rated_catalog_embeddings, liked_embeddings, seed_embeddings])

    historical_weights = confidences  # Original training confidences
    liked_weights = np.full(len(liked_catalog_ids) if liked_catalog_ids else 0, likeness_boost)
    seed_weights = np.full(len(seed_catalog_ids) if seed_catalog_ids else 0, seed_boost)
    all_weights = np.concatenate([historical_weights, liked_weights, seed_weights])

    # Weighted average across all items
    user_profile = np.average(all_embeddings, axis=0, weights=all_weights)

    logger.debug(
        f"[CF] User profile: historical weight={historical_weights.sum():.2f}, liked weight={liked_weights.sum():.2f}, seed weight={seed_weights.sum():.2f}"
    )

    return user_profile
