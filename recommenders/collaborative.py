"""
Collaborative filtering based recommendations.
Combines user-item interactions with content-based embeddings for hybrid recommendations.
"""

import numpy as np

from common.constants import *
from common.utils import *

logger = setup_logging(__name__, PATHS["app_log_file"])


def get_collaborative_scorer(context, user_cf, candidate_pool_size, swiped_books=None):
    """Generates normalized CF scores for all items, excluding rated ones."""

    # Compute raw ALS scores (dot product of user and book factors)
    user_vec = context["user_factors"][user_cf]
    cf_scores = context["book_factors"].dot(user_vec).ravel()

    mappings = context["index_mappings"]

    logger.debug(f"[CF] CF scores: min={cf_scores.min():.6f}, max={cf_scores.max():.6f}, mean={cf_scores.mean():.6f}")

    # Mask rated items with -infinity (to be filtered out later)
    if EVALUATION["filter_rated"]:
        user_train_row = context["train_matrix"][user_cf]
        rated_cf_indices = user_train_row.indices
        cf_scores[rated_cf_indices] = -np.inf

    # Remove swiped books from CF scores
    if swiped_books:
        swiped_book_ids = [row["book_id"] for row in swiped_books]
        for b in swiped_book_ids:
            cf_id = mappings["book_id_to_cf"].get(b)
            if cf_id is not None:
                cf_scores[cf_id] = -np.inf

    # Convert CF indices to Catalog indices and store in a map
    cf_to_catalog_map = mappings["book_cf_to_catalog_id"]

    # We need a list of (catalog_id, score) pairs
    catalog_scores = {}
    for cf_idx, score in enumerate(cf_scores):
        if np.isfinite(score):
            catalog_id = cf_to_catalog_map[cf_idx]
            catalog_scores[catalog_id] = score

    if candidate_pool_size is not None:
        top_n_scores = dict(
            sorted(catalog_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_pool_size]
        )

    return top_n_scores  # Returns a dict {catalog_id: cf_score}


# region HELPERS
def create_user_profile_from_history(context, user_cf, liked_catalog_ids, disliked_catalog_ids):
    """Create user's semantic profile from their rated books + swipes."""
    mappings = context["index_mappings"]
    user_row = context["train_matrix"][user_cf].toarray().flatten()
    rated_cf_book_indices = np.where(user_row > 0)[0]

    like_boost = 1.0
    dislike_penalty = 0.5

    liked_embeddings = (
        context["catalog_embeddings"][list(liked_catalog_ids)]
        if liked_catalog_ids
        else np.empty((0, EMBEDDINGS["dim"]))
    )
    disliked_embeddings = (
        context["catalog_embeddings"][list(disliked_catalog_ids)]
        if disliked_catalog_ids
        else np.empty((0, EMBEDDINGS["dim"]))
    )

    if len(liked_embeddings) > 0:
        liked_profile = liked_embeddings.mean(axis=0)
    else:
        liked_profile = np.zeros(EMBEDDINGS["dim"])

    if len(disliked_embeddings) > 0:
        disliked_profile = disliked_embeddings.mean(axis=0)
    else:
        disliked_profile = np.zeros(EMBEDDINGS["dim"])

    # Cold/new users: use seeds (preferred_catalog_ids) to avoid zero-vector profile
    if len(rated_cf_book_indices) == 0:
        user_profile = (liked_profile * like_boost) - (dislike_penalty * disliked_profile)
        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)  # normalize

        logger.debug(
            f"[CF] Cold user with {len(liked_catalog_ids)} liked books and {len(disliked_catalog_ids)} disliked books."
        )
        return user_profile

    rated_catalog_embeddings = []
    confidences = []
    if len(rated_cf_book_indices) > 0:
        confidences = user_row[rated_cf_book_indices]
        rated_catalog_ids = np.array([mappings["book_cf_to_catalog_id"][cf_idx] for cf_idx in rated_cf_book_indices])
        rated_catalog_embeddings = context["catalog_embeddings"][rated_catalog_ids]

    logger.debug(
        f"[CF] Combining {len(rated_cf_book_indices)} historical + {len(liked_catalog_ids)} liked books + {len(disliked_catalog_ids)} - disliked books"
    )

    # Positive components (historical + likes)
    liked_weights = np.full(len(liked_catalog_ids) if liked_catalog_ids else 0, like_boost)
    positive_embeddings = np.vstack([rated_catalog_embeddings, liked_embeddings])
    positive_weights = np.concatenate([confidences, liked_weights])
    positive_profile = np.average(positive_embeddings, axis=0, weights=positive_weights)

    # Negative component (dislikes)
    disliked_profile = disliked_embeddings.mean(axis=0) if len(disliked_embeddings) > 0 else np.zeros(EMBEDDINGS["dim"])

    user_profile = positive_profile - (disliked_profile * dislike_penalty)
    user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)

    return user_profile


# endregion
