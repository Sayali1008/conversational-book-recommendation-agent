import logging
import time

import numpy as np

from common.constants import EVALUATION, PATHS
from common.helpers import *
from common.utils import setup_logging

from .collaborative import create_user_profile_from_history, get_collaborative_scorer
from .content_based import get_content_based_scorer

logger = setup_logging(__name__, PATHS["app_log_file"], logging.DEBUG)


def get_recommendations(
    context,
    user_cf,
    candidate_pool_size,
    lambda_weight,
    is_warm_user=True,
    top_k=10,
    swiped_books=None,
    user_profile=None,
):
    exclusions = _build_exclusions(context, user_cf, swiped_books)

    if not is_warm_user:
        cold_indices, cold_scores = get_content_based_scorer(
            context=context, exclude_catalog_rows=exclusions, user_profile=user_profile
        )
        return cold_indices[:top_k], cold_scores[:top_k]

    mappings = context["index_mappings"]
    liked_catalog_ids = (
        {mappings["book_id_to_catalog_id"][b["book_id"]] for b in swiped_books if b["action"] == "like"}
        if swiped_books
        else set()
    )
    disliked_catalog_ids = (
        {mappings["book_id_to_catalog_id"][b["book_id"]] for b in swiped_books if b["action"] == "dislike"}
        if swiped_books
        else set()
    )

    # Get all CF scores as {catalog_id: raw_cf_score} dictionary
    cf_score_map = get_collaborative_scorer(context, user_cf, candidate_pool_size)

    # Early return if no candidates found in CF scoring
    if not cf_score_map or len(cf_score_map) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    user_profile = create_user_profile_from_history(context, user_cf, liked_catalog_ids, disliked_catalog_ids)

    # Get the candidate pool from the top CF scores
    candidate_catalog_rows = np.array(list(cf_score_map.keys()))

    # Get all CB scores for the candidates
    cb_items, cb_scores_norm = get_content_based_scorer(context, exclusions, candidate_catalog_rows, user_profile)

    # Blend and Rank: Align the raw CF scores to the items returned by the CB function
    aligned_cf_scores = np.array([cf_score_map[idx] for idx in cb_items])

    # Normalize the CF scores now that we have a consistent list of items
    cf_scores_norm = normalize_scores(aligned_cf_scores, EVALUATION["norm"], EVALUATION["norm_metadata"])

    # Calculate final hybrid score
    hybrid_scores = (lambda_weight * cf_scores_norm) + ((1 - lambda_weight) * cb_scores_norm)

    # Final Sort
    top_k_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    if len(top_k_indices) == 0:
        logger.info(f"No recommendations available")
        return np.array([], dtype=int), np.array([], dtype=float)

    return (cb_items[top_k_indices], hybrid_scores[top_k_indices])


# region HELPERS
def _build_exclusions(context, user_cf, swiped_books=None):
    """Combines training data and swiped books into a single set of catalog IDs to exclude."""
    mappings = context["index_mappings"]
    exclude_set = set()

    # 1. Add items from the training matrix (Historical interactions)
    if user_cf is not None:
        user_train_row = context["train_matrix"][user_cf]
        for cf_idx in user_train_row.indices:
            catalog_idx = mappings["book_cf_to_catalog_id"].get(cf_idx)
            if catalog_idx is not None:
                exclude_set.add(catalog_idx)

    # 2. Add items from the current session (Swiped books)
    if swiped_books:
        for book in swiped_books:
            catalog_idx = mappings["book_id_to_catalog_id"].get(book["book_id"])
            if catalog_idx is not None:
                exclude_set.add(catalog_idx)

    return exclude_set


# endregion
