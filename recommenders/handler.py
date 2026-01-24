"""
Recommendation orchestrator that coordinates collaborative and content-based recommendations.
Implements Option 2 strategy: warm users get hybrid, cold items as fallback; cold users get content-based.
"""

from typing import Optional, Tuple

import numpy as np

from common.constants import PATHS
from common.utils import setup_logging

from .collaborative import _create_user_profile_from_history, get_collaborative_recommendations
from .content_based import _get_cold_catalog_indices, get_content_based_recommendations

logger = setup_logging(__name__, PATHS["app_log_file"])


def recommend(context, config, user_idx, is_warm_user, k=10, exclude_catalog_rows=None):
    """
    Generate recommendations.
    - Warm users: Get hybrid (CF + content) recommendations, use cold as fallback
    - Cold users: Get content-based recommendations only
    """

    logger.info(f"is_warm_user={is_warm_user}, user_idx={user_idx}, k={k}")

    # ===================================================================
    # WARM USER: Get hybrid recommendations with cold fallback
    # ===================================================================
    if is_warm_user:
        logger.info(f"Going on a warm user path")

        # Get warm (hybrid) recommendations
        warm_indices, warm_scores, warm_sources = get_collaborative_recommendations(
            context=context,
            config=config,
            user_idx=user_idx,
            k=k,
            exclude_catalog_rows=exclude_catalog_rows,
        )
        logger.info(f"Warm recommender returned {len(warm_indices)} items")
        if len(warm_indices) > 0:
            logger.info(f"  → warm_scores: {warm_scores[:min(3, len(warm_scores))]}")

        exclude_rows = set(warm_indices.tolist())

        # Add session-level exclusions (e.g., recently swiped items)
        if exclude_catalog_rows is not None:
            exclude_rows.update(
                exclude_catalog_rows.tolist() if isinstance(exclude_catalog_rows, np.ndarray) else exclude_catalog_rows
            )

        # Add rated items to exclusions
        if config["filter_rated"]:
            user_row = context["train_matrix"][user_idx].toarray().flatten()
            rated_cf_indices = np.where(user_row > 0)[0]
            rated_catalog_indices = {
                context["index_mappings"]["cf_idx_to_catalog_id"][cf_idx] for cf_idx in rated_cf_indices
            }
            exclude_rows.update(rated_catalog_indices)
        
        logger.info(f"Excluding {len(exclude_rows)} items (rated + warm + session swipes)")

        # Check if we have enough warm recommendations
        if len(warm_indices) >= k:
            # Enough warm recommendations, return only those
            logger.info(f"Warm items ({len(warm_indices)}) >= k ({k}), returning ONLY warm")
            final_indices = warm_indices
            final_scores = warm_scores
            final_sources = warm_sources
        else:
            # Need fallback cold items
            n_needed = k - len(warm_indices)
            logger.info(f"Warm items ({len(warm_indices)}) < k ({k}), need {n_needed} cold items")

            # Get cold catalog indices upfront (used by content-based recommender)
            unrated_catalog_rows = _get_cold_catalog_indices(context)
            logger.info(f"Cold catalog items: {len(unrated_catalog_rows)}")

            cold_indices, cold_scores, cold_sources = get_content_based_recommendations(
                context=context,
                config=config,
                k=n_needed,
                exclude_catalog_rows=exclude_rows,
                candidate_catalog_rows=unrated_catalog_rows,
            )

            logger.info(f"Cold recommender returned {len(cold_indices)} fallback items")
            if len(cold_indices) > 0:
                logger.info(f"  → cold_scores: {cold_scores[:min(3, len(cold_scores))]}")

            # Concatenate warm + cold
            final_indices = np.concatenate([warm_indices, cold_indices])
            final_scores = np.concatenate([warm_scores, cold_scores])
            final_sources = np.concatenate([warm_sources, cold_sources])

    # ===================================================================
    # COLD USER: Get content-based recommendations only
    # ===================================================================
    else:
        logger.info(f"Going on a cold user path")

        exclude_rows = set()

        # Add session-level exclusions (e.g., recently swiped items)
        if exclude_catalog_rows is not None:
            exclude_rows.update(
                exclude_catalog_rows.tolist() if isinstance(exclude_catalog_rows, np.ndarray) else exclude_catalog_rows
            )

        final_indices, final_scores, final_sources = get_content_based_recommendations(
            context=context,
            config=config,
            k=k,
            exclude_catalog_rows=exclude_rows,
        )

        logger.info(f"Cold recommender returned {len(final_indices)} items")
        if len(final_indices) > 0:
            logger.info(f"  → scores: {final_scores[:min(3, len(final_scores))]}")

    # ===================================================================
    # Return results
    # ===================================================================
    if len(final_indices) == 0:
        logger.info(f"No recommendations available")
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=object)

    # Trim to k
    final_indices = final_indices[:k]
    final_scores = final_scores[:k]
    final_sources = final_sources[:k]

    logger.info(f"Final results: {len(final_indices)} items")
    logger.info(f"Sources: {list(final_sources)}")
    logger.info(f"Scores: {final_scores[:min(3, len(final_scores))]}")

    return final_indices, final_scores, final_sources
