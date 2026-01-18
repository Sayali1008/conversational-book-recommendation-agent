"""
Recommendation orchestrator that coordinates collaborative and content-based recommendations.
Implements Option 2 strategy: warm users get hybrid, cold items as fallback; cold users get content-based.
"""

from typing import Optional, Tuple

import numpy as np

from common.constants import *
from common.utils import *

from .collaborative import _create_user_profile_from_history, get_collaborative_recommendations
from .content_based import _get_cold_catalog_indices, get_content_based_recommendations
from .data_models import EvalConfig, RecommendationContext

logger = setup_logging(__name__, APP_LOG_FILE)


def recommend(
    context: RecommendationContext,
    config: EvalConfig,
    user_idx: Optional[int],
    is_warm_user: bool,
    k: int = 10,
    seed_catalog_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate recommendations.
    - Warm users: Get hybrid (CF + content) recommendations, use cold as fallback
    - Cold users: Get content-based recommendations only
    """

    logger.info(f"is_warm_user={is_warm_user}, user_idx={user_idx}, k={k}")

    # Get cold catalog indices upfront (used by content-based recommender)
    cold_catalog_indices = _get_cold_catalog_indices(context)
    logger.info(f"Cold catalog items: {len(cold_catalog_indices)}")

    # Build user profile for content-based filtering
    if is_warm_user:
        logger.info("Building user embedding profile for warm user")
        user_profile = _create_user_profile_from_history(context, user_idx)
    elif seed_catalog_indices is not None:
        logger.info(f"Building user profile from {len(seed_catalog_indices)} seed items")
        user_profile = context["catalog_embeddings"][seed_catalog_indices].mean(axis=0)
    else:
        logger.info(f"Using catalog mean as user profile")
        user_profile = context["catalog_embeddings"].mean(axis=0)

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
        )
        logger.info(f"Warm recommender returned {len(warm_indices)} items")
        if len(warm_indices) > 0:
            logger.info(f"  → warm_scores: {warm_scores[:min(3, len(warm_scores))]}")

        # Determine exclusions
        exclude_indices = set(seed_catalog_indices.tolist() if seed_catalog_indices is not None else [])
        exclude_indices.update(warm_indices.tolist())

        # Add rated items to exclusions
        if config["filter_rated"]:
            user_row = context["train_matrix"][user_idx].toarray().flatten()
            rated_cf_indices = np.where(user_row > 0)[0]
            rated_catalog_indices = {
                context["index_mappings"]["cf_idx_to_catalog_id"][cf_idx] for cf_idx in rated_cf_indices
            }
            exclude_indices.update(rated_catalog_indices)

        logger.info(f"Excluding {len(exclude_indices)} items (rated + warm + seed)")

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

            cold_indices, cold_scores, cold_sources = get_content_based_recommendations(
                context=context,
                config=config,
                k=n_needed,
                user_profile=user_profile,
                exclude_indices=exclude_indices,
                candidate_catalog_indices=cold_catalog_indices,
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

        exclude_indices = set(seed_catalog_indices.tolist() if seed_catalog_indices is not None else [])

        final_indices, final_scores, final_sources = get_content_based_recommendations(
            context=context,
            config=config,
            k=k,
            seed_catalog_indices=seed_catalog_indices,
            # user_profile=user_profile,
            exclude_indices=exclude_indices,
            candidate_catalog_indices=cold_catalog_indices,
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
