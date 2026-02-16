from typing import Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common.constants import EVALUATION, PATHS
from common.helpers import normalize_scores
from common.utils import setup_logging

from .data_models import RecommendationConfig, RecommendationContext

logger = setup_logging(__name__, PATHS["app_log_file"])


def get_content_based_scorer(
    context,
    exclude_catalog_rows: Optional[Set[int]] = None,
    candidate_catalog_rows: Optional[np.ndarray] = None,
    user_profile: Optional[np.ndarray] = None,
):
    logger.info(f"[CB] Not a warm user, using content-based scorer with user profile {user_profile.shape}")

    # Catalog rows to be excluded are either seed items, liked items, or both
    exclude_catalog_rows = exclude_catalog_rows or set()
    logger.info(f"[CB] exclusions length: {len(exclude_catalog_rows)}")

    if user_profile is None:
        if exclude_catalog_rows is not None and len(exclude_catalog_rows) > 0:
            user_profile = context["catalog_embeddings"][list(exclude_catalog_rows)].mean(axis=0)
            logger.info(f"[CB] Built profile from {len(exclude_catalog_rows)} seed items")
        else:
            user_profile = context["catalog_embeddings"].mean(axis=0)
            logger.info(f"[CB] Built profile from the entire catalog mean")

    if candidate_catalog_rows is None:
        candidate_catalog_rows = _get_cold_catalog_indices(context)

    # exclude_catalog_rows will contain seed item calalog, already rated books (for warm users), and CBommendations (for warm users)
    candidates = np.array([c for c in candidate_catalog_rows if c not in exclude_catalog_rows], dtype=int)

    if not candidates.size:
        logger.debug(f"[CB] No candidates available after filtering")
        return np.array([], dtype=int), np.array([], dtype=float)

    logger.info(
        f"[CB] Searching {len(candidates)} items from total {candidate_catalog_rows.shape} candidate catalog rows"
    )

    # Compute similarity scores
    candidate_embeddings = context["catalog_embeddings"][candidates]
    scores = cosine_similarity(user_profile.reshape(1, -1), candidate_embeddings).flatten()

    logger.info(f"[CB] Raw scores: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

    scores = normalize_scores(scores, EVALUATION["norm"], EVALUATION["norm_metadata"])
    top_k_indices = np.argsort(scores)[::-1]

    if EVALUATION["norm"] == "none" and len(top_k_indices) > 0:
        max_score = scores[top_k_indices[0]]
        if max_score < 0.3:
            logger.warning(f"[CB] Low quality recommendations: max score = {max_score:.3f} (< 0.3 threshold)")
        elif max_score < 0.5:
            logger.info(f"[CB] Moderate quality recommendations: max score = {max_score:.3f} (< 0.3 threshold)")

    return np.array(candidates)[top_k_indices], scores[top_k_indices]


def _get_cold_catalog_indices(context: RecommendationContext) -> np.ndarray:
    """Get indices of books NOT in the CF training set."""
    # These would be all the books for cold users
    n_catalog = context["catalog_embeddings"].shape[0]
    warm_catalog = set(context["index_mappings"]["book_cf_to_catalog_id"].values())
    return np.array([i for i in range(n_catalog) if i not in warm_catalog], dtype=int)
