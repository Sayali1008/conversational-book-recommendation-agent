import pickle
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from utils import setup_logging

# Set up logging to file
log_file = str(Path(__file__).parent / "app_logs" / "01172026.log")
logger = setup_logging("app", log_file, level=logging.DEBUG)


# region [HELPERS]
def _log_recommendations(logger, book_indices, scores, sources, catalog_df):
    """Pretty print recommendations"""
    logger.info(f"{'=' * REPEATS}")
    logger.info(f"{'Rank':<6} {'Title':<40} {'Score':<8} {'Source':<15}")
    logger.info(f"{'=' * REPEATS}")

    for rank, (idx, score, source) in enumerate(zip(book_indices, scores, sources), 1):
        book = catalog_df.iloc[idx]
        title = book["title"][:37] + "..." if len(book["title"]) > 40 else book["title"]
        logger.info(f"{rank:<6} {title:<40} {score:.4f}   {source:<15}")


# endregion


# region [NORMALIZATION]
def minmax_score_normalization(scores):
    """Normalize scores to [0, 1] range."""
    scores = np.array(scores)

    # Currently, using minmax normalization
    # Scale to [0, 1]
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)


def softmax_normalization(scores, temperature=0.7):
    """
    Convert scores to sharp probabilities that emphasize top-K differences.

    temperature: lower τ → sharper peaks (best for precision@K)
                higher τ → flatter distribution (best for recall)
    """
    scores = np.asarray(scores, dtype=np.float32)
    # Standardize to prevent numerical overflow
    s = (scores - scores.mean()) / (scores.std() + 1e-8)
    # Scale by temperature
    s = s / max(temperature, 1e-4)
    # Softmax
    e = np.exp(s - s.max())  # Subtract max for numerical stability
    return e / (e.sum() + 1e-8)


def zscore_normalization(scores):
    """
    Normalize via z-score + sigmoid squashing.
    More robust to outliers than min-max.
    """
    scores = np.asarray(scores, dtype=np.float32)
    mu, sigma = scores.mean(), scores.std()
    if sigma < 1e-8:  # All same value
        return np.ones_like(scores) * 0.5
    # Z-score
    z = (scores - mu) / sigma
    # Sigmoid squash to [0, 1]
    return 1.0 / (1.0 + np.exp(-z))


# endregion


def cf_idx_to_catalog_id(cf_idx, cf_idx_to_book):
    """
    Convert CF matrix column index to catalog row index.

    Args:
        cf_idx: Index in CF matrix (0 to n_cf_books-1)
        cf_idx_to_book: Mapping from CF index to book_id

    Returns:
        catalog_id: Row index in catalog (0 to n_catalog_books-1)
    """
    book_id = cf_idx_to_book[cf_idx]
    # book_id was assigned sequentially starting at 1, so catalog_id = book_id - 1
    # Eg. catalog_books_df.iloc[i] has book_id = i + 1
    catalog_id = book_id - 1
    return catalog_id


def create_user_embedding_profile(user_idx, train_matrix, catalog_embeddings, idx_to_book_id):
    """
    Create user's embedding profile from their interaction history.

    user_profile is the 'likeness profile' for the user created by blending the semantics of all books they've read.
    Each number in the resulting (n_dim,) vector represents how much the user
    values a specific semantic feature (like 'dark humor' or 'technical detail')
    based on the average of their highly-rated books.
    """
    # Get user's interaction row from CF matrix
    user_row = train_matrix[user_idx].toarray().flatten()  # (n_cf_books,)

    # Find books this user rated (CF indices)
    rated_cf_indices = np.where(user_row > 0)[0]
    confidences = user_row[rated_cf_indices]

    if len(rated_cf_indices) == 0:
        # User has no ratings (shouldn't happen in train, but handle it)
        return np.zeros(catalog_embeddings.shape[1])

    # Go back to the catalog book embeddings
    # Convert CF indices to catalog indices and get embeddings
    rated_catalog_indices = np.array([cf_idx_to_catalog_id(cf_idx, idx_to_book_id) for cf_idx in rated_cf_indices])
    rated_embeddings = catalog_embeddings[rated_catalog_indices]  # (n_rated, n_dim)

    # Weighted average by confidence (higher rated books have more influence)
    user_profile = np.average(rated_embeddings, axis=0, weights=confidences)

    return user_profile  # (n_dim,)


def hybrid_recommender(
    user_idx,
    user_factors,
    book_factors,
    train_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
    norm="minmax",
    norm_metadata=None,
    k=10,
    lambda_weight=0.5,
    candidate_pool_size=None,
    filter_rated=True,
):
    """
    Unified hybrid CF + embedding recommender (warm-start only).

    Supports both single-stage and two-stage ranking:
    - candidate_pool_size=None: Rank ALL CF-trained books (single-stage, slower)
    - candidate_pool_size=N: Rank top-N CF candidates + hybrid re-rank (two-stage, faster)

    Returns:
        top_k_catalog_indices: (k,) array of recommended catalog indices
        top_k_scores: (k,) array of recommendation scores
        top_k_sources: (k,) array of source labels ["hybrid"]
    """
    n_cf_books = len(book_factors)
    logger.info(f"[HYBRID] Starting for user_idx={user_idx}, n_cf_books={n_cf_books}")

    # ===================================================================
    # Stage 1: Compute CF scores and mask rated items
    # ===================================================================
    user_vec = user_factors[user_idx]  # (n_factors,)
    cf_scores = book_factors.dot(user_vec)  # (n_cf_books,)
    cf_scores = np.asarray(cf_scores).ravel()
    logger.info(f"[HYBRID] CF scores computed: min={cf_scores.min():.6f}, max={cf_scores.max():.6f}, mean={cf_scores.mean():.6f}")

    # Mask rated items so they don't appear in recommendations
    if filter_rated:
        user_train_row = train_matrix[user_idx].toarray().flatten()
        rated_cf_indices = np.where(user_train_row > 0)[0]
        logger.info(f"[HYBRID] User has {len(rated_cf_indices)} rated books")
        cf_scores[rated_cf_indices] = -np.inf

    # ===================================================================
    # Stage 2: Candidate selection (single-stage or two-stage)
    # ===================================================================
    if candidate_pool_size is None:
        # Single-stage: use all CF books as candidates
        candidate_cf_book_indices = np.arange(n_cf_books)
        candidate_cf_book_indices = candidate_cf_book_indices[np.isfinite(cf_scores[candidate_cf_book_indices])]
    else:
        # Two-stage: select top-N by CF score only
        candidate_cf_book_indices = np.argsort(cf_scores)[::-1][:candidate_pool_size]
        candidate_cf_book_indices = candidate_cf_book_indices[np.isfinite(cf_scores[candidate_cf_book_indices])]

    logger.info(f"[HYBRID] Stage 2: {len(candidate_cf_book_indices)} candidate CF indices selected")

    # Fallback if no candidates (e.g., user rated all books)
    if len(candidate_cf_book_indices) == 0:
        logger.info(f"[HYBRID] NO CANDIDATES AVAILABLE - returning empty results")
        # Return empty recommendations
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=object),
        )

    # ===================================================================
    # Stage 3: Compute embedding scores on candidates
    # ===================================================================
    user_profile = create_user_embedding_profile(user_idx, train_matrix, catalog_embeddings, cf_idx_to_book)
    # (n_dim,)

    candidate_catalog_indices = np.array([cf_idx_to_catalog_id_map[cf_idx] for cf_idx in candidate_cf_book_indices])
    candidate_embeddings = catalog_embeddings[candidate_catalog_indices]  # (n_candidates, n_dim)

    embedding_scores = cosine_similarity(user_profile.reshape(1, -1), candidate_embeddings).flatten()  # (n_candidates,)

    # ===================================================================
    # Stage 4: Compute hybrid scores and rank for one candidate
    # ===================================================================
    cf_scores_candidate = cf_scores[candidate_cf_book_indices]

    if norm == "softmax":
        cf_scores_norm = softmax_normalization(cf_scores_candidate, temperature=norm_metadata)
        embedding_scores_norm = softmax_normalization(embedding_scores)
    elif norm == "minmax":
        cf_scores_norm = minmax_score_normalization(cf_scores_candidate)
        embedding_scores_norm = minmax_score_normalization(embedding_scores)
    elif norm == "zscore":
        cf_scores_norm = zscore_normalization(cf_scores_candidate)
        embedding_scores_norm = zscore_normalization(embedding_scores)

    hybrid_scores = lambda_weight * cf_scores_norm + (1 - lambda_weight) * embedding_scores_norm

    # Sort by hybrid score and take top-k
    sorted_idx = np.argsort(hybrid_scores)[::-1][:k]
    top_k_catalog_indices = candidate_catalog_indices[sorted_idx]
    top_k_scores = hybrid_scores[sorted_idx]
    top_k_sources = np.array(["hybrid"] * len(top_k_catalog_indices), dtype=object)

    logger.info(f"[HYBRID] Final results: {len(top_k_catalog_indices)} recommendations with scores {top_k_scores[:3]}")
    return top_k_catalog_indices, top_k_scores, top_k_sources


def get_cold_catalog_indices(n_catalog, cf_idx_to_catalog_id_map):
    warm_catalog = set(cf_idx_to_catalog_id_map.values())
    return np.array([i for i in range(n_catalog) if i not in warm_catalog], dtype=int)


def embedding_only_recommender(
    user_profile, catalog_embeddings, candidate_catalog_indices, k=10, norm="minmax", exclude=None
):
    exclude = exclude or set()
    cands = [c for c in candidate_catalog_indices if c not in exclude]
    if not cands:
        return np.array([], dtype=int), np.array([], dtype=float)
    logger.info(f"[EMBEDDING_ONLY] candidates pool size: {len(cands)} (from {len(candidate_catalog_indices)})")

    cand_emb = catalog_embeddings[cands]
    scores = cosine_similarity(user_profile.reshape(1, -1), cand_emb).flatten()
    logger.info(f"[EMBEDDING_ONLY] raw scores: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

    if norm == "softmax":
        scores = softmax_normalization(scores)
    elif norm == "zscore":
        scores = zscore_normalization(scores)
    else:
        scores = minmax_score_normalization(scores)

    logger.info(f"[EMBEDDING_ONLY] after {norm} norm: min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")

    top_idx = np.argsort(scores)[::-1][:k]
    return np.array(cands)[top_idx], scores[top_idx]


def recommend_with_cold_start(
    user_idx,
    is_warm_user,
    user_factors,
    book_factors,
    train_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
    k=10,
    lambda_weight=0.5,
    candidate_pool_size=200,
    filter_rated=True,
    seed_catalog_indices=None,
    norm="minmax",
    norm_metadata=None,
):
    n_catalog = catalog_embeddings.shape[0]
    cold_catalog_indices = get_cold_catalog_indices(n_catalog, cf_idx_to_catalog_id_map)
    logger.info(f"[RECOMMEND] is_warm_user={is_warm_user}, user_idx={user_idx}, k={k}")
    logger.info(f"[RECOMMEND] Cold catalog items: {len(cold_catalog_indices)}")

    # Build user profile
    if is_warm_user:
        logger.info("building user embedding profile for warm user..")
        user_profile = create_user_embedding_profile(user_idx, train_matrix, catalog_embeddings, cf_idx_to_book)
    elif seed_catalog_indices:
        logger.info("building user embedding profile for cold user using seed catalog indices..")
        user_profile = catalog_embeddings[seed_catalog_indices].mean(axis=0)
    else:
        logger.info("building user embedding profile for cold user..")
        user_profile = catalog_embeddings.mean(axis=0)  # very weak prior

    # Warm (hybrid) part — only if the user is warm
    warm_indices = np.array([], dtype=int)
    warm_scores = np.array([], dtype=float)
    warm_sources = np.array([], dtype=object)
    rated_catalog_indices = set()

    if is_warm_user:
        warm_indices, warm_scores, warm_sources = hybrid_recommender(
            user_idx,
            user_factors,
            book_factors,
            train_matrix,
            catalog_embeddings,
            cf_idx_to_catalog_id_map,
            cf_idx_to_book,
            norm=norm,
            norm_metadata=norm_metadata,
            k=k,
            lambda_weight=lambda_weight,
            candidate_pool_size=candidate_pool_size,
            filter_rated=filter_rated,
        )
        logger.info(f"[RECOMMEND] warm_indices returned: {len(warm_indices)} items")
        if len(warm_indices) > 0:
            logger.info(f"  → warm_scores: {warm_scores[:min(3, len(warm_scores))]}")

        if filter_rated:
            user_row = train_matrix[user_idx].toarray().flatten()
            rated_cf_indices = np.where(user_row > 0)[0]
            rated_catalog_indices = {cf_idx_to_catalog_id_map[c] for c in rated_cf_indices}

    # Cold items: embedding-only for books without CF factors
    exclude = rated_catalog_indices | set(seed_catalog_indices or []) | set(warm_indices.tolist())
    logger.info(f"[RECOMMEND] Excluding {len(exclude)} items (rated={len(rated_catalog_indices)} + warm={len(warm_indices)} + seed={len(seed_catalog_indices or [])})")
    logger.info(f"[RECOMMEND] Cold pool will search from {len(cold_catalog_indices)} total cold books, minus {len(exclude & set(cold_catalog_indices))} exclusions = {len(cold_catalog_indices) - len(exclude & set(cold_catalog_indices))} available")

    cold_indices, cold_scores = embedding_only_recommender(
        user_profile,
        catalog_embeddings,
        cold_catalog_indices,
        k=k,
        norm=norm,
        exclude=exclude,
    )
    logger.info(f"[RECOMMEND] cold_indices returned: {len(cold_indices)} items")
    if len(cold_indices) > 0:
        logger.info(f"  → cold_scores: {cold_scores[:min(3, len(cold_scores))]}")

    # ===================================================================
    # OPTION 2: For warm users, only use cold items as fallback
    # Only return cold recommendations if warm doesn't have enough results
    # ===================================================================
    if is_warm_user:
        # For warm users, prioritize warm (hybrid) recommendations
        # Only use cold items if we don't have enough warm results
        logger.info(f"[RECOMMEND] OPTION 2 STRATEGY: Warm user with {len(warm_indices)} warm items")
        
        if len(warm_indices) >= k:
            # We have enough warm recommendations, return only those
            logger.info(f"[RECOMMEND] Warm items ({len(warm_indices)}) >= k ({k}), returning ONLY warm recommendations")
            all_indices = warm_indices
            all_scores = warm_scores
            all_sources = warm_sources
        else:
            # Use warm as primary, fill gap with cold
            n_needed = k - len(warm_indices)
            logger.info(f"[RECOMMEND] Warm items ({len(warm_indices)}) < k ({k}), need {n_needed} cold items as fallback")
            all_indices = np.concatenate([warm_indices, cold_indices[:n_needed]])
            all_scores = np.concatenate([warm_scores, cold_scores[:n_needed]])
            all_sources = np.concatenate([warm_sources, np.array(["embedding_only"] * min(n_needed, len(cold_indices)), dtype=object)])
    else:
        # For cold users, use only cold (embedding-only) recommendations
        logger.info(f"[RECOMMEND] OPTION 2 STRATEGY: Cold user, returning embedding-only recommendations")
        all_indices = cold_indices
        all_scores = cold_scores
        all_sources = np.array(["embedding_only"] * len(cold_indices), dtype=object)

    if len(all_indices) == 0:
        logger.info(f"[RECOMMEND] No recommendations available")
        return all_indices, all_scores, all_sources

    # Take top-k from what we have
    order = np.argsort(all_scores)[::-1][:k]
    final_sources = all_sources[order]
    final_scores = all_scores[order]
    
    logger.info(f"[RECOMMEND] Final merged results: {len(all_indices[order])} items with sources: {list(final_sources)}")
    logger.info(f"[RECOMMEND] Final scores: {final_scores[:min(3, len(order))]}")
    
    return all_indices[order], final_scores, final_sources
