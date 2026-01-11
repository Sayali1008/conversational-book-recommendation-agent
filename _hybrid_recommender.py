import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from config import *


def load_index_mappings(pkl_file):
    """
    Load item index mappings from pickle files.

    Returns:
        item_to_cf_idx: dict mapping item_id (book or user) → CF matrix column index
        cf_idx_to_item_id: dict mapping CF matrix column index → item_id
    """
    with open(pkl_file, "rb") as f:
        item_to_cf_idx = pickle.load(f)

    # Create reverse mapping: CF index → item_id
    cf_idx_to_item_id = {cf_idx: item_id for item_id, cf_idx in item_to_cf_idx.items()}

    return item_to_cf_idx, cf_idx_to_item_id


def cf_idx_to_catalog_id(cf_idx, book_cf_idx_to_book_id):
    """
    Convert CF matrix column index to catalog row index.

    Args:
        cf_idx: Index in CF matrix (0 to n_cf_books-1)
        book_cf_idx_to_book_id: Mapping from CF index to book_id

    Returns:
        catalog_id: Row index in catalog (0 to n_catalog_books-1)
    """
    book_id = book_cf_idx_to_book_id[cf_idx]
    # book_id was assigned sequentially starting at 1, so catalog_id = book_id - 1
    # Eg. catalog_books_df.iloc[i] has book_id = i + 1
    catalog_id = book_id - 1
    return catalog_id


def create_user_embedding_profile(
    user_idx, train_matrix, catalog_embeddings, idx_to_book_id
):
    """
    Create user's embedding profile from their interaction history.

    Args:
        user_idx: User's index in CF matrix (0 - (n_users-1))
        train_matrix: Sparse interaction matrix (n_users, n_cf_books)
        catalog_embeddings: All book embeddings (n_catalog_books, n_dim)
        idx_to_book_id: Mapping from CF index to book_id

    Returns:
        user_profile: (n_dim,) array representing user in embedding space
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
    rated_catalog_indices = np.array(
        [cf_idx_to_catalog_id(cf_idx, idx_to_book_id) for cf_idx in rated_cf_indices]
    )
    rated_embeddings = catalog_embeddings[rated_catalog_indices]  # (n_rated, n_dim)

    # Weighted average by confidence (higher rated books have more influence)
    user_profile = np.average(rated_embeddings, axis=0, weights=confidences)

    return user_profile  # (n_dim,)


def normalize_scores(scores):
    """Normalize scores to [0, 1] range."""
    scores = np.array(scores)

    # Currently, using minmax normalization
    # Scale to [0, 1]
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)


def build_cf_to_catalog_mapping(book_cf_idx_to_book_id):
    """
    Build mapping from CF book indices to catalog indices.

    Args:
        book_cf_idx_to_book_id: Mapping from CF index to book_id

    Returns:
        dict: CF index → catalog index mapping
    """
    cf_to_catalog_map = {}
    for cf_idx, book_id in book_cf_idx_to_book_id.items():
        catalog_idx = book_id - 1  # book_id starts at 1, catalog indices start at 0
        cf_to_catalog_map[cf_idx] = catalog_idx

    return cf_to_catalog_map


def hybrid_recommender(user_idx, user_factors, book_factors, train_matrix, catalog_embeddings, cf_to_catalog_map, 
                       book_cf_idx_to_book_id, k=10, lambda_weight=0.5, filter_rated=True,):
    """
    Generate hybrid CF + embedding recommendations.

    Args:
        user_idx: User's CF index
        user_factors: (n_users, n_factors) CF user factors
        book_factors: (n_cf_books, n_factors) CF book factors
        train_matrix: (n_users, n_cf_books) interaction matrix
        catalog_embeddings: (n_catalog_books, n_dim) all book embeddings
        cf_to_catalog_map: Dict mapping CF indices to catalog indices
        idx_to_book_id: Mapping from CF index to book_id (n_cf_books key-value pairs)
        k: Number of recommendations
        lambda_weight: Weight for CF (0=pure embedding, 1=pure CF)
        filter_rated: Whether to exclude already-rated books

    Returns:
        book_indices: (k,) catalog indices of recommended books
        scores: (k,) hybrid scores
        sources: (k,) strings indicating score source ('hybrid', 'embedding_only')
    """
    print(f"user_idx={user_idx} | K={k} | lambda={lambda_weight}: hybrid_recommender()")

    n_catalog_books = len(catalog_embeddings)
    n_cf_books = len(book_factors)

    # Create user embedding profile
    # NOTE
    # user_profile is the 'likeness profile' for the user created by blending the semantics of all books they've read.
    # Each number in the resulting (n_dim,) vector represents how much the user
    # values a specific semantic feature (like 'dark humor' or 'technical detail')
    # based on the average of their highly-rated books.
    user_profile = create_user_embedding_profile(
        user_idx, train_matrix, catalog_embeddings, book_cf_idx_to_book_id
    ) # (n_dim,)

    # Compute CF scores (for CF-trainable books only)
    user_vec = user_factors[user_idx]  # (n_factors,) 
    cf_scores = book_factors.dot(user_vec)  # (n_cf_books,)
    # cf_scores[i] = score for CF column i

    # Compute embedding similarity scores (for all books)
    # Cosine similarity: dot product of normalized vectors
    # embedding_scores[i] = score for catalog row i
    embedding_scores = cosine_similarity(
        user_profile.reshape(1, -1), catalog_embeddings
    ).flatten()  # (n_catalog,)

    # Normalize and combine both scores
    cf_scores_norm = normalize_scores(cf_scores)
    embedding_scores_norm = normalize_scores(embedding_scores)

    final_scores = np.zeros(n_catalog_books)
    sources = np.array(["unknown"] * n_catalog_books, dtype=object)

    # For CF-trainable books: hybrid score
    # lambda_weight = 0 is pure embedding
    # lambda_weight = 1 is pure CF
    for cf_idx in range(n_cf_books):
        catalog_idx = cf_to_catalog_map[cf_idx]
        final_scores[catalog_idx] = (
            lambda_weight * cf_scores_norm[cf_idx]
            + ((1 - lambda_weight) * embedding_scores_norm[catalog_idx])
        )
        sources[catalog_idx] = "hybrid"

    # For cold-start books: embedding only
    cf_catalog_indices = set(cf_to_catalog_map.values())
    for catalog_idx in range(n_catalog_books):
        if catalog_idx not in cf_catalog_indices:
            final_scores[catalog_idx] = embedding_scores_norm[catalog_idx]
            sources[catalog_idx] = "embedding_only"

    # Filter already-rated books
    # Instead of deleting the books from the list, it sets their score to negative infinity. 
    # When the system later sorts by "highest score", these already-read books will automatically 
    # drop to the absolute bottom of the list and never be shown.
    if filter_rated:
        user_row = train_matrix[user_idx].toarray().flatten()
        rated_cf_indices = np.where(user_row > 0)[0]
        rated_catalog_indices = [cf_to_catalog_map[cf_idx] for cf_idx in rated_cf_indices]
        final_scores[rated_catalog_indices] = -np.inf

    # Get top-k
    top_k_catalog_indices = np.argsort(final_scores)[::-1][:k]
    top_k_scores = final_scores[top_k_catalog_indices]
    top_k_sources = sources[top_k_catalog_indices]

    return top_k_catalog_indices, top_k_scores, top_k_sources


def display_recommendations(logger, book_indices, scores, sources, catalog_df):
    """Pretty print recommendations"""
    logger.info(f"\n{'=' * REPEATS}")
    logger.info(f"{'Rank':<6} {'Title':<40} {'Score':<8} {'Source':<15}")
    logger.info(f"{'=' * REPEATS}")

    for rank, (idx, score, source) in enumerate(zip(book_indices, scores, sources), 1):
        book = catalog_df.iloc[idx]
        title = book["title"][:37] + "..." if len(book["title"]) > 40 else book["title"]
        logger.info(f"{rank:<6} {title:<40} {score:.4f}   {source:<15}")
