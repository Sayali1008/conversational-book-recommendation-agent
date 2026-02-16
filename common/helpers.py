import numpy as np


# region NORMALIZATION
def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] range using min-max scaling."""
    scores = np.asarray(scores, dtype=np.float32)
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)


def softmax_normalize(scores: np.ndarray, temperature: float = 0.7) -> np.ndarray:
    """Normalize using softmax with temperature scaling."""
    scores = np.asarray(scores, dtype=np.float32)
    s = (scores - scores.mean()) / (scores.std() + 1e-8)
    s = s / max(temperature, 1e-4)
    e = np.exp(s - s.max())
    return e / (e.sum() + 1e-8)


def zscore_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize using z-score + sigmoid squashing."""
    scores = np.asarray(scores, dtype=np.float32)
    mu, sigma = scores.mean(), scores.std()
    if sigma < 1e-8:
        return np.ones_like(scores) * 0.5
    z = (scores - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


def normalize_scores(scores: np.ndarray, norm: str = "none", norm_metadata: float = None) -> np.ndarray:
    """Normalize scores using specified method."""
    if norm == "softmax":
        return softmax_normalize(scores, norm_metadata or 0.7)
    elif norm == "zscore":
        return zscore_normalize(scores)
    elif norm == "minmax":
        return minmax_normalize(scores)
    else:  # default none
        return scores


# endregion


# region METRICS
def precision_at_k(pred_indices, true_indices, k):
    """
    Compute Precision@K for a single user.
    Interpretation: Of the K items we recommended, how many did the user actually like?
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = pred_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / k


def recall_at_k(pred_indices, true_indices, k):
    """
    Compute Recall@K for a single user.
    Interpretation: Of all the books User liked, we found N of them in our top-K recommendations.
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = pred_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / len(true_indices)


def ap_at_k(pred_indices, true_indices, k):
    """
    Compute Average Precision@K (AP@K) for a single user.
    Key insight: AP@K heavily penalizes missing relevant items at the top of the ranking.
    """
    if len(true_indices) == 0:
        return np.nan

    score = 0.0
    num_hits = 0

    for i, idx in enumerate(pred_indices[:k]):
        if idx in true_indices:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(k, len(true_indices))


def ndcg_at_k(pred_indices, true_indices, k):
    """
    Compute Normalized Discounted Cumulative Gain@K (NDCG@K) for a single user.
    Binary relevance: true_indices are marked as relevant (1), others as irrelevant (0).
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = pred_indices[:k]
    dcg = 0.0

    for i, idx in enumerate(top_k):
        if idx in true_indices:
            dcg += 1.0 / np.log2(i + 2)

    # Compute ideal DCG: best possible ranking (all relevant books first)
    ideal_dcg = 0.0
    for i in range(min(k, len(true_indices))):
        ideal_dcg += 1.0 / np.log2(i + 2)

    if ideal_dcg == 0:
        return np.nan

    return dcg / ideal_dcg


def compute_aggregate_metrics(metric_dict):
    """Compute mean and std of per-user metrics, filtering out NaN values."""
    aggregated = {}

    for metric_name, scores in metric_dict.items():
        scores_arr = np.array(scores)
        valid_scores = scores_arr[~np.isnan(scores_arr)]

        if len(valid_scores) > 0:
            aggregated[metric_name] = {
                "mean": valid_scores.mean(),
                "std": valid_scores.std(),
                "count": len(valid_scores),
            }
        else:
            aggregated[metric_name] = {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
            }

    return aggregated


def mean_average_precision_at_k(model, train_matrix, val_matrix, K=10):
    train_matrix = train_matrix.tocsr() if not isinstance(train_matrix, type(train_matrix.tocsr())) else train_matrix
    val_matrix = val_matrix.tocsr() if not isinstance(val_matrix, type(val_matrix.tocsr())) else val_matrix

    # Vectorized: Score all items for all users at once
    # user_factors: (n_users, factors)
    # item_factors: (n_items, factors)
    # Result: scores = (n_items, n_users)
    scores_all = np.dot(model.item_factors, model.user_factors.T)  # (n_items, n_users)
    
    ap_scores = []
    n_users = val_matrix.shape[0]
    
    # Iterate through users with validation interactions
    for user_id in range(n_users):
        # Get user's validation interactions
        val_row = val_matrix[user_id]
        
        # Skip users with no validation interactions
        if val_row.nnz == 0:
            continue
        
        # Get true positive items in validation set
        true_items = set(val_row.indices)
        
        # Get scores for this user (already computed above)
        scores = scores_all[:, user_id].copy()  # (n_items,)
        
        # CRITICAL: Exclude items the user already interacted with in training set
        # Use boolean mask for faster masking than setting to -inf
        training_items = np.array(train_matrix[user_id].indices)
        scores[training_items] = -np.inf
        
        # Get top-K items from remaining (unseen) items
        # Use partition for O(n) instead of O(n log n) sorting
        if K < len(scores):
            # Partition to get K largest elements
            top_k_idx = np.argpartition(-scores, K-1)[:K]
            # Sort these K elements
            top_k_items = top_k_idx[np.argsort(-scores[top_k_idx])]
        else:
            top_k_items = np.argsort(-scores)
        
        # Filter out -inf items (unseen items only)
        top_k_items = top_k_items[np.isfinite(scores[top_k_items])]
        
        # Compute AP@K for this user
        ap_k = ap_at_k(top_k_items, true_items, K)
        
        # Only include valid scores (not NaN)
        if not np.isnan(ap_k):
            ap_scores.append(ap_k)
    
    # Return mean AP@K across all users
    if len(ap_scores) == 0:
        return 0.0
    
    return np.mean(ap_scores)


# endregion
