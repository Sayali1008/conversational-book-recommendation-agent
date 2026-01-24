import numpy as np

 
# region Normalization
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


def normalize_scores(scores: np.ndarray, norm: str, norm_metadata: float = None) -> np.ndarray:
    """Normalize scores using specified method."""
    if norm == "softmax":
        return softmax_normalize(scores, norm_metadata or 0.7)
    elif norm == "zscore":
        return zscore_normalize(scores)
    else:  # minmax
        return minmax_normalize(scores)


# endregion


# region Metrics
def precision_at_k(recommended_indices, true_indices, k):
    """
    Compute Precision@K for a single user.
    Interpretation: Of the K items we recommended, how many did the user actually like?
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / k


def recall_at_k(recommended_indices, true_indices, k):
    """
    Compute Recall@K for a single user.
    Interpretation: Of all the books User liked, we found N of them in our top-K recommendations.
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / len(true_indices)


def ap_at_k(recommended_indices, true_indices, k):
    """
    Compute Average Precision@K (AP@K) for a single user.
    Key insight: AP@K heavily penalizes missing relevant items at the top of the ranking.
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    score = 0.0
    num_hits = 0

    for i, idx in enumerate(top_k):
        if idx in true_indices:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(k, len(true_indices))


def ndcg_at_k(recommended_indices, true_indices, k):
    """
    Compute Normalized Discounted Cumulative Gain@K (NDCG@K) for a single user.
    Binary relevance: true_indices are marked as relevant (1), others as irrelevant (0).
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
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


# endregion
