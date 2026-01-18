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
