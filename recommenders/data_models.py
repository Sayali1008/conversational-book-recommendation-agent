"""
Type definitions for the recommendation system.
Using TypedDicts for structured data with type hints.
"""

from typing import TypedDict, Optional, Dict, Any, List
import numpy as np
import scipy.sparse as sp
import pandas as pd


class IndexMappings(TypedDict):
    """Index mapping structures for converting between different ID spaces."""
    cf_idx_to_catalog_id: Dict[int, int]  # CF matrix book index → catalog row index
    user_to_cf_idx: Dict[Any, int]  # User ID → CF matrix user index
    cf_idx_to_user: Dict[int, Any]  # CF matrix user index → User ID
    cf_idx_to_book: Dict[int, int]  # CF matrix book index → Book ID
    book_id_to_catalog_idx: Dict[int, int]  # Book ID → catalog row index


class RecommendationContext(TypedDict):
    """
    World state - all the pre-loaded data needed to make recommendations.
    Load once at startup, passed to all recommendation functions.
    """
    # factors are generated from the trained model
    user_factors: np.ndarray  # Shape: (n_users, n_factors)
    book_factors: np.ndarray  # Shape: (n_cf_books, n_factors)

    train_matrix: sp.spmatrix  # Shape: (n_users, n_cf_books) - sparse user-item interaction matrix
    catalog_embeddings: np.ndarray  # Shape: (n_catalog_books, embedding_dim)
    index_mappings: IndexMappings
    catalog_df: pd.DataFrame  # Metadata for all books in catalog


class ModelConfig(TypedDict, total=False):
    """
    Configuration for how recommendation models behave internally.
    Learned/training hyperparameters.
    """
    factors: int  # Dimensionality of latent factors
    regularization: float  # L2 regularization strength
    iterations: int  # Number of ALS iterations
    alpha: int  # Confidence weighting for implicit feedback
    # book_factor_scale: float  # Scale up tiny factor values


class EvalConfig(TypedDict, total=False):
    """
    Configuration for how recommendation results are computed, normalized, and merged.
    Runtime behavior parameters.
    """
    # Scoring normalization
    norm: str  # "minmax", "softmax", or "zscore"
    norm_metadata: Optional[float]  # Temperature for softmax (e.g., 0.01, 0.3, 0.9)
    
    lambda_weight: float  # 0=pure embedding, 1=pure CF
    k: int  # Number of recommendations to return
    candidate_pool_size: int  # Size of candidate pool before re-ranking
    filter_rated: bool  # Exclude already-rated items from recommendations
    k_values: List[int]  # K values to evaluate [5, 10, ...]
    lambda_values: List[float]  # Lambda values to evaluate [0.0, 0.5, 1.0, ...]
    min_validation_items: int  # Minimum validation items required
    min_confidence: int  # Minimum confidence threshold

