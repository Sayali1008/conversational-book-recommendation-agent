"""
Centralized configuration for the recommendation system pipeline.
Defines all paths, hyperparameters, and constants used across stages.
"""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MATRICES_DIR = DATA_DIR / "matrices"
PKL_DIR = DATA_DIR / "pkl"
MODEL_DIR = DATA_DIR / "model"
LOGS_DIR = PROJECT_ROOT / "logs"
APP_LOGS_DIR = PROJECT_ROOT / "logs" / "app_logs"
EVAL_LOGS_DIR = PROJECT_ROOT / "logs" / "eval_logs"

date_str = datetime.now().strftime("%m%d%Y")
APP_LOG_FILE = str(APP_LOGS_DIR / f"{date_str}_1.log")

REPEATS = 100

DATA_PREPROCESSING = {
    # columns
    "input_cols_books": ["title", "description", "authors", "infolink", "categories"],
    "input_cols_ratings": [
        "title",
        "user_id",
        "profilename",
        "review/helpfulness",
        "review/score",
        "review/time",
        "review/summary",
        "review/text",
    ],
    "output_cols_books": ["book_id", "title", "authors", "description", "genres", "infolink"],
    "output_cols_ratings": [
        "book_id",
        "user_id",
        "review/score",
        "confidence",
        "datetime",
        "review/summary",
        "review/text",
    ],
    # configurations
    "min_desc_length": 10,
    "top_n_genres": 50,
    "common_delims": [";", "|", "/", "•"],
    "min_user_interactions": 5,
    "min_book_interactions": 5,
    "max_user_interactions": 500,
}

EMBEDDINGS = {
    "batch_size": 64,
    "embedding_model": "all-MiniLM-L6-v2",  # "all-mpnet-base-v2"
}
# Example output: (num_rows, 384) for all-MiniLM-L6-v2
# Example output: (num_rows, 768) for all-mpnet-base-v2

EMBEDDINGS["dim"] = 384 if EMBEDDINGS["embedding_model"] == "all-MiniLM-L6-v2" else 768

INTERACTION_MATRIX = {
    "train_test_split": 0.8,
    "val_test_split": 0.5,
    "random_state": 42,
}

CF_MODEL_PARAMS = {
    "factors": 50,
    "regularization": 0.01,
    "iterations": 20,
    "alpha": 40,
    # "book_factor_scale": 10.0,
}

# Evaluation parameters
EVALUATION = {
    "k_values": [5, 10],
    "lambda_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "norm": "softmax",
    "norm_metadata": 0.9,  # Temperature for softmax during evaluation
    "min_validation_items": 2,
    "min_confidence": 1,
}

RECOMMEND = {
    "k": 10,
    "candidate_pool_size": 200,
    "lambda_weight": 0.7,
    "norm": "minmax",  # "minmax", "softmax", "zscore"
    "norm_metadata": None,  # Temperature for softmax (e.g., 0.01, 0.3, 0.9)
    "filter_rated": True,  # Exclude already-rated books from recommendations
    "book_factor_scale": 10.0,  # Scale book factors to address tiny raw values
}

# Cold-start strategy
# COLD_START_USE_EMBEDDING_ONLY = True  # Use embedding-only for cold users and items
# COLD_START_FALLBACK_ONLY = True  # For warm users, only use cold items as fallback if warm has <k results

PATHS = {
    # data preprocessing
    "books_path": str(RAW_DATA_DIR / "books_data.csv"),
    "ratings_path": str(RAW_DATA_DIR / "books_rating.csv"),
    "clean_books_path": str(CLEAN_DATA_DIR / "cleaned_books_data.ftr"),
    "clean_ratings_path": str(CLEAN_DATA_DIR / "cleaned_ratings_data.ftr"),
    # embeddings
    "catalog_books_index_path": str(EMBEDDINGS_DIR / f"catalog_books_{EMBEDDINGS['dim']}.index"),
    "catalog_books_embeddings_path": str(EMBEDDINGS_DIR / f"catalog_books_{EMBEDDINGS['dim']}.npy"),
    # interaction matrices
    "train_matrix_path": str(MATRICES_DIR / "train_matrix.npz"),
    "val_matrix_path": str(MATRICES_DIR / "val_matrix.npz"),
    "test_matrix_path": str(MATRICES_DIR / "test_matrix.npz"),
    "user_idx_pkl": str(PKL_DIR / "user_to_idx.pkl"),
    "book_idx_pkl": str(PKL_DIR / "book_to_idx.pkl"),
    # model artifacts
    "als_model_path": str(MODEL_DIR / "als_model.pkl"),
    "user_factors_path": str(MODEL_DIR / "user_factors.npy"),
    "book_factors_path": str(MODEL_DIR / "book_factors.npy"),
}
