"""
Centralized configuration for the recommendation system pipeline.
Defines all paths, hyperparameters, and constants used across stages.
"""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path('/Users/sayalimoghe/Documents/Career/GitHub/conversational-book-recommendation-agent')
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
DATABASE_DIR = DATA_DIR / "database"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MATRICES_DIR = DATA_DIR / "matrices"
PKL_DIR = DATA_DIR / "pkl"
MODEL_DIR = DATA_DIR / "model"
LOGS_DIR = PROJECT_ROOT / "logs"
APP_LOGS_DIR = PROJECT_ROOT / "logs" / "app_logs"
EVAL_LOGS_DIR = PROJECT_ROOT / "logs" / "eval_logs"

# Ensure directories exist
dirs = [
    DATA_DIR,
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    DATABASE_DIR,
    EMBEDDINGS_DIR,
    MATRICES_DIR,
    PKL_DIR,
    MODEL_DIR,
    LOGS_DIR,
    APP_LOGS_DIR,
    EVAL_LOGS_DIR
]
for dir in dirs:
    dir.mkdir(parents=True, exist_ok=True)

date_str = datetime.now().strftime("%Y%m%d")

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
    "min_user_interactions": 5,
    "min_book_interactions": 5,
    "max_user_interactions": 500,
}

CF_MODEL_PARAMS = {
    "random_state": 42,
    "factors": 50,
    "regularization": 0.01,
    "iterations": 20,
    "alpha": 40,
}

# Evaluation parameters
EVALUATION = {
    "type": "CF", # CF, CB, default = CF
    "k_values": [5, 10],
    "lambda_values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "norm": "softmax",
    "norm_metadata": 0.9,  # Temperature for softmax during evaluation
    "min_validation_items": 2,
    "min_confidence": 1,
    "candidate_pool_size": 200,
    "filter_rated": True,  # Exclude already-rated books from recommendations
}

RECOMMEND = {
    "k": 3,
    "candidate_pool_size": 200,
    "lambda_weight": 0.7,
    "norm": "minmax",  # "minmax", "softmax", "zscore"
    "norm_metadata": None,  # Temperature for softmax (e.g., 0.01, 0.3, 0.9)
    "filter_rated": True,  # Exclude already-rated books from recommendations
}

PATHS = {
    # data preprocessing
    "books": str(RAW_DATA_DIR / "books_data.csv"),
    "ratings": str(RAW_DATA_DIR / "books_rating.csv"),
    "clean_books": str(CLEAN_DATA_DIR / "cleaned_books_data.ftr"),
    "clean_ratings": str(CLEAN_DATA_DIR / "cleaned_ratings_data.ftr"),
    # embeddings
    "catalog_books_index": str(EMBEDDINGS_DIR / f"catalog_books_{EMBEDDINGS['dim']}.index"),
    "catalog_books_embeddings": str(EMBEDDINGS_DIR / f"catalog_books_{EMBEDDINGS['dim']}.npy"),
    # interaction matrices
    "train_matrix": str(MATRICES_DIR / "train_matrix.npz"),
    "val_matrix": str(MATRICES_DIR / "val_matrix.npz"),
    "test_matrix": str(MATRICES_DIR / "test_matrix.npz"),
    "user_idx_pkl": str(PKL_DIR / "user_to_idx.pkl"),
    "book_idx_pkl": str(PKL_DIR / "book_to_idx.pkl"),
    # model artifacts
    "als_model": str(MODEL_DIR / "als_model.pkl"),
    "user_factors": str(MODEL_DIR / "user_factors.npy"),
    "book_factors": str(MODEL_DIR / "book_factors.npy"),
    "database": str(DATABASE_DIR / "system.db"),
    # logs
    "app_log_file": str(APP_LOGS_DIR / f"{date_str}.log"),
    "eval_log_file": str(EVAL_LOGS_DIR / f"{date_str}.log"),
}