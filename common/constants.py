"""
Centralized configuration for the recommendation system pipeline.
Defines all paths, hyperparameters, and constants used across stages.
"""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/Users/sayalimoghe/Documents/Career/GitHub/conversational-book-recommendation-agent")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
DATABASE_DIR = DATA_DIR / "database"
MODEL_DIR = DATA_DIR / "model"
BACKUP_DIR = DATA_DIR / "backup"
LOGS_DIR = PROJECT_ROOT / "logs"
APP_LOGS_DIR = PROJECT_ROOT / "logs" / "app_logs"
EVAL_LOGS_DIR = PROJECT_ROOT / "logs" / "eval_logs"

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
        "profilename",
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


CROSS_VALIDATION = {
    "num_folds": 3,
    "seed": 42,
    "candidate_pool_size_values": [100, 300, 500, 700, 900],
    "lambda_values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "top_k": 10,
}

# Evaluation parameters
EVALUATION = {
    "type": "CF",  # CF, CB, default = CF
    "norm": "none",
    "norm_metadata": None,  # Temperature for softmax during evaluation # previous values: 0.9
    "min_validation_items": 2,
    "filter_rated": True,
}

PATHS = {
    # data preprocessing
    "books": str(RAW_DATA_DIR / "books_data.csv"),
    "ratings": str(RAW_DATA_DIR / "books_rating.csv"),
    "clean_books": str(CLEAN_DATA_DIR / "cleaned_books_data.ftr"),
    "clean_ratings": str(CLEAN_DATA_DIR / "cleaned_ratings_data.ftr"),
    "database": str(DATABASE_DIR / "system.db"),
    # embeddings
    "catalog_books_index": str(CLEAN_DATA_DIR / f"catalog_books_{EMBEDDINGS['dim']}.index"),
    "catalog_books_embeddings": str(CLEAN_DATA_DIR / f"catalog_books_{EMBEDDINGS['dim']}.npy"),
    # interaction matrices
    "train_matrix": str(MODEL_DIR / "train_matrix.npz"),
    "val_matrix": str(MODEL_DIR / "val_matrix.npz"),
    # model artifacts
    "user_idx_pkl": str(MODEL_DIR / "user_to_idx.pkl"),
    "book_idx_pkl": str(MODEL_DIR / "book_to_idx.pkl"),
    "als_model": str(MODEL_DIR / "als_model.pkl"),
    "user_factors": str(MODEL_DIR / "user_factors.npy"),
    "book_factors": str(MODEL_DIR / "book_factors.npy"),
    "best_model_params": str(MODEL_DIR / "best_model_params.pkl"),
    "best_rec_params": str(MODEL_DIR / "best_rec_params.pkl"),
    # logs
    "app_log_file": str(APP_LOGS_DIR / f"{date_str}.log"),
    "eval_log_file": str(EVAL_LOGS_DIR / f"{date_str}.log"),
}
