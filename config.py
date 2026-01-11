"""
Centralized configuration for the recommendation system pipeline.
Defines all paths, hyperparameters, and constants used across stages.
"""

import os
from pathlib import Path

# ============================================================
# Project Structure
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MATRICES_DIR = DATA_DIR / "matrices"
PKL_DIR = DATA_DIR / "pkl"
FACTORS_DIR = DATA_DIR / "factors"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = str(LOGS_DIR / "app.log")

# Create directories if they don't exist
dir_list = [CLEAN_DATA_DIR, EMBEDDINGS_DIR, MATRICES_DIR, PKL_DIR, FACTORS_DIR, LOGS_DIR, ]
for directory in dir_list:
    os.makedirs(directory, exist_ok=True)

REPEATS = 100

# ============================================================
# Stage 1: Data Preprocessing
# ============================================================
# Input files
INPUT_BOOKS = str(RAW_DATA_DIR / "books_data.csv")
INPUT_RATINGS = str(RAW_DATA_DIR / "books_rating.csv")
INPUT_COLS_BOOKS = ["title", "description", "authors", "infolink", "categories"]
INPUT_COLS_RATINGS = [
    "title",
    "user_id",
    "profilename",
    "review/helpfulness",
    "review/score",
    "review/time",
    "review/summary",
    "review/text",
]

# Output files
OUTPUT_BOOKS = str(CLEAN_DATA_DIR / "cleaned_books_data.ftr")
OUTPUT_COLS_BOOKS = ["book_id", "title", "authors", "description", "genres", "infolink"]
OUTPUT_RATINGS = str(CLEAN_DATA_DIR / "cleaned_ratings_data.ftr")
OUTPUT_COLS_RATINGS = [
    "book_id",
    "user_id",
    "review/score",
    "confidence",
    "datetime",
    "review/summary",
    "review/text",
]

# Configuration
MIN_DESC_LENGTH = 10
TOP_N_GENRES = 50
COMMON_DELIMS = [";", "|", "/", "•"]
MIN_USER_INTERACTIONS = 5
MAX_USER_INTERACTIONS = 500
MIN_BOOK_INTERACTIONS = 5

# ============================================================
# Stage 2: Semantic Search (Embeddings)
# ============================================================
# Output files
OUTPUT_CATALOG_BOOKS_INDEX = str(EMBEDDINGS_DIR / "catalog_books.index")
OUTPUT_CATALOG_BOOKS_EMBEDDINGS = str(EMBEDDINGS_DIR / "catalog_books.npy")

# Configuration
BATCH_SIZE = 64
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2"
# Example output: (num_rows, 384) for all-MiniLM-L6-v2
# Example output: (num_rows, 768) for all-mpnet-base-v2

# ============================================================
# Stage 3: Build Interaction Matrix
# ============================================================
# Input files
USER_IDX_PKL = str(PKL_DIR / "user_to_idx.pkl")
BOOK_IDX_PKL = str(PKL_DIR / "book_to_idx.pkl")

# Output files
OUTPUT_TRAIN_MATRIX = str(MATRICES_DIR / "train_matrix.npz")
OUTPUT_VAL_MATRIX = str(MATRICES_DIR / "val_matrix.npz")
OUTPUT_TEST_MATRIX = str(MATRICES_DIR / "test_matrix.npz")

# Configuration
TRAIN_TEST_SPLIT = 0.8
VAL_TEST_SPLIT = 0.5
RANDOM_STATE = 42

# ============================================================
# Stage 4: Train Collaborative Filtering
# ============================================================
# Output files
OUTPUT_ALS_MODEL = str(PKL_DIR / "als_model.pkl")
OUTPUT_USER_FACTORS = str(FACTORS_DIR / "user_factors.npy")
OUTPUT_BOOK_FACTORS = str(FACTORS_DIR / "book_factors.npy")
