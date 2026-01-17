"""
Centralized configuration for the recommendation system pipeline.
Defines all paths, hyperparameters, and constants used across stages.
"""

from datetime import datetime
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
MODEL_DIR = DATA_DIR / "model"
LOGS_DIR = PROJECT_ROOT / "logs"

date_str = datetime.now().strftime("%m%d%Y")
LOG_FILE = str(LOGS_DIR / f"{date_str}_3.log")

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
MIN_BOOK_INTERACTIONS = 5
MAX_USER_INTERACTIONS = 500

# ============================================================
# Stage 2: Semantic Search (Embeddings)
# ============================================================
# Configuration
BATCH_SIZE = 64
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "all-mpnet-base-v2"
# Example output: (num_rows, 384) for all-MiniLM-L6-v2
# Example output: (num_rows, 768) for all-mpnet-base-v2

# Output files
dim = 384 if EMBEDDING_MODEL == 'all-MiniLM-L6-v2' else 768
OUTPUT_CATALOG_BOOKS_INDEX = str(EMBEDDINGS_DIR / f"catalog_books_{dim}.index")
OUTPUT_CATALOG_BOOKS_EMBEDDINGS = str(EMBEDDINGS_DIR / f"catalog_books_{dim}.npy")
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
OUTPUT_ALS_MODEL = str(MODEL_DIR / "als_model.pkl")
OUTPUT_USER_FACTORS = str(MODEL_DIR / "user_factors.npy")
OUTPUT_BOOK_FACTORS = str(MODEL_DIR / "book_factors.npy")


# ============================================================
# Hybrid Recommender Configuration
# ============================================================
CF_CANDIDATE_POOL_SIZE = 200
FINAL_K = 10

# Hybrid recommendation parameters
HYBRID_LAMBDA_WEIGHT = 0.65  # 0=pure embedding, 1=pure CF
HYBRID_CANDIDATE_POOL_SIZE = 300  # Number of CF candidates to consider before embedding re-ranking
HYBRID_FILTER_RATED = True  # Exclude already-rated books from recommendations
HYBRID_NORM = "minmax"  # Normalization method: "minmax", "softmax", "zscore"
HYBRID_NORM_METADATA = None  # Temperature for softmax (e.g., 0.01, 0.3, 0.9)

# Cold-start strategy
COLD_START_USE_EMBEDDING_ONLY = True  # Use embedding-only for cold users and items
COLD_START_FALLBACK_ONLY = True  # For warm users, only use cold items as fallback if warm has <k results

# Evaluation parameters
EVAL_K_VALUES = [5, 10]
EVAL_LAMBDA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EVAL_NORM = "softmax"
EVAL_NORM_METADATA = 0.9  # Temperature for softmax during evaluation
EVAL_CANDIDATE_POOL_SIZE = 300
EVAL_MIN_VALIDATION_ITEMS = 2
EVAL_MIN_CONFIDENCE = 1
EVAL_FILTER_RATED = True