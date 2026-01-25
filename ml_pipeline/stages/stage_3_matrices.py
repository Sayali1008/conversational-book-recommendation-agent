import numpy as np
import pandas as pd
import scipy.sparse as sp

from common.constants import PATHS, INTERACTION_MATRIX
from common.logging import log_user_coverage
from common.utils import save_pickle, setup_logging

logger = setup_logging(__name__, PATHS["eval_log_file"])


def filter_min_max_interactions(ratings_df):
    # Filtering once is not enough when constraints interact so we use a convergence loop
    # This guarantees that both constraints are simultaneously satisfied in the final dataframe.
    # Without a loop, it is likely that only one filter will stay true at a time.
    while True:
        prev_len = len(ratings_df)

        user_counts = ratings_df["user_id"].value_counts()
        ratings_df = ratings_df[
            ratings_df["user_id"].isin(user_counts[(user_counts >= INTERACTION_MATRIX["min_user_interactions"])].index)
        ]

        book_counts = ratings_df["book_id"].value_counts()
        ratings_df = ratings_df[
            ratings_df["book_id"].isin(book_counts[book_counts >= INTERACTION_MATRIX["min_book_interactions"]].index)
        ]

        if len(ratings_df) == prev_len:
            break

    return ratings_df


def stratify_user_distribution(ratings_df, seed=42):
    df = ratings_df.copy()

    rng = np.random.RandomState(seed)
    df["_rand"] = rng.rand(len(df))

    # Random order within each user
    df["_rank"] = df.groupby("user_id")["_rand"].rank(method="first")
    df["_n"] = df.groupby("user_id")["user_id"].transform("size")

    # Default everything to train
    df["split"] = "train"

    # Users with exactly 2 interactions → 1 validation
    mask_2 = df["_n"] == 2
    df.loc[mask_2 & (df["_rank"] == 2), "split"] = "val"

    # Users with 3+ interactions
    mask_3p = df["_n"] >= 3

    # Highest ranks → validation (preferred)
    df.loc[mask_3p & (df["_rank"] > 0.9 * df["_n"]), "split"] = "val"

    # Just before that → test
    df.loc[mask_3p & (df["_rank"] > 0.8 * df["_n"]) & (df["_rank"] <= 0.9 * df["_n"]), "split"] = "test"

    # Build final splits
    train_df = df[df["split"] == "train"].drop(columns=["_rand", "_rank", "_n", "split"])
    val_df = df[df["split"] == "val"].drop(columns=["_rand", "_rank", "_n", "split"])
    test_df = df[df["split"] == "test"].drop(columns=["_rand", "_rank", "_n", "split"])

    # Supporting warm-start handling only (TODO: change this later)
    train_books = set(train_df["book_id"])
    val_df = val_df[val_df["book_id"].isin(train_books)]
    test_df = test_df[test_df["book_id"].isin(train_books)]

    return train_df, val_df, test_df


def prepare_data():
    ratings_df = pd.read_feather(PATHS["clean_ratings"])
    logger.info(f"Ratings data size before filtering min/max interactions: {ratings_df.shape}")

    ratings_df = filter_min_max_interactions(ratings_df)
    logger.info(f"Ratings data size after filtering min/max interactions: {ratings_df.shape}")

    # Create Index Mappings
    # Example: unique_books = [5, 12, 8, 100, 7, ...]  (in order of appearance)
    unique_users = ratings_df["user_id"].unique()
    unique_books = ratings_df["book_id"].unique()

    n_users = len(unique_users)
    n_cf_books = len(unique_books)
    logger.info("Users: %s", f"{n_users:,}")
    logger.info("CF-trainable books: %s", f"{n_cf_books:,}")

    # idx helps define the appearance order in ratings_df - hence can be used in reverse as well.
    user_id_to_cf = {user_id: idx for idx, user_id in enumerate(unique_users)}
    book_id_to_cf = {book_id: idx for idx, book_id in enumerate(unique_books)}
    save_pickle(user_id_to_cf, PATHS["user_idx_pkl"])
    save_pickle(book_id_to_cf, PATHS["book_idx_pkl"])
    logger.info("Index mappings saved.")

    # Add Index Columns to DataFrame
    ratings_df["user_idx"] = ratings_df["user_id"].map(user_id_to_cf)
    ratings_df["book_idx"] = ratings_df["book_id"].map(book_id_to_cf)
    assert ratings_df["user_idx"].notna().all(), "Some users failed to map to indices"
    assert ratings_df["book_idx"].notna().all(), "Some books failed to map to indices"
    logger.info("Index mapping complete.")

    # Stratify by user to ensure each user has training data
    train_df, val_df, test_df = stratify_user_distribution(ratings_df)
    log_user_coverage(logger, ratings_df, train_df, val_df, test_df)

    return n_users, n_cf_books, train_df, val_df, test_df


def build_interaction_matrix(df, n_users, n_cf_books, binary=False):
    """Build sparse interaction matrix from DataFrame."""

    # uses user_idx and book_idx to map confidences to correct user-book index combinations

    if binary:
        df = df.copy()
        df["confidence"] = 1.0  # Uniform confidence

    row = df["user_idx"].values
    col = df["book_idx"].values
    data = df["confidence"].values

    matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_cf_books), dtype=np.float32)
    return matrix
