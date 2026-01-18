import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from constants import *
from utils import *
from utils import save_pickle

# region [HELPERS]


def _log_interaction_summary(logger, n_users, n_cf_books, train_matrix, val_matrix, test_matrix):
    logger.info("=== Train Matrix ===")
    logger.info("Shape: %s", train_matrix.shape)
    logger.info("Non-zero entries: %s", f"{train_matrix.nnz:,}")
    logger.info(
        "Sparsity: %.4f%%",
        100 * (1 - train_matrix.nnz / (n_users * n_cf_books)),
    )
    logger.info(
        "Density: %.4f%%",
        100 * train_matrix.nnz / (n_users * n_cf_books),
    )

    logger.info("=== Val Matrix ===")
    logger.info("Shape: %s", val_matrix.shape)
    logger.info("Non-zero entries: %s", f"{val_matrix.nnz:,}")
    logger.info(
        "Sparsity: %.4f%%",
        100 * (1 - val_matrix.nnz / (n_users * n_cf_books)),
    )
    logger.info(
        "Density: %.4f%%",
        100 * val_matrix.nnz / (n_users * n_cf_books),
    )

    logger.info("=== Test Matrix ===")
    logger.info("Shape: %s", test_matrix.shape)
    logger.info("Non-zero entries: %s", f"{test_matrix.nnz:,}")
    logger.info(
        "Sparsity: %.4f%%",
        100 * (1 - test_matrix.nnz / (n_users * n_cf_books)),
    )
    logger.info(
        "Density: %.4f%%",
        100 * test_matrix.nnz / (n_users * n_cf_books),
    )

    # Check confidence value distribution in train matrix
    train_values = train_matrix.data
    logger.info("=== Confidence Distribution in Train Matrix ===")
    logger.info(f"Min confidence: {train_values.min()}")
    logger.info(f"Max confidence: {train_values.max()}")
    logger.info(f"Mean confidence: {train_values.mean():.2f}")
    logger.info(f"Confidence value counts:")
    unique, counts = np.unique(train_values, return_counts=True)
    for val, count in zip(unique, counts):
        logger.info(f"  {val}: {count:,} ({100*count/len(train_values):.1f}%)")

    # Check user interaction distribution
    user_interactions = np.array(train_matrix.sum(axis=1)).flatten()
    users_with_1_interaction = (user_interactions == 1).sum()
    logger.info("=== User Interaction Distribution (Train) ===")
    logger.info(f"Min interactions per user: {user_interactions.min()}")
    logger.info(
        f"Users with only 1 user interaction: {users_with_1_interaction:,} ({100*users_with_1_interaction/n_users:.1f}%)"
    )
    logger.info(f"Max interactions per user: {user_interactions.max()}")
    logger.info(f"Mean interactions per user: {user_interactions.mean():.2f}")
    logger.info(f"Median interactions per user: {np.median(user_interactions):.2f}")

    # Check book interaction distribution
    book_interactions = np.array(train_matrix.sum(axis=0)).flatten()
    books_with_0_interactions = (book_interactions == 0).sum()
    logger.info("=== Book Interaction Distribution (Train) ===")
    logger.info(f"Min interactions per book: {book_interactions.min()}")
    logger.info(
        f"Books with 0 book interactions: {books_with_0_interactions:,} ({100*books_with_0_interactions/n_cf_books:.1f}%)"
    )
    logger.info(f"Max interactions per book: {book_interactions.max()}")
    logger.info(f"Mean interactions per book: {book_interactions.mean():.2f}")
    logger.info(f"Median interactions per book: {np.median(book_interactions):.2f}")


def _log_user_coverage(logger, ratings_df, train_df, val_df, test_df):
    logger.info("Train interactions: %s", f"{len(train_df):,}")
    logger.info("Val interactions: %s", f"{len(val_df):,}")
    logger.info("Test interactions: %s", f"{len(test_df):,}")
    logger.info(
        "Split ratio: %.1f%% / %.1f%% / %.1f%%",
        100 * len(train_df) / len(ratings_df),
        100 * len(val_df) / len(ratings_df),
        100 * len(test_df) / len(ratings_df),
    )

    # Log user coverage
    train_users = set(train_df["user_id"])
    all_users = set(ratings_df["user_id"])
    coverage = len(train_users) / len(all_users)
    logger.info(f"Number of users in train: {len(train_users)}")
    logger.info(f"Total number of users: {len(all_users)}")
    logger.info(f"User coverage in train: {coverage:.1%}")


def _log_model_statistics(logger, user_factors, book_factors):
    logger.info("Model and factors saved")

    logger.info("=== Learned Factors ===")
    logger.info(f"User factors shape: {user_factors.shape}")
    logger.info(f"Book factors shape: {book_factors.shape}")

    # Look at a sample user vector
    logger.info(f"Sample user vector (user_cf_idx=0):")
    logger.info(user_factors[0])

    # Look at a sample book vector
    logger.info(f"Sample book vector (book_cf_idx=0):")
    logger.info(book_factors[0])

    # Check factor magnitudes
    logger.info(f"User factor statistics:")
    logger.info(f"  Mean: {user_factors.mean():.4f}")
    logger.info(f"  Std: {user_factors.std():.4f}")
    logger.info(f"  Min: {user_factors.min():.4f}")
    logger.info(f"  Max: {user_factors.max():.4f}")

    logger.info(f"book factor statistics:")
    logger.info(f"  Mean: {book_factors.mean():.4f}")
    logger.info(f"  Std: {book_factors.std():.4f}")
    logger.info(f"  Min: {book_factors.min():.4f}")
    logger.info(f"  Max: {book_factors.max():.4f}")


# endregion


def filter_min_max_interactions(ratings_df):
    # Filtering once is not enough when constraints interact so we use a convergence loop
    # This guarantees that both constraints are simultaneously satisfied in the final dataframe.
    # Without a loop, it is likely that only one filter will stay true at a time.
    while True:
        prev_len = len(ratings_df)

        user_counts = ratings_df["user_id"].value_counts()
        ratings_df = ratings_df[ratings_df["user_id"].isin(user_counts[(user_counts >= MIN_USER_INTERACTIONS)].index)]

        book_counts = ratings_df["book_id"].value_counts()
        ratings_df = ratings_df[ratings_df["book_id"].isin(book_counts[book_counts >= MIN_BOOK_INTERACTIONS].index)]

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


def prepare_data(logger):
    ratings_df = pd.read_feather(PATH_CLEAN_RATINGS)
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
    user_to_cf_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    book_to_cf_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
    save_pickle(user_to_cf_idx, USER_IDX_PKL)
    save_pickle(book_to_cf_idx, BOOK_IDX_PKL)
    logger.info("Index mappings saved.")

    # Add Index Columns to DataFrame
    ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_cf_idx)
    ratings_df["book_idx"] = ratings_df["book_id"].map(book_to_cf_idx)
    assert ratings_df["user_idx"].notna().all(), "Some users failed to map to indices"
    assert ratings_df["book_idx"].notna().all(), "Some books failed to map to indices"
    logger.info("Index mapping complete.")

    # Stratify by user to ensure each user has training data
    train_df, val_df, test_df = stratify_user_distribution(ratings_df)
    _log_user_coverage(logger, ratings_df, train_df, val_df, test_df)

    return n_users, n_cf_books, train_df, val_df, test_df


def build_interaction_matrix(df, n_users, n_cf_books, binary=False):
    """
    Build sparse interaction matrix from DataFrame.

    Args:
        df: DataFrame with columns ['user_idx', 'book_idx', 'confidence']
        n_users: Total number of users
        n_cf_books: Total number of books

    Returns:
        scipy.sparse.csr_matrix of shape (n_users, n_cf_books)
    """
    # uses user_idx and book_idx to map confidences to correct user-book index combinations

    if binary:
        df = df.copy()
        df["confidence"] = 1.0  # Uniform confidence

    row = df["user_idx"].values
    col = df["book_idx"].values
    data = df["confidence"].values

    matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_cf_books), dtype=np.float32)
    return matrix


def model_initialization(logger, factors=32, regularization=0.05, iterations=15, alpha=40):
    """
    Params are the hyperparameters required for model initialization

    :param factors: Number of latent factors
    :param regularization: L2 regularization
    :param iterations: Number of ALS iterations
    :param alpha: Confidence scaling factor (used as: 1 + alpha * rating)
    """

    # Initialize model
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        use_gpu=False,  # Set to True if you have GPU support
        random_state=42,
    )

    logger.info(f"ALS Model Initialized:")
    logger.info(f"  Factors: {factors}")
    logger.info(f"  Regularization: {regularization}")
    logger.info(f"  Iterations: {iterations}")
    logger.info(f"  Alpha: {alpha}")

    return model


def model_training(logger, model, train_matrix):
    # train_matrix = (user x book) sparse matrix
    logger.info(f"Training matrix shape (user x book): {train_matrix.shape}")
    logger.info("Starting training...")
    start_time = time.time()

    # Fit the model
    model.fit(train_matrix, show_progress=True)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    return model


def test_predictions(logger, user_factors, book_factors, user_cf_idx=0, k=10):
    try:
        # Get this user's factor vector
        user_vec = user_factors[user_cf_idx]  # (64,)

        # Compute scores for all books: user_vec · book_factors^T
        scores = book_factors.dot(user_vec)  # (11591,)

        # Get top K books
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]

        logger.info(f"Top {k} predictions for user {user_cf_idx} (manual calculation):")
        logger.info(f"Book indices: {top_k_indices}")
        logger.info(f"Scores: {top_k_scores}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"Score mean: {scores.mean():.4f}")
        logger.info(f"Score std: {scores.std():.4f}")

        # Check if scores are reasonable
        if top_k_scores.max() < 0.01:
            logger.warning("⚠️ Prediction scores are very small! Model may not be learning well.")
        else:
            logger.info("✓ Prediction scores look reasonable")

        return top_k_indices, top_k_scores

    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        return None, None
