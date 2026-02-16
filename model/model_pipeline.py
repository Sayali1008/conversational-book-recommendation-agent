from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import ParameterGrid

from common.constants import *
from common.helpers import *
from common.logging import log_timing_summary
from common.utils import setup_logging
from recommenders.handler import get_recommendations

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


def build_cv_folds(ratings_df, num_folds=5, seed=42):
    """
    Build K folds for CV.
    Returns: list of (train_df, val_df) tuples
    """
    df = ratings_df.copy()
    rng = np.random.RandomState(seed)
    df["_rand"] = rng.rand(len(df))
    df["_rank"] = df.groupby("user_id")["_rand"].rank(method="first")
    df["_n"] = df.groupby("user_id")["user_id"].transform("size")

    # per user, we have 80% data in train and 20% in val
    folds = []
    for fold_idx in range(num_folds):
        lower = fold_idx / num_folds
        upper = (fold_idx + 1) / num_folds

        df["split"] = "train"
        mask = df["_n"] >= 2  # users with at least 2 ratings are used for validation
        df.loc[mask & (df["_rank"] > lower * df["_n"]) & (df["_rank"] <= upper * df["_n"]), "split"] = "val"

        train_df = df[df["split"] == "train"].drop(columns=["_rand", "_rank", "_n", "split"])
        val_df = df[df["split"] == "val"].drop(columns=["_rand", "_rank", "_n", "split"])

        train_books = set(train_df["book_id"])
        val_df = val_df[val_df["book_id"].isin(train_books)]

        folds.append((train_df, val_df))

    return folds


def create_train_val_split(ratings_df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a single 80-20 train/val split stratified by user.

    Unlike CV folds, this creates non-overlapping train and val sets.
    Each user has 80% of their ratings in train, 20% in val.
    """
    df = ratings_df.copy()
    rng = np.random.RandomState(seed)

    # Create random ordering within each user
    df["_rand"] = rng.rand(len(df))
    df["_rank"] = df.groupby("user_id")["_rand"].rank(method="first")
    df["_n"] = df.groupby("user_id")["user_id"].transform("size")

    # Split: 80% train, 20% val per user
    df["split"] = "train"
    mask = df["_n"] >= 2  # Users with at least 2 ratings can have validation
    df.loc[mask & (df["_rank"] / df["_n"] > 0.8), "split"] = "val"

    train_df = df[df["split"] == "train"].drop(columns=["_rand", "_rank", "_n", "split"])
    val_df = df[df["split"] == "val"].drop(columns=["_rand", "_rank", "_n", "split"])

    # Ensure validation set only contains books from training set
    train_books = set(train_df["book_id"])
    val_df = val_df[val_df["book_id"].isin(train_books)]

    logger.info(f"Created train/val split: {len(train_df)} train, {len(val_df)} val ratings")

    return train_df, val_df


def build_interaction_matrix(df: pd.DataFrame, n_users: int, n_cf_books: int):
    """
    Build sparse interaction matrix from DataFrame.
    Uses user_idx and book_idx to map confidences to correct user-book index combinations.
    """
    row = df["user_idx"].values
    col = df["book_idx"].values
    data = df["confidence"].values

    matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_cf_books), dtype=np.float32)
    return matrix


def find_best_model_params(train_matrix, val_matrix):
    """Grid search for best ALS hyperparameters on train/val split."""
    if not sp.issparse(train_matrix) or not sp.issparse(val_matrix):
        raise ValueError("Matrices must be sparse CSR")

    logger.info(f"Training matrix shape (user x book): {train_matrix.shape}")
    logger.info(f"Validation matrix shape: {val_matrix.shape}")

    # params = { "random_state": [42], "factors": [64, 80, 96, 112, 128], "regularization": [0.05, 0.1, 0.15, 0.2], "iterations": [15], "alpha": [40, 60, 80] }

    # ✓ Best Configuration: {'alpha': 80, 'factors': 128, 'iterations': 15, 'random_state': 42, 'regularization': 0.2} with MAP@10: 0.3823
    params = {"random_state": [42], "factors": [128], "regularization": [0.2], "iterations": [15], "alpha": [80]}

    grid = list(ParameterGrid(params))
    num_configs = len(grid)
    logger.info(f"Grid search: {num_configs} configurations to test")

    train_matrix = train_matrix.tocsr()
    val_matrix = val_matrix.tocsr()

    best_map = -np.inf
    best_params = {}
    for idx, g in enumerate(grid):
        logger.info(f"[{idx+1}/{num_configs}] Testing: {g}")
        # Train model with current hyperparams
        model = AlternatingLeastSquares(**g)
        model.fit(train_matrix, show_progress=False)

        # Evaluate using Mean Average Precision (MAP)
        metric = mean_average_precision_at_k(model, train_matrix, val_matrix, K=10)
        logger.info(f"  → MAP@10: {metric:.4f}")

        if metric > best_map:
            best_map = metric
            best_params = g

    logger.info(f"✓ Best Configuration: {best_params} with MAP@10: {best_map:.4f}")

    return best_params


def cv_evaluation(fold_idx, context, book_cf_to_catalog_id, train_matrix, val_matrix):
    """Evaluate one CV fold, grid-searching candidate_pool_size and lambda_weight"""

    k_eval = CROSS_VALIDATION.get("top_k", 10)
    results = {}

    for cps in CROSS_VALIDATION["candidate_pool_size_values"]:
        results[cps] = {}
        for lambda_w in CROSS_VALIDATION["lambda_values"]:
            logger.info(f"Cross validation on candidate_pool_size: {cps} and lambda: {lambda_w}")
            metrics = evaluate(context, book_cf_to_catalog_id, train_matrix, val_matrix, cps, lambda_w, k_eval)
            agg = compute_aggregate_metrics(metrics)
            results[cps][lambda_w] = agg

            mean_map = agg["ap@k"]["mean"]
            std_map = agg["ap@k"]["std"]
            count = agg["ap@k"]["count"]
            logger.info(f"[CPS={cps}, Lambda={lambda_w}] → MAP@{k_eval}: {mean_map:.4f} ± {std_map:.4f} (n={count}) ")

    # Find best (cps, lambda_w) by MAP@K
    best_cps = max(
        results, key=lambda cps: max((results[cps][lw]["ap@k"]["mean"] for lw in results[cps]), default=-np.inf)
    )
    best_lambda = max(results[best_cps], key=lambda lw: results[best_cps][lw]["ap@k"]["mean"])
    best_map = results[best_cps][best_lambda]["ap@k"]["mean"]

    logger.info(f"Fold {fold_idx}: Best MAP@{k_eval} = {best_map:.4f} (cps={best_cps}, lambda={best_lambda})")

    return results


def evaluate(context, book_cf_to_catalog_id, train_matrix, eval_matrix, candidate_pool_size, lambda_weight, k=10):
    interactions_per_user = eval_matrix.getnnz(axis=1)
    users_with_eval = np.where(interactions_per_user > EVALUATION["min_validation_items"])[0]
    logger.info(f"Number of users to evaluate: {len(users_with_eval)}")

    ap_scores = []

    for user_cf in users_with_eval:
        row = eval_matrix[user_cf]
        cf_indices = row.indices

        # Determine if warm user (has training history)
        is_warm = train_matrix[user_cf].nnz > 0

        # True positive items in validation set
        true_catalog_indices = set(book_cf_to_catalog_id[cf_idx] for cf_idx in cf_indices)

        # Get predictions using hybrid recommender
        pred_catalog_indices, _ = get_recommendations(
            context, user_cf, candidate_pool_size, lambda_weight, is_warm_user=is_warm, top_k=k
        )

        # Compute actual_k because reults will be askew if returned recs are less than k
        actual_k = len(pred_catalog_indices)
        ap_k = ap_at_k(pred_catalog_indices, true_catalog_indices, actual_k)
        ap_scores.append(ap_k)

    return {
        "ap@k": ap_scores,
    }


# region HELPERS
def build_context(user_factors, book_factors, train_matrix, catalog_embeddings, book_cf_to_catalog_id, cf_to_book_id):
    return {
        "user_factors": user_factors,
        "book_factors": book_factors,
        "train_matrix": train_matrix,
        "catalog_embeddings": catalog_embeddings,
        "index_mappings": {
            "book_cf_to_catalog_id": book_cf_to_catalog_id,
            "user_id_to_cf": {},
            "cf_to_user_id": {},
            "book_cf_to_id": {book_id: cf_id for cf_id, book_id in cf_to_book_id.items()},
            "cf_to_book_id": cf_to_book_id,
            "book_id_to_catalog_id": {},
        },
        "catalog_df": None,
    }


# endregion
