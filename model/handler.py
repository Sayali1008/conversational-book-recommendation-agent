import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from common import utils
from common.constants import *
from model.model_pipeline import *

logger = setup_logging(__name__, PATHS["eval_log_file"])

def run_model_pipeline():
    # Check if model artifacts already exist to skip training if not needed
    if os.path.exists(PATHS["best_rec_params"]):
        return logger.info("Model already exists. Skipping training pipeline.")

    for dir_path in [MODEL_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting model training...")

    # Load and preprocess
    ratings_df = utils.safe_read_feather(PATHS["clean_ratings"])
    ratings_df = filter_min_max_interactions(ratings_df)

    # Create index mappings
    unique_users = ratings_df["user_id"].unique()
    unique_books = ratings_df["book_id"].unique()
    n_users = len(unique_users)
    n_cf_books = len(unique_books)

    user_id_to_cf = {u: i for i, u in enumerate(unique_users)}
    book_id_to_cf = {b: i for i, b in enumerate(unique_books)}
    utils.save_pickle(user_id_to_cf, PATHS["user_idx_pkl"])
    utils.save_pickle(book_id_to_cf, PATHS["book_idx_pkl"])

    ratings_df["user_idx"] = ratings_df["user_id"].map(user_id_to_cf)
    ratings_df["book_idx"] = ratings_df["book_id"].map(book_id_to_cf)

    # Create a single 80-20 train/val split for hyperparameter grid search
    # This is separate from CV folds which are used for final evaluation
    hp_train_df, hp_val_df = create_train_val_split(ratings_df, seed=CROSS_VALIDATION["seed"])

    # Find best model parameters on this clean split
    hp_train_matrix = build_interaction_matrix(hp_train_df, n_users, n_cf_books)
    hp_val_matrix = build_interaction_matrix(hp_val_df, n_users, n_cf_books)

    sp.save_npz(PATHS["train_matrix"], hp_train_matrix)
    sp.save_npz(PATHS["val_matrix"], hp_val_matrix)
    
    if os.path.exists(PATHS["als_model"] and PATHS["best_model_params"]):
        logger.info(f"Loading existing model and parameters...")
        best_params = utils.load_pickle(PATHS["best_model_params"])
        als_model = utils.load_pickle(PATHS["als_model"])
    else:
        logger.info("Existing model not found. Running hyperparameter grid search...")
        best_params = find_best_model_params(hp_train_matrix, hp_val_matrix)

        als_model = AlternatingLeastSquares(**best_params)
        als_model.fit(hp_train_matrix, show_progress=True)

        # Save final model artifacts for production
        utils.save_pickle(als_model, PATHS["als_model"])
        np.save(PATHS["user_factors"], als_model.user_factors)
        np.save(PATHS["book_factors"], als_model.item_factors)

    if os.path.exists(PATHS["best_rec_params"]):
        logger.info("Skipping CV evaluation since best recommendation parameters already exist.")
        return []

    # Build CV folds on FULL dataset for final evaluation
    folds = build_cv_folds(ratings_df, num_folds=CROSS_VALIDATION["num_folds"], seed=CROSS_VALIDATION["seed"])

    catalog_embeddings = np.load(PATHS["catalog_books_embeddings"])
    _, cf_to_book_id = utils.load_index_mappings(PATHS["book_idx_pkl"])
    book_cf_to_catalog_id = utils.map_book_cf_to_catalog_id(cf_to_book_id)

    all_fold_results = []
    for fold_idx, (train_df, val_df) in enumerate(folds):
        logger.info(f"-" * 80)
        logger.info(f"Fold {fold_idx + 1}/{CROSS_VALIDATION['num_folds']}")
        logger.info(f"-" * 80)

        fold_train_matrix = build_interaction_matrix(train_df, n_users, n_cf_books)
        fold_val_matrix = build_interaction_matrix(val_df, n_users, n_cf_books)

        model = AlternatingLeastSquares(**best_params)
        model.fit(fold_train_matrix, show_progress=False)

        user_factors = model.user_factors
        book_factors = model.item_factors

        context = build_context(
            user_factors, book_factors, fold_train_matrix, catalog_embeddings, book_cf_to_catalog_id, cf_to_book_id
        )

        fold_results = cv_evaluation(fold_idx, context, book_cf_to_catalog_id, fold_train_matrix, fold_val_matrix)
        all_fold_results.append(fold_results)

    # Aggregate across folds
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    _aggregate_cv_results(all_fold_results)

    return all_fold_results


# region HELPERS
def _aggregate_cv_results(all_fold_results):
    """Find best (cps, lambda_w) across all folds."""
    best_cps = None
    best_lambda = None
    best_map = -np.inf

    for fold_result in all_fold_results:
        for cps in fold_result:
            for lambda_w in fold_result[cps]:
                map_score = fold_result[cps][lambda_w]["ap@k"]["mean"]
                if map_score > best_map:
                    best_map = map_score
                    best_cps = cps
                    best_lambda = lambda_w

    logger.info(f"Best recommendation params: CPS={best_cps}, Lambda={best_lambda}, MAP={best_map:.4f}")

    # Save for production
    best_rec_params = {"candidate_pool_size": best_cps, "lambda": best_lambda}
    utils.save_pickle(best_rec_params, PATHS["best_rec_params"])

    return best_rec_params


# endregion
