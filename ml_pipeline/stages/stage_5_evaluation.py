import numpy as np
import scipy.sparse as sp

from common.constants import *
from common.helpers import *
from common.logging import log_evaluation_summary, log_hyperparameters, _get_strategy_name
from common.utils import *
from recommenders.collaborative import _create_user_profile_from_history, get_collaborative_recommendations
from recommenders.content_based import get_content_based_recommendations
from recommenders import collaborative, content_based
from recommenders.data_models import RecommendationContext, RecommendationConfig

logger = setup_logging(__name__, PATHS["eval_log_file"])


def run_evaluation():
    logger.info("Loading artifacts...")
    user_factors = np.load(PATHS["user_factors"])
    book_factors = np.load(PATHS["book_factors"])
    train_matrix = sp.load_npz(PATHS["train_matrix"])
    val_matrix = sp.load_npz(PATHS["val_matrix"])
    test_matrix = sp.load_npz(PATHS["test_matrix"])
    catalog_embeddings = np.load(PATHS["catalog_books_embeddings"])

    logger.info("Loading mappings...")
    _, cf_idx_to_book = load_index_mappings(PATHS["book_idx_pkl"])
    cf_idx_to_catalog_id = map_book_cf_idx_to_catalog_index(cf_idx_to_book)

    log_hyperparameters(logger)

    # Store results for comparison
    results = {}

    for k in EVALUATION["k_values"]:
        results[k] = {}

        for lambda_w in EVALUATION["lambda_values"]:
            strategy_name = _get_strategy_name(lambda_w)
            print(f"Evaluating @ K={k} | Strategy: {strategy_name}")

            logger.info("-" * 80)
            logger.info(f"EVALUATION @ K={k} | Strategy: {strategy_name}")
            logger.info("-" * 80)

            # Validation set
            val_metrics = evaluate(
                user_factors,
                book_factors,
                train_matrix,
                val_matrix,
                catalog_embeddings,
                cf_idx_to_catalog_id,
                cf_idx_to_book,
                # norm=norm,
                # norm_metadata=norm_metadata,
                k=k,
                lambda_weight=lambda_w,
                # min_validation_items=min_validation_items,
                # min_confidence=min_confidence,
                # candidate_pool_size=candidate_pool_size,
                # filter_rated=filter_rated,
                # recommender_type=EVALUATION["type"],
            )
            val_agg = compute_aggregate_metrics(val_metrics)

            logger.info(f"📊 Validation Set:")
            for metric_name, stats in val_agg.items():
                logger.info(
                    f"  {metric_name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} " f"(n_users={stats['count']})"
                )

            # Test set
            test_metrics = evaluate(
                user_factors,
                book_factors,
                train_matrix,
                test_matrix,
                catalog_embeddings,
                cf_idx_to_catalog_id,
                cf_idx_to_book,
                # norm=norm,
                # norm_metadata=norm_metadata,
                k=k,
                lambda_weight=lambda_w,
                # min_validation_items=min_validation_items,
                # min_confidence=min_confidence,
                # candidate_pool_size=candidate_pool_size,
                # filter_rated=filter_rated,
                # recommender_type=EVALUATION["type"],
            )
            test_agg = compute_aggregate_metrics(test_metrics)

            logger.info(f"📊 Test Set:")
            for metric_name, stats in test_agg.items():
                logger.info(
                    f"  {metric_name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} " f"(n_users={stats['count']})"
                )

            # Store for comparison
            results[k][lambda_w] = {
                "val": val_agg,
                "test": test_agg,
                "strategy": strategy_name,
            }

    # Print summary comparison
    log_evaluation_summary(logger, results)


def _build_recommendation_context(
    user_factors,
    book_factors,
    train_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
):
    return {
        "user_factors": user_factors,
        "book_factors": book_factors,
        "train_matrix": train_matrix,
        "catalog_embeddings": catalog_embeddings,
        "index_mappings": {
            "cf_idx_to_catalog_id": cf_idx_to_catalog_id_map,
            "user_to_cf_idx": {},
            "cf_idx_to_user": {},
            "cf_idx_to_book": cf_idx_to_book,
            "book_id_to_catalog_idx": {},
        },
        "catalog_df": None,
    }


def _build_recommendation_config(
    norm,
    norm_metadata,
    lambda_weight,
    k,
    candidate_pool_size,
    filter_rated,
    recency_boost,
):
    """Create RecommendationConfig dictionary expected by recommenders/ functions."""
    return {
        "norm": norm,
        "norm_metadata": norm_metadata,
        "lambda_weight": lambda_weight,
        "k": k,
        "candidate_pool_size": candidate_pool_size,
        "filter_rated": filter_rated,
        "recency_boost": recency_boost,
    }


def evaluate(
    user_factors,
    book_factors,
    train_matrix,
    eval_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
    k,
    lambda_weight,
):
    """
    Evaluate recommender on validation/test set.

    Works for:
    - Collaborative/Content hybrid (recommender_type="CF")
    - Content-based only (recommender_type="CB")
    """

    # Filter rows with confidence < min_confidence
    eval_matrix_filtered = eval_matrix.copy()
    eval_matrix_filtered.data[eval_matrix_filtered.data < EVALUATION["min_confidence"]] = 0
    eval_matrix_filtered.eliminate_zeros()

    # Build list of evaluable users: rows with >= min_validation_items nonzeros
    # This implicitly filters out users with 0 validation items
    interactions_per_user = eval_matrix_filtered.getnnz(axis=1)
    users_with_eval = np.where(interactions_per_user >= EVALUATION["min_validation_items"])[0]
    logger.info(f"Number of users to evaluate: {len(users_with_eval)}")

    precision_scores = []
    recall_scores = []
    ap_scores = []
    ndcg_scores = []

    for user_idx in users_with_eval:
        row = eval_matrix_filtered[user_idx]
        cf_indices = row.indices

        # Skip if no evaluable items remain
        # TODO can this affect metrics?
        if len(cf_indices) == 0:
            continue

        # Convert CF indices to catalog indices for comparison
        eval_catalog_indices = set(cf_idx_to_catalog_id_map[cf_idx] for cf_idx in cf_indices)

        context: RecommendationContext = _build_recommendation_context(
            user_factors, book_factors, train_matrix, catalog_embeddings, cf_idx_to_catalog_id_map, cf_idx_to_book
        )

        config: RecommendationConfig = _build_recommendation_config(
            EVALUATION["norm"],
            EVALUATION.get("norm_metadata"),
            lambda_weight,
            k,
            EVALUATION.get("candidate_pool_size"),
            EVALUATION.get("filter_rated", True),
            RECOMMEND["recency_boost"],
        )

        if EVALUATION["type"] == "CB":
            exclude_indices = (
                _get_rated_catalog_indices(train_matrix, user_idx, cf_idx_to_catalog_id_map)
                if config["filter_rated"]
                else set()
            )
            user_profile = _create_user_profile_from_history(context, user_idx)
            top_k_catalog_indices, _, _ = content_based.get_content_based_recommendations(
                context=context,
                config=config,
                k=k,
                user_profile=user_profile,
                exclude_catalog_rows=exclude_indices,
                # candidate_catalog_indices=np.arange(catalog_embeddings.shape[0]),
            )
        else:  # default to collaborative ("CF")
            # For content-based: build user profile and exclusions
            top_k_catalog_indices, _, _ = collaborative.get_collaborative_recommendations(
                context=context,
                config=config,
                user_idx=user_idx,
                k=k,
            )

        # ===========================
        # Metrics
        # ===========================
        p_k = precision_at_k(top_k_catalog_indices, eval_catalog_indices, k)
        r_k = recall_at_k(top_k_catalog_indices, eval_catalog_indices, k)
        ap_k = ap_at_k(top_k_catalog_indices, eval_catalog_indices, k)
        ndcg_k = ndcg_at_k(top_k_catalog_indices, eval_catalog_indices, k)

        precision_scores.append(p_k)
        recall_scores.append(r_k)
        ap_scores.append(ap_k)
        ndcg_scores.append(ndcg_k)

    return {
        "precision@k": precision_scores,
        "recall@k": recall_scores,
        "ap@k": ap_scores,
        "ndcg@k": ndcg_scores,
    }


def _get_rated_catalog_indices(train_matrix, user_idx, cf_idx_to_catalog_id_map):
    """Return catalog indices the user has already interacted with."""
    row = train_matrix[user_idx].toarray().flatten()
    rated_cf_indices = np.where(row > 0)[0]
    return {cf_idx_to_catalog_id_map[cf_idx] for cf_idx in rated_cf_indices}
