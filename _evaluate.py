import pickle

from matplotlib.pylab import norm
import numpy as np
import scipy.sparse as sp
from sklearn import logger

import _hybrid_recommender
from config import *

# region [HELPERS]


def _log_split_summary(logger, results, k, lambda_values, split_name):
    logger.info(f"📈 Results for K={k} ({split_name.capitalize()} Set):")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<20} {'Precision@K':>12} {'Recall@K':>12} " f"{'MAP@K':>12} {'NDCG@K':>12}")
    logger.info("-" * 80)

    for lambda_w in lambda_values:
        strategy_name = get_strategy_name(lambda_w)
        agg = results[k][lambda_w][split_name]

        logger.info(
            f"{strategy_name:<20} "
            f"{agg['precision@k']['mean']:>12.4f} "
            f"{agg['recall@k']['mean']:>12.4f} "
            f"{agg['ap@k']['mean']:>12.4f} "
            f"{agg['ndcg@k']['mean']:>12.4f}"
        )

    logger.info("-" * 80)

    best = {
        "Precision": max(
            lambda_values,
            key=lambda lw: results[k][lw][split_name]["precision@k"]["mean"],
        ),
        "Recall": max(lambda_values, key=lambda lw: results[k][lw][split_name]["recall@k"]["mean"]),
        "MAP": max(lambda_values, key=lambda lw: results[k][lw][split_name]["ap@k"]["mean"]),
        "NDCG": max(lambda_values, key=lambda lw: results[k][lw][split_name]["ndcg@k"]["mean"]),
    }

    for metric, lambda_w in best.items():
        logger.info(f"🏆 Best {metric}@{k}: {get_strategy_name(lambda_w)}")


def _log_recommendation_quality(logger, train_matrix, val_matrix, test_matrix):
    """
    Diagnose data quality issues that may hurt recommendations.
    """
    logger.info("=" * 80)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("=" * 80)

    # 1. Check train/val distribution
    logger.info("1️⃣ Train/Val Distribution:")
    train_density = train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])
    val_density = val_matrix.nnz / (val_matrix.shape[0] * val_matrix.shape[1])
    test_density = test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1])

    logger.info(f"Train density: {train_density:.6f} ({train_density*100:.4f}%)")
    logger.info(f"Val density:   {val_density:.6f} ({val_density*100:.4f}%)")
    logger.info(f"Test density:  {test_density:.6f} ({test_density*100:.4f}%)")
    logger.info(f"Ratio (Val/Train): {val_density/train_density:.2f}")

    if val_density > train_density * 0.5:
        logger.warning("⚠️  Validation set is too large - model hasn't seen enough training data!")

    # 2. Check users with very few validation items
    logger.info("2️⃣ User Validation Coverage:")
    val_per_user = np.array(val_matrix.sum(axis=1)).flatten()
    users_0_val = (val_per_user == 0).sum()
    users_1_val = (val_per_user == 1).sum()
    users_ge5_val = (val_per_user >= 5).sum()

    logger.info(f"Users with 0 validation items: {users_0_val} ({100*users_0_val/len(val_per_user):.1f}%)")
    logger.info(f"Users with 1 validation item:  {users_1_val} ({100*users_1_val/len(val_per_user):.1f}%)")
    logger.info(f"Users with ≥5 validation items: {users_ge5_val} ({100*users_ge5_val/len(val_per_user):.1f}%)")

    if users_0_val > 0.3 * len(val_per_user):
        logger.warning("⚠️  >30% of users have NO validation items - they're being evaluated unfairly!")

    # 3. Check train sparsity per user
    logger.info("3️⃣ User Training Sparsity:")
    train_per_user = np.array(train_matrix.sum(axis=1)).flatten()
    users_single_train = (train_per_user == 1).sum()
    users_few_train = (train_per_user < 5).sum()

    logger.info(f"Users with 1 training item: {users_single_train} ({100*users_single_train/len(train_per_user):.1f}%)")
    logger.info(f"Users with <5 training items: {users_few_train} ({100*users_few_train/len(train_per_user):.1f}%)")

    if users_few_train > 0.5 * len(train_per_user):
        logger.warning("⚠️  >50% of users have <5 training items - cold-start is a major problem!")

    # 4. Recommend minimum thresholds
    logger.info("4️⃣ Recommendations:")
    if train_density < 0.0001:
        logger.warning("🔴 Data is extremely sparse - consider:")
        logger.warning("   - Increasing MIN_USER_INTERACTIONS (currently {})".format(MIN_USER_INTERACTIONS))
        logger.warning("   - Increasing MIN_BOOK_INTERACTIONS (currently {})".format(MIN_BOOK_INTERACTIONS))

    if users_few_train > 0.4 * len(train_per_user):
        logger.warning("🔴 Too many cold-start users - adjust filtering thresholds")


def _log_evaluation_summary(logger, results, k_values, lambda_values):
    """Log a comparison table of all strategies for both validation and test sets."""
    logger.info("=" * 80)
    logger.info("SUMMARY: Strategy Comparison")
    logger.info("=" * 80)

    for k in k_values:
        _log_split_summary(logger, results, k, lambda_values, split_name="val")
        _log_split_summary(logger, results, k, lambda_values, split_name="test")

        # ============================================================
        # CROSS-SET COMPARISON (Val vs Test)
        # ============================================================
        logger.info(f"📊 Val vs Test Comparison for K={k}:")
        logger.info("-" * 80)
        logger.info(f"{'Strategy':<20} {'Val NDCG':>12} {'Test NDCG':>12} {'Δ NDCG':>12}")
        logger.info("-" * 80)

        for lambda_w in lambda_values:
            strategy_name = get_strategy_name(lambda_w)
            val_ndcg = results[k][lambda_w]["val"]["ndcg@k"]["mean"]
            test_ndcg = results[k][lambda_w]["test"]["ndcg@k"]["mean"]
            delta = test_ndcg - val_ndcg

            status = "✓" if abs(delta) < 0.05 else "⚠" if delta < 0 else "⚡"

            logger.info(
                f"{strategy_name:<20} " f"{val_ndcg:>12.4f} " f"{test_ndcg:>12.4f} " f"{delta:>+12.4f} {status}"
            )


def get_strategy_name(lambda_weight):
    """Get human-readable strategy name based on lambda weight."""
    if lambda_weight == 0.0:
        return "Pure Embeddings"
    elif lambda_weight == 1.0:
        return "Pure CF"
    else:
        return f"Hybrid (λ={lambda_weight})"


def _log_hyperparameters(
    logger,
    k_values,
    lambda_values,
    min_validation_items,
    min_confidence,
    candidate_pool_size,
    filter_rated,
    norm,
    norm_metadata=None,
):
    logger.info("=" * 80)
    logger.info("EVALUATION HYPERPARAMETERS")
    logger.info("=" * 80)

    # Data filtering thresholds used upstream
    logger.info("Data thresholds:")
    logger.info(f"  MIN_USER_INTERACTIONS: {MIN_USER_INTERACTIONS}")
    logger.info(f"  MIN_BOOK_INTERACTIONS: {MIN_BOOK_INTERACTIONS}")
    logger.info(f"  MAX_USER_INTERACTIONS: {MAX_USER_INTERACTIONS}")

    # Embedding settings
    logger.info("Embedding settings:")
    logger.info(f"  Model: {EMBEDDING_MODEL}")
    logger.info(f"  Batch size: {BATCH_SIZE}")

    # ALS model hyperparameters (re-log in evaluation)
    try:
        with open(OUTPUT_ALS_MODEL, "rb") as f:
            als_model = pickle.load(f)
        logger.info("ALS settings:")
        logger.info(f"  Factors: {getattr(als_model, 'factors', 'N/A')}")
        logger.info(f"  Regularization: {getattr(als_model, 'regularization', 'N/A')}")
        logger.info(f"  Iterations: {getattr(als_model, 'iterations', 'N/A')}")
        logger.info(f"  Alpha: {getattr(als_model, 'alpha', 'N/A')}")
        logger.info(f"  Use GPU: {getattr(als_model, 'use_gpu', 'N/A')}")
    except Exception as e:
        logger.warning(f"⚠️ Unable to load ALS model for hyperparameter logging: {e}")

    # Ranking and normalization
    logger.info("Ranking & normalization:")
    logger.info(f"  K values: {k_values}")
    logger.info(f"  Lambda values: {lambda_values} (0=Embeddings, 1=CF)")
    logger.info(f"  Candidate pool size: {candidate_pool_size} (None=all CF items)")
    logger.info(f"  Filter rated items: {filter_rated}")
    logger.info(f"  Score normalization: {norm}{f' temperature: {norm_metadata}' if norm_metadata else ''}")
    logger.info("  Hybrid combination: lambda * CF_norm + (1 - lambda) * Emb_norm")

    # Evaluation filters
    logger.info("Evaluation filters:")
    logger.info(f"  Min validation items per user: {min_validation_items}")
    logger.info(f"  Min confidence threshold: {min_confidence}")


# endregion

# region [METRICS]


def precision_at_k(recommended_indices, true_indices, k):
    """
    Compute Precision@K for a single user.
    Interpretation: Of the K items we recommended, how many did the user actually like?
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / k


def recall_at_k(recommended_indices, true_indices, k):
    """
    Compute Recall@K for a single user.
    Interpretation: Of all the books User liked, we found N of them in our top-K recommendations.
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    hits = len(set(top_k) & true_indices)
    return hits / len(true_indices)


def ap_at_k(recommended_indices, true_indices, k):
    """
    Compute Average Precision@K (AP@K) for a single user.
    Key insight: AP@K heavily penalizes missing relevant items at the top of the ranking.
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    score = 0.0
    num_hits = 0

    for i, idx in enumerate(top_k):
        if idx in true_indices:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / min(k, len(true_indices))


def ndcg_at_k(recommended_indices, true_indices, k):
    """
    Compute Normalized Discounted Cumulative Gain@K (NDCG@K) for a single user.
    Binary relevance: true_indices are marked as relevant (1), others as irrelevant (0).
    """
    if len(true_indices) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    dcg = 0.0

    for i, idx in enumerate(top_k):
        if idx in true_indices:
            dcg += 1.0 / np.log2(i + 2)

    # Compute ideal DCG: best possible ranking (all relevant books first)
    ideal_dcg = 0.0
    for i in range(min(k, len(true_indices))):
        ideal_dcg += 1.0 / np.log2(i + 2)

    if ideal_dcg == 0:
        return np.nan

    return dcg / ideal_dcg


def compute_aggregate_metrics(metric_dict):
    """Compute mean and std of per-user metrics, filtering out NaN values."""
    aggregated = {}

    for metric_name, scores in metric_dict.items():
        scores_arr = np.array(scores)
        valid_scores = scores_arr[~np.isnan(scores_arr)]

        if len(valid_scores) > 0:
            aggregated[metric_name] = {
                "mean": valid_scores.mean(),
                "std": valid_scores.std(),
                "count": len(valid_scores),
            }
        else:
            aggregated[metric_name] = {
                "mean": np.nan,
                "std": np.nan,
                "count": 0,
            }

    return aggregated


# endregion


def evaluate_on_split(
    user_factors,
    book_factors,
    train_matrix,
    eval_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
    norm="minmax",
    norm_metadata=None,
    k=10,
    lambda_weight=0.5,
    min_validation_items=3,
    min_confidence=2,
    candidate_pool_size=None,
    filter_rated=True,
):
    """
    Evaluate recommender on validation/test set.

    Works for:
    - Pure CF (lambda_weight=1.0)
    - Pure embeddings (lambda_weight=0.0)
    - Hybrid (lambda_weight=0.5 or any value in between)
    """

    # Filter rows with confidence < min_confidence
    eval_matrix_filtered = eval_matrix.copy()
    eval_matrix_filtered.data[eval_matrix_filtered.data < min_confidence] = 0
    eval_matrix_filtered.eliminate_zeros()

    # Build list of evaluable users: rows with >= min_validation_items nonzeros
    # This implicitly filters out users with 0 validation items
    interactions_per_user = eval_matrix_filtered.getnnz(axis=1)
    users_with_eval = np.where(interactions_per_user >= min_validation_items)[0]
    print(f"Number of users to evaluate: {len(users_with_eval)}")

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

        # ===========================
        # Ranking
        # ===========================
        top_k_catalog_indices, _, _ = _hybrid_recommender.hybrid_recommender(
            user_idx,
            user_factors,
            book_factors,
            train_matrix,
            catalog_embeddings,
            cf_idx_to_catalog_id_map,
            cf_idx_to_book,
            norm,
            norm_metadata,
            k,
            lambda_weight,
            candidate_pool_size,
            filter_rated,
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


def run_evaluation(
    logger,
    user_factors,
    book_factors,
    train_matrix,
    val_matrix,
    test_matrix,
    catalog_embeddings,
    cf_idx_to_catalog_id_map,
    cf_idx_to_book,
    norm="minmax",
    norm_metadata=None,
    k_values=[10],
    lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0],
    min_validation_items=3,
    min_confidence=2,
    candidate_pool_size=200,
    filter_rated=True,
):
    """
    Complete evaluation pipeline: evaluate all strategies on validation and test sets.

    Evaluates:
    - Pure CF (lambda=1.0)
    - Pure Embeddings (lambda=0.0)
    - Hybrid approaches (lambda=0.3, 0.5, 0.7, etc.)

    Args:
        logger: Logging object
        user_factors: (n_users, n_factors) user factor matrix
        book_factors: (n_cf_books, n_factors) item factor matrix
        train_matrix: (n_users, n_cf_books) sparse training matrix
        val_matrix: (n_users, n_cf_books) sparse validation matrix
        test_matrix: (n_users, n_cf_books) sparse test matrix
        catalog_embeddings: (n_catalog_books, n_dim) all book embeddings
        cf_idx_to_catalog_id_map: Dict mapping CF indices to catalog indices
        cf_idx_to_book: Mapping from CF index to book_id
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        lambda_values: List of lambda weights (0.0=pure embedding, 1.0=pure CF)
    """
    _log_hyperparameters(
        logger,
        k_values=k_values,
        lambda_values=lambda_values,
        min_validation_items=min_validation_items,
        min_confidence=min_confidence,
        candidate_pool_size=candidate_pool_size,
        filter_rated=filter_rated,
        norm=norm,
        norm_metadata=norm_metadata,
    )

    # Store results for comparison
    results = {}

    for k in k_values:
        results[k] = {}

        for lambda_w in lambda_values:
            strategy_name = get_strategy_name(lambda_w)
            print(f"Evaluating @ K={k} | Strategy: {strategy_name}")

            logger.info("-" * 80)
            logger.info(f"EVALUATION @ K={k} | Strategy: {strategy_name}")
            logger.info("-" * 80)

            # Validation set
            val_metrics = evaluate_on_split(
                user_factors,
                book_factors,
                train_matrix,
                val_matrix,
                catalog_embeddings,
                cf_idx_to_catalog_id_map,
                cf_idx_to_book,
                norm=norm,
                norm_metadata=norm_metadata,
                k=k,
                lambda_weight=lambda_w,
                min_validation_items=min_validation_items,
                min_confidence=min_confidence,
                candidate_pool_size=candidate_pool_size,
                filter_rated=filter_rated,
            )
            val_agg = compute_aggregate_metrics(val_metrics)

            logger.info(f"📊 Validation Set:")
            for metric_name, stats in val_agg.items():
                logger.info(
                    f"  {metric_name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} " f"(n_users={stats['count']})"
                )

            # Test set
            test_metrics = evaluate_on_split(
                user_factors,
                book_factors,
                train_matrix,
                test_matrix,
                catalog_embeddings,
                cf_idx_to_catalog_id_map,
                cf_idx_to_book,
                norm=norm,
                norm_metadata=norm_metadata,
                k=k,
                lambda_weight=lambda_w,
                min_validation_items=min_validation_items,
                min_confidence=min_confidence,
                candidate_pool_size=candidate_pool_size,
                filter_rated=filter_rated,
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
    _log_evaluation_summary(logger, results, k_values, lambda_values)
