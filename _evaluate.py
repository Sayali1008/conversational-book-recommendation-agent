import numpy as np
import scipy.sparse as sp
from _hybrid_recommender import hybrid_recommender


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


def weighted_precision_at_k(recommended_indices, true_indices_with_conf, k):
    """
    Precision weighted by confidence values.
    
    A hit on a 5-star book counts more than a hit on a 1-star book.
    """
    if len(true_indices_with_conf) == 0:
        return np.nan

    top_k = recommended_indices[:k]
    
    # true_indices_with_conf = dict: {book_idx: confidence_value}
    total_confidence = sum(true_indices_with_conf.values())
    hit_confidence = 0
    
    for book_idx in top_k:
        if book_idx in true_indices_with_conf:
            hit_confidence += true_indices_with_conf[book_idx]
    
    return hit_confidence / total_confidence


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


def get_top_k_recommendations(user_idx, user_factors, book_factors, train_matrix, catalog_embeddings, 
                              cf_to_catalog_map, book_cf_idx_to_book_id, k=10, lambda_weight=0.5,):
    """
    Generate top-K recommendations using hybrid approach.

    Args:
        user_idx: User index in CF matrix
        user_factors: (n_users, n_factors) user factor matrix
        book_factors: (n_cf_books, n_factors) item factor matrix
        train_matrix: (n_users, n_cf_books) sparse training matrix
        catalog_embeddings: (n_catalog_books, n_dim) all book embeddings
        cf_to_catalog_map: Dict mapping CF indices to catalog indices
        book_cf_idx_to_book_id: Mapping from CF index to book_id
        k: Number of recommendations
        lambda_weight: Weight for CF score (0=pure embedding, 1=pure CF)

    Returns:
        top_k_catalog_indices: (k,) array of recommended book catalog indices
    """
    print(f"user_idx={user_idx} | K={k} | lambda={lambda_weight}: get_top_k_recommendations()")

    top_k_catalog_indices, _, _ = hybrid_recommender(
        user_idx=user_idx,
        user_factors=user_factors,
        book_factors=book_factors,
        train_matrix=train_matrix,
        catalog_embeddings=catalog_embeddings,
        cf_to_catalog_map=cf_to_catalog_map,
        book_cf_idx_to_book_id=book_cf_idx_to_book_id,
        k=k,
        lambda_weight=lambda_weight,
        filter_rated=True,
    )

    return top_k_catalog_indices


def evaluate_on_split(user_factors, book_factors, train_matrix, eval_matrix, catalog_embeddings,
                      cf_to_catalog_map, book_cf_idx_to_book_id, k=10, lambda_weight=0.5,):
    """
    Evaluate recommender on validation/test set.

    Works for:
    - Pure CF (lambda_weight=1.0)
    - Pure embeddings (lambda_weight=0.0)
    - Hybrid (lambda_weight=0.5 or any value in between)
    """
    # eval_matrix rows = users, columns = books
    # get only users who have interactions available
    users_with_eval = eval_matrix.nonzero()[0]
    n_users_to_evaluate = len(users_with_eval)
    print(f"Number of users to evaluate: {n_users_to_evaluate}")
    print(f"Users to evaluate: {users_with_eval}")

    precision_scores = []
    recall_scores = []
    ap_scores = []
    ndcg_scores = []

    for user_idx in users_with_eval:
        print(f"user_idx={user_idx}")

        # Get ground-truth from evaluation matrix (CF indices)
        eval_cf_indices = set(eval_matrix[user_idx].nonzero()[1])
        
        # Skip users with no evaluation interactions
        if len(eval_cf_indices) == 0:
            print(f"No val interactions found for user_idx={user_idx}. Not generating recommendations for the user.")
            continue

        # Convert CF indices to catalog indices for comparison
        eval_catalog_indices = set(
            cf_to_catalog_map[cf_idx] for cf_idx in eval_cf_indices
        )

        # Get recommendations (returns catalog indices)
        top_k_catalog_indices = get_top_k_recommendations(
            user_idx=user_idx,
            user_factors=user_factors,
            book_factors=book_factors,
            train_matrix=train_matrix,
            catalog_embeddings=catalog_embeddings,
            cf_to_catalog_map=cf_to_catalog_map,
            book_cf_idx_to_book_id=book_cf_idx_to_book_id,
            k=k,
            lambda_weight=lambda_weight,
        )

        # Compute metrics using catalog indices
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


def compute_aggregate_metrics(metric_dict):
    """
    Compute mean and std of per-user metrics, filtering out NaN values.

    Args:
        metric_dict: dict from evaluate_on_split

    Returns:
        dict with aggregated stats: mean, std, count per metric
    """
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


def get_strategy_name(lambda_weight):
    """Get human-readable strategy name based on lambda weight."""
    if lambda_weight == 0.0:
        return "Pure Embeddings"
    elif lambda_weight == 1.0:
        return "Pure CF"
    else:
        return f"Hybrid (λ={lambda_weight})"


def run_evaluation(logger, user_factors, book_factors, train_matrix, val_matrix, test_matrix,
                   catalog_embeddings, cf_to_catalog_map, book_cf_idx_to_book_id, k_values=[10],
                   lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0],):
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
        cf_to_catalog_map: Dict mapping CF indices to catalog indices
        book_cf_idx_to_book_id: Mapping from CF index to book_id
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        lambda_values: List of lambda weights (0.0=pure embedding, 1.0=pure CF)
    """
    # Store results for comparison
    results = {}

    for k in k_values:
        results[k] = {}

        for lambda_w in lambda_values:
            strategy_name = get_strategy_name(lambda_w)

            logger.info("-" * 80)
            logger.info(f"EVALUATION @ K={k} | Strategy: {strategy_name}")
            logger.info("-" * 80)

            # Validation set
            logger.info(f"📊 Validation Set:")
            val_metrics = evaluate_on_split(
                user_factors,
                book_factors,
                train_matrix,
                val_matrix,
                catalog_embeddings,
                cf_to_catalog_map,
                book_cf_idx_to_book_id,
                k=k,
                lambda_weight=lambda_w,
            )
            val_agg = compute_aggregate_metrics(val_metrics)

            for metric_name, stats in val_agg.items():
                logger.info(
                    f"  {metric_name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(n_users={stats['count']})"
                )

            # Test set
            # logger.info(f"\n📊 Test Set:")
            # test_metrics = evaluate_on_split(
            #     user_factors,
            #     book_factors,
            #     train_matrix,
            #     test_matrix,
            #     catalog_embeddings,
            #     cf_to_catalog_map,
            #     book_cf_idx_to_book_id,
            #     k=k,
            #     lambda_weight=lambda_w,
            # )
            # test_agg = compute_aggregate_metrics(test_metrics)

            # for metric_name, stats in test_agg.items():
            #     logger.info(
            #         f"  {metric_name:12s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
            #         f"(n_users={stats['count']})"
            #     )

            # Store for comparison
            results[k][lambda_w] = {
                "val": val_agg,
                # "test": test_agg,
                "strategy": strategy_name,
            }

    # Print summary comparison
    print_evaluation_summary(logger, results, k_values, lambda_values)


def _log_split_summary(logger, results, k, lambda_values, split_name):
    logger.info(f"📈 Results for K={k} ({split_name.capitalize()} Set):")
    logger.info("-" * 80)
    logger.info(
        f"{'Strategy':<20} {'Precision@K':>12} {'Recall@K':>12} "
        f"{'MAP@K':>12} {'NDCG@K':>12}"
    )
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
        "Precision": max(lambda_values, key=lambda lw: results[k][lw][split_name]["precision@k"]["mean"]),
        "Recall": max(lambda_values, key=lambda lw: results[k][lw][split_name]["recall@k"]["mean"]),
        "MAP": max(lambda_values, key=lambda lw: results[k][lw][split_name]["ap@k"]["mean"]),
        "NDCG": max(lambda_values, key=lambda lw: results[k][lw][split_name]["ndcg@k"]["mean"]),
    }

    for metric, lambda_w in best.items():
        logger.info(f"🏆 Best {metric}@{k}: {get_strategy_name(lambda_w)}")


def print_evaluation_summary(logger, results, k_values, lambda_values):
    """Print a comparison table of all strategies for both validation and test sets."""
    logger.info("=" * 80)
    logger.info("SUMMARY: Strategy Comparison")
    logger.info("=" * 80)

    for k in k_values:
        _log_split_summary(logger, results, k, lambda_values, split_name="val")
        # _log_split_summary(logger, results, k, lambda_values, split_name="test")

        # ============================================================
        # CROSS-SET COMPARISON (Val vs Test)
        # ============================================================
        # logger.info(f"\n📊 Val vs Test Comparison for K={k}:")
        # logger.info("-" * 80)
        # logger.info(
        #     f"{'Strategy':<20} {'Val NDCG':>12} {'Test NDCG':>12} {'Δ NDCG':>12}"
        # )
        # logger.info("-" * 80)

        # for lambda_w in lambda_values:
        #     strategy_name = get_strategy_name(lambda_w)
        #     val_ndcg = results[k][lambda_w]["val"]["ndcg@k"]["mean"]
        #     test_ndcg = results[k][lambda_w]["test"]["ndcg@k"]["mean"]
        #     delta = test_ndcg - val_ndcg

        #     status = "✓" if abs(delta) < 0.05 else "⚠" if delta < 0 else "⚡"

        #     logger.info(
        #         f"{strategy_name:<20} "
        #         f"{val_ndcg:>12.4f} "
        #         f"{test_ndcg:>12.4f} "
        #         f"{delta:>+12.4f} {status}"
        #     )
