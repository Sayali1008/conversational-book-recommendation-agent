import pickle
import numpy as np
from common.constants import *


def log_interaction_summary(logger, n_users, n_cf_books, train_matrix, val_matrix, test_matrix):
    logger.info("=== Train Matrix ===")
    logger.info("Shape: %s", train_matrix.shape)
    logger.info("Non-zero entries: %s", f"{train_matrix.nnz:,}")
    logger.info("Sparsity: %.4f%%", 100 * (1 - train_matrix.nnz / (n_users * n_cf_books)))
    logger.info("Density: %.4f%%", 100 * train_matrix.nnz / (n_users * n_cf_books))

    logger.info("=== Val Matrix ===")
    logger.info("Shape: %s", val_matrix.shape)
    logger.info("Non-zero entries: %s", f"{val_matrix.nnz:,}")
    logger.info("Sparsity: %.4f%%", 100 * (1 - val_matrix.nnz / (n_users * n_cf_books)))
    logger.info("Density: %.4f%%", 100 * val_matrix.nnz / (n_users * n_cf_books))

    logger.info("=== Test Matrix ===")
    logger.info("Shape: %s", test_matrix.shape)
    logger.info("Non-zero entries: %s", f"{test_matrix.nnz:,}")
    logger.info("Sparsity: %.4f%%", 100 * (1 - test_matrix.nnz / (n_users * n_cf_books)))
    logger.info("Density: %.4f%%", 100 * test_matrix.nnz / (n_users * n_cf_books))

    train_values = train_matrix.data
    logger.info("=== Confidence Distribution in Train Matrix ===")
    logger.info(f"Min confidence: {train_values.min()}")
    logger.info(f"Max confidence: {train_values.max()}")
    logger.info(f"Mean confidence: {train_values.mean():.2f}")
    unique, counts = np.unique(train_values, return_counts=True)
    for val, count in zip(unique, counts):
        logger.info(f"  {val}: {count:,} ({100*count/len(train_values):.1f}%)")

    user_interactions = np.array(train_matrix.sum(axis=1)).flatten()
    users_with_1_interaction = (user_interactions == 1).sum()
    logger.info("=== User Interaction Distribution (Train) ===")
    logger.info(f"Min interactions per user: {user_interactions.min()}")
    logger.info(
        f"Users with only 1 interaction: {users_with_1_interaction:,} ({100*users_with_1_interaction/n_users:.1f}%)"
    )
    logger.info(f"Max interactions per user: {user_interactions.max()}")
    logger.info(f"Mean interactions per user: {user_interactions.mean():.2f}")
    logger.info(f"Median interactions per user: {np.median(user_interactions):.2f}")

    book_interactions = np.array(train_matrix.sum(axis=0)).flatten()
    books_with_0_interactions = (book_interactions == 0).sum()
    logger.info("=== Book Interaction Distribution (Train) ===")
    logger.info(f"Min interactions per book: {book_interactions.min()}")
    logger.info(
        f"Books with 0 interactions: {books_with_0_interactions:,} ({100*books_with_0_interactions/n_cf_books:.1f}%)"
    )
    logger.info(f"Max interactions per book: {book_interactions.max()}")
    logger.info(f"Mean interactions per book: {book_interactions.mean():.2f}")
    logger.info(f"Median interactions per book: {np.median(book_interactions):.2f}")


def log_user_coverage(logger, ratings_df, train_df, val_df, test_df):
    logger.info("Train interactions: %s", f"{len(train_df):,}")
    logger.info("Val interactions: %s", f"{len(val_df):,}")
    logger.info("Test interactions: %s", f"{len(test_df):,}")
    logger.info(
        "Split ratio: %.1f%% / %.1f%% / %.1f%%",
        100 * len(train_df) / len(ratings_df),
        100 * len(val_df) / len(ratings_df),
        100 * len(test_df) / len(ratings_df),
    )

    train_users = set(train_df["user_id"])
    all_users = set(ratings_df["user_id"])
    coverage = len(train_users) / len(all_users)
    logger.info(f"Number of users in train: {len(train_users)}")
    logger.info(f"Total number of users: {len(all_users)}")
    logger.info(f"User coverage in train: {coverage:.1%}")


def log_model_statistics(logger, user_factors, book_factors):
    logger.info("Model and factors saved")
    logger.info("=== Learned Factors ===")
    logger.info(f"User factors shape: {user_factors.shape}")
    logger.info(f"Book factors shape: {book_factors.shape}")
    logger.info("Sample user vector (cf_idx=0):")
    logger.info(user_factors[0])
    logger.info("Sample book vector (cf_idx=0):")
    logger.info(book_factors[0])
    logger.info(
        f"User factors mean/std/min/max: {user_factors.mean():.4f} / {user_factors.std():.4f} / {user_factors.min():.4f} / {user_factors.max():.4f}"
    )
    logger.info(
        f"Book factors mean/std/min/max: {book_factors.mean():.4f} / {book_factors.std():.4f} / {book_factors.min():.4f} / {book_factors.max():.4f}"
    )


def log_split_summary(logger, results, k, lambda_values, split_name):
    logger.info(f"📈 Results for K={k} ({split_name.capitalize()} Set):")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<20} {'Precision@K':>12} {'Recall@K':>12} {'MAP@K':>12} {'NDCG@K':>12}")
    logger.info("-" * 80)

    for lambda_w in lambda_values:
        strategy_name = _get_strategy_name(lambda_w)
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
        logger.info(f"🏆 Best {metric}@{k}: {_get_strategy_name(lambda_w)}")


def log_recommendation_quality(logger, train_matrix, val_matrix, test_matrix):
    logger.info("=" * 80)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("=" * 80)

    logger.info("[01] Train/Val Distribution:")
    train_density = train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])
    val_density = val_matrix.nnz / (val_matrix.shape[0] * val_matrix.shape[1])
    test_density = test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1])
    logger.info(f"Train density: {train_density:.6f} ({train_density*100:.4f}%)")
    logger.info(f"Val density:   {val_density:.6f} ({val_density*100:.4f}%)")
    logger.info(f"Test density:  {test_density:.6f} ({test_density*100:.4f}%)")
    logger.info(f"Ratio (Val/Train): {val_density/train_density:.2f}")
    if val_density > train_density * 0.5:
        logger.warning("⚠️  Validation set is too large - model hasn't seen enough training data!")

    logger.info("[02] User Validation Coverage:")
    val_per_user = np.array(val_matrix.sum(axis=1)).flatten()
    users_0_val = (val_per_user == 0).sum()
    users_1_val = (val_per_user == 1).sum()
    users_ge5_val = (val_per_user >= 5).sum()
    logger.info(f"Users with 0 validation items: {users_0_val} ({100*users_0_val/len(val_per_user):.1f}%)")
    logger.info(f"Users with 1 validation item:  {users_1_val} ({100*users_1_val/len(val_per_user):.1f}%)")
    logger.info(f"Users with ≥5 validation items: {users_ge5_val} ({100*users_ge5_val/len(val_per_user):.1f}%)")
    if users_0_val > 0.3 * len(val_per_user):
        logger.warning("⚠️  >30% of users have NO validation items - they're being evaluated unfairly!")

    logger.info("[03] User Training Sparsity:")
    train_per_user = np.array(train_matrix.sum(axis=1)).flatten()
    users_single_train = (train_per_user == 1).sum()
    users_few_train = (train_per_user < 5).sum()
    logger.info(f"Users with 1 training item: {users_single_train} ({100*users_single_train/len(train_per_user):.1f}%)")
    logger.info(f"Users with <5 training items: {users_few_train} ({100*users_few_train/len(train_per_user):.1f}%)")
    if users_few_train > 0.5 * len(train_per_user):
        logger.warning("⚠️  >50% of users have <5 training items - cold-start is a major problem!")

    logger.info("[04] Recommendations:")
    if train_density < 0.0001:
        logger.warning("🔴 Data is extremely sparse - consider:")
        logger.warning(
            "   - Increasing MIN_USER_INTERACTIONS (currently {})".format(INTERACTION_MATRIX["min_user_interactions"])
        )
        logger.warning(
            "   - Increasing MIN_BOOK_INTERACTIONS (currently {})".format(INTERACTION_MATRIX["min_book_interactions"])
        )
    if users_few_train > 0.4 * len(train_per_user):
        logger.warning("🔴 Too many cold-start users - adjust filtering thresholds")


def log_evaluation_summary(logger, results):
    logger.info("=" * 80)
    logger.info("SUMMARY: Strategy Comparison")
    logger.info("=" * 80)
    for k in EVALUATION["k_values"]:
        log_split_summary(logger, results, k, EVALUATION["lambda_values"], split_name="val")
        log_split_summary(logger, results, k, EVALUATION["lambda_values"], split_name="test")
        logger.info(f"📊 Val vs Test Comparison for K={k}:")
        logger.info("-" * 80)
        logger.info(f"{'Strategy':<20} {'Val NDCG':>12} {'Test NDCG':>12} {'Δ NDCG':>12}")
        logger.info("-" * 80)
        for lambda_w in EVALUATION["lambda_values"]:
            strategy_name = _get_strategy_name(lambda_w)
            val_ndcg = results[k][lambda_w]["val"]["ndcg@k"]["mean"]
            test_ndcg = results[k][lambda_w]["test"]["ndcg@k"]["mean"]
            delta = test_ndcg - val_ndcg
            status = "✓" if abs(delta) < 0.05 else "⚠" if delta < 0 else "⚡"
            logger.info(f"{strategy_name:<20} {val_ndcg:>12.4f} {test_ndcg:>12.4f} {delta:>12.4f} {status}")


def log_hyperparameters(logger):
    logger.info("=" * 80)
    logger.info("EVALUATION HYPERPARAMETERS")
    logger.info("=" * 80)
    logger.info("Data thresholds:")
    logger.info(f"  MIN_USER_INTERACTIONS: {INTERACTION_MATRIX['min_user_interactions']}")
    logger.info(f"  MIN_BOOK_INTERACTIONS: {INTERACTION_MATRIX['min_book_interactions']}")
    logger.info(f"  MAX_USER_INTERACTIONS: {INTERACTION_MATRIX['max_user_interactions']}")
    logger.info("Embedding settings:")
    logger.info(f"  Model: {EMBEDDINGS['embedding_model']}")
    logger.info(f"  Batch size: {EMBEDDINGS['batch_size']}")
    try:
        with open(PATHS["als_model"], "rb") as f:
            als_model = pickle.load(f)
        logger.info("ALS settings:")
        logger.info(f"  Factors: {getattr(als_model, 'factors', 'N/A')}")
        logger.info(f"  Regularization: {getattr(als_model, 'regularization', 'N/A')}")
        logger.info(f"  Iterations: {getattr(als_model, 'iterations', 'N/A')}")
        logger.info(f"  Alpha: {getattr(als_model, 'alpha', 'N/A')}")
        logger.info(f"  Use GPU: {getattr(als_model, 'use_gpu', 'N/A')}")
    except Exception as e:
        logger.warning(f"⚠️ Unable to load ALS model for hyperparameter logging: {e}")
    logger.info("Ranking & normalization:")
    logger.info(f"  K values: {EVALUATION['k_values']}")
    logger.info(f"  Lambda values: {EVALUATION['lambda_values']} (0=Embeddings, 1=CF)")
    logger.info(f"  Candidate pool size: {EVALUATION['candidate_pool_size']} (None=all CF items)")
    logger.info(f"  Filter rated items: {EVALUATION['filter_rated']}")
    if EVALUATION["norm_metadata"]:
        logger.info(f"  Score normalization: {EVALUATION['norm']} | temperature: {EVALUATION['norm_metadata']}")
    else:
        logger.info(f"  Score normalization: {EVALUATION['norm']}")
    logger.info("  Hybrid combination: lambda * CF_norm  (1 - lambda) * Emb_norm")
    logger.info(f"  Recommender type: {EVALUATION['type']}")
    logger.info("Evaluation filters:")
    logger.info(f"  Min validation items per user: {EVALUATION['min_validation_items']}")
    logger.info(f"  Min confidence threshold: {EVALUATION['min_confidence']}")


def _get_strategy_name(lambda_weight):
    if lambda_weight == 0.0:
        return f"Pure Embeddings (λ={lambda_weight})"
    if lambda_weight == 1.0:
        return f"Pure CF (λ={lambda_weight})"
    return f"Hybrid (λ={lambda_weight})"
