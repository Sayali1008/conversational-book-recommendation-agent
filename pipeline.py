import ast
import os
import pickle
import sys
import time

import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

import _create_embeddings
import _data_preprocessing
import _hybrid_recommender
import _evaluate
import _train_cf
from config import *
from utils import *


def run_data_preprocessing(logger):
    books_df = safe_read_csv(INPUT_BOOKS, INPUT_COLS_BOOKS)
    catalog_books_df = _data_preprocessing.clean_books_data(logger, books_df)
    logger.info(f"✓ Books data cleaned: shape={catalog_books_df.shape}")

    ratings_df = safe_read_csv(INPUT_RATINGS, INPUT_COLS_RATINGS)
    ratings_df = _data_preprocessing.clean_ratings_data(logger, ratings_df, catalog_books_df)
    logger.info(f"✓ Ratings data cleaned: shape={ratings_df.shape}")

    catalog_books_df[OUTPUT_COLS_BOOKS].to_feather(OUTPUT_BOOKS)
    ratings_df[OUTPUT_COLS_RATINGS].to_feather(OUTPUT_RATINGS)


def run_create_embeddings(logger):
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    catalog_books_df = safe_read_feather(OUTPUT_BOOKS)
    embeddings, index = _create_embeddings.generate_embeddings(logger, catalog_books_df, embeddings_model, BATCH_SIZE)

    # Example output: (n_catalog_books, 384) for all-MiniLM-L6-v2
    # Example output: (n_catalog_books, 768) for all-mpnet-base-v2


def run_build_interaction_matrix(logger):
    n_users, n_cf_books, train_df, val_df, test_df = _train_cf.prepare_data(logger)

    # Build matrices for train/val/test
    train_matrix = _train_cf.build_interaction_matrix(train_df, n_users, n_cf_books, binary=True)
    val_matrix = _train_cf.build_interaction_matrix(val_df, n_users, n_cf_books, binary=True)
    test_matrix = _train_cf.build_interaction_matrix(test_df, n_users, n_cf_books, binary=True)

    _train_cf._log_interaction_summary(logger, n_users, n_cf_books, train_matrix, val_matrix, test_matrix)

    sp.save_npz(OUTPUT_TRAIN_MATRIX, train_matrix)
    sp.save_npz(OUTPUT_VAL_MATRIX, val_matrix)
    sp.save_npz(OUTPUT_TEST_MATRIX, test_matrix)

    logger.info("Matrices saved successfully")


def run_train_cf_model(logger):
    # train_matrix = (user x book) sparse matrix
    train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)

    cf_model = _train_cf.model_initialization(
        logger,
        factors=64,
        regularization=0.10,
        iterations=20,
        alpha=20,
    )
    cf_model = _train_cf.model_training(logger, cf_model, train_matrix)

    # Extract learned factors
    book_factors = cf_model.item_factors  # Shape: (n_cf_books, n_factors)
    user_factors = cf_model.user_factors  # Shape: (n_users, n_factors)

    # Save model and factors
    save_pickle(cf_model, OUTPUT_ALS_MODEL)
    np.save(OUTPUT_USER_FACTORS, user_factors)
    np.save(OUTPUT_BOOK_FACTORS, book_factors)
    _train_cf._log_model_statistics(logger, user_factors, book_factors)


def run_evaluation(logger):
    """
    Evaluate all recommendation strategies:
    - Pure CF (lambda=1.0)
    - Pure Embeddings (lambda=0.0)
    - Hybrid (various lambda values)
    """

    user_factors = np.load(OUTPUT_USER_FACTORS)
    book_factors = np.load(OUTPUT_BOOK_FACTORS)
    train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)
    val_matrix = sp.load_npz(OUTPUT_VAL_MATRIX)
    test_matrix = sp.load_npz(OUTPUT_TEST_MATRIX)
    catalog_embeddings = np.load(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)

    _, cf_idx_to_book = load_index_mappings(BOOK_IDX_PKL)
    cf_idx_to_catalog_id_map = map_book_cf_idx_to_catalog_index(cf_idx_to_book)

    # Analyze data quality issues
    logger.info("Loading model artifacts for evaluation...")
    logger.info(f"Loaded factors: user={user_factors.shape}, book={book_factors.shape}")
    logger.info(f"Loaded embeddings: {catalog_embeddings.shape}")
    logger.info(f"Loaded matrices: train={train_matrix.shape}, val={val_matrix.shape}, test={test_matrix.shape}")
    _evaluate._log_recommendation_quality(logger, train_matrix, val_matrix, test_matrix)

    # Evaluate multiple K values and lambda strategies
    # 0=pure embedding, 1=pure CF

    _evaluate.run_evaluation(
        logger=logger,
        user_factors=user_factors,
        book_factors=book_factors,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        test_matrix=test_matrix,
        catalog_embeddings=catalog_embeddings,
        cf_idx_to_catalog_id_map=cf_idx_to_catalog_id_map,
        cf_idx_to_book=cf_idx_to_book,
        norm=EVAL_NORM,
        norm_metadata=EVAL_NORM_METADATA,
        k_values=EVAL_K_VALUES,
        lambda_values=EVAL_LAMBDA_VALUES,
        min_validation_items=EVAL_MIN_VALIDATION_ITEMS,
        min_confidence=EVAL_MIN_CONFIDENCE,
        candidate_pool_size=EVAL_CANDIDATE_POOL_SIZE,
        filter_rated=EVAL_FILTER_RATED,
    )


def run_hybrid_recommender(logger):
    """Test hybrid recommendations for sample users"""

    start_all = time.time()
    logger.info("Starting hybrid recommender smoke-test")

    # Load everything
    user_factors = np.load(OUTPUT_USER_FACTORS)
    book_factors = np.load(OUTPUT_BOOK_FACTORS)
    train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)
    catalog_embeddings = np.load(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)
    catalog_df = safe_read_feather(OUTPUT_BOOKS)
    ratings_df = safe_read_feather(OUTPUT_RATINGS)

    # Load book and user index mappings
    user_to_cf_idx, cf_idx_to_user = load_index_mappings(USER_IDX_PKL)
    book_to_cf_idx, cf_idx_to_book = load_index_mappings(BOOK_IDX_PKL)

    # Build CF → catalog mapping
    book_cf_idx_to_catalog_map = map_book_cf_idx_to_catalog_index(cf_idx_to_book)

    # Test different users and lambda values
    # lambda_weight = 0 is pure embedding, 1 is pure CF
    lambda_values = [0.5]
    test_users = [9393, 21756, 4570, 8260, 25932, 3818, 17023, 20445, 118, 18286]

    for user_idx in test_users:
        # Get the actual user_id from the index mapping
        user_id = cf_idx_to_user.get(user_idx)

        logger.info(f"{'#' * REPEATS}")
        logger.info(f"USER {user_idx}")
        logger.info(f"{'#' * REPEATS}")

        # Analyze user's rated books and their genres
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]

        if len(user_ratings) == 0:
            logger.warning(f"No ratings found for user {user_id}")
        else:
            rated_book_ids = user_ratings["book_id"].values
            rated_books = catalog_df[catalog_df["book_id"].isin(rated_book_ids)]

            # Extract and analyze genres from rated books
            all_genres = []
            for genres_str in rated_books["genres"]:
                if isinstance(genres_str, str):
                    genres = ast.literal_eval(genres_str)
                    all_genres.extend(genres)

            user_preferred_genres = pd.Series(all_genres).value_counts().to_dict()

            # logger.info(f"Number of rated books: {len(rated_books)}")
            # logger.info(f"Genre Distribution in Rated Books (Top 10):")
            # for genre, count in list(user_preferred_genres.items())[:10]:
            #     logger.info(f"  {genre}: {count}")

        for lambda_w in lambda_values:
            print(
                "Running recommendations for user=%s with lambda=%.2f",
                user_idx,
                lambda_w,
            )
            logger.info(f"{'-' * REPEATS}")
            logger.info(
                "Running recommendations for user=%s with lambda=%.2f",
                user_idx,
                lambda_w,
            )

            indices, scores, sources = _hybrid_recommender.hybrid_recommender(
                user_idx,
                user_factors,
                book_factors,
                train_matrix,
                catalog_embeddings,
                book_cf_idx_to_catalog_map,
                cf_idx_to_book,
                k=10,
                lambda_weight=lambda_w,
                candidate_pool_size=None,
                filter_rated=True,
            )

            # _hybrid_recommender.display_recommendations(logger, indices, scores, sources, catalog_df)

            # Analyze recommended books genres
            if len(indices) > 0:
                recommended_books = catalog_df.iloc[indices]

                rec_genres = []
                for _, row in recommended_books.iterrows():
                    genres_str = row["genres"]
                    if isinstance(genres_str, str):
                        genres = ast.literal_eval(genres_str)
                        rec_genres.extend(genres)

                rec_genre_counts = pd.Series(rec_genres).value_counts().to_dict()
                # logger.info(f"Genre Distribution in Recommended Books:")
                # for genre, count in list(rec_genre_counts.items())[:10]:
                #     logger.info(f"  {genre}: {count}")

                # Compare with user preferences
                if user_preferred_genres:
                    # logger.info(f"Genre Correlation Analysis:")
                    common_genres = set(user_preferred_genres.keys()) & set(rec_genre_counts.keys())
                    logger.info(f"User preferred genres: {list(user_preferred_genres.keys())[:10]}")
                    logger.info(f"Recommended genres: {list(rec_genre_counts.keys())[:10]}")
                    logger.info(f"Genres in both rated and recommended: {common_genres}")
                    if common_genres:
                        logger.info(f"✓ Good correlation found!")
                    else:
                        logger.info(f"⚠ No genre overlap detected")


def run_pipeline(logger, skip_done=True):
    """Run the complete recommendation system pipeline."""

    # Create necessary directories
    directories = [
        CLEAN_DATA_DIR,
        EMBEDDINGS_DIR,
        MATRICES_DIR,
        PKL_DIR,
        MODEL_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # ============================================================
    # STAGE 1: Data Preprocessing
    # ============================================================
    if skip_done and os.path.exists(OUTPUT_BOOKS) and os.path.exists(OUTPUT_RATINGS):
        logger.info("Stage 1 outputs already exist, skipping...")
    else:
        logger.info("=" * REPEATS)
        logger.info("STAGE 1: Data Preprocessing")
        logger.info("=" * REPEATS)

        run_data_preprocessing(logger)
        logger.info("✓ Stage 1 completed successfully")

    # ============================================================
    # STAGE 2: Create Embeddings
    # ============================================================
    # Check if outputs already exist
    if skip_done and os.path.exists(OUTPUT_CATALOG_BOOKS_INDEX) and os.path.exists(OUTPUT_CATALOG_BOOKS_EMBEDDINGS):
        logger.info("Stage 2 outputs already exist, skipping...")
    else:
        logger.info("=" * REPEATS)
        logger.info("STAGE 2: Generate Embeddings")
        logger.info("=" * REPEATS)

        run_create_embeddings(logger)
        logger.info("✓ Stage 2 completed successfully")

    # ============================================================
    # STAGE 3: Build Interaction Matrix
    # ============================================================
    if skip_done and os.path.exists(OUTPUT_TRAIN_MATRIX):
        logger.info("Stage 3 outputs already exist, skipping...")
    else:
        logger.info("=" * REPEATS)
        logger.info("STAGE 3: Build Interaction Matrix")
        logger.info("=" * REPEATS)

        run_build_interaction_matrix(logger)
        logger.info("✓ Stage 3 completed successfully")

    # ============================================================
    # STAGE 4: Train Collaborative Filtering Model
    # ============================================================
    if skip_done and os.path.exists(OUTPUT_ALS_MODEL):
        logger.info("Stage 4 outputs already exist, skipping...")
    else:
        logger.info("=" * REPEATS)
        logger.info("STAGE 4: Train Collaborative Filtering Model")
        logger.info("=" * REPEATS)

        run_train_cf_model(logger)
        logger.info("✓ Stage 4 completed successfully")

    # ============================================================
    # STAGE 5: Evaluate Recommendations
    # ============================================================
    logger.info("=" * REPEATS)
    logger.info("STAGE 5: Evaluate Recommendations")
    logger.info("=" * REPEATS)

    run_evaluation(logger)
    logger.info("✓ Stage 5 completed successfully")

    # ============================================================
    # STAGE 6: Run Hybrid Recommendations (Smoke Test)
    # ============================================================
    # logger.info("=" * REPEATS)
    # logger.info("STAGE 6: Run Recommendations")
    # logger.info("=" * REPEATS)

    # run_hybrid_recommender(logger)
    # logger.info("✓ Stage 6 completed successfully")

    logger.info("=" * REPEATS)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * REPEATS)
    return True


def main():
    """Parse arguments and run the pipeline."""
    logger = setup_logging("", LOG_FILE)
    success = run_pipeline(logger, skip_done=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
