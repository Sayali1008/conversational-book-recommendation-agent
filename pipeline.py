import ast
import os
import pickle
import sys
import time

import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

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
    ratings_df = _data_preprocessing.clean_ratings_data(
        logger, ratings_df, catalog_books_df
    )
    logger.info(f"✓ Ratings data cleaned: shape={ratings_df.shape}")

    catalog_books_df[OUTPUT_COLS_BOOKS].to_feather(OUTPUT_BOOKS)
    ratings_df[OUTPUT_COLS_RATINGS].to_feather(OUTPUT_RATINGS)


def run_create_embeddings(logger): 
    embeddings_model = SentenceTransformer(EMBEDDING_MODEL)
    catalog_books_df = safe_read_feather(OUTPUT_BOOKS)
    embeddings, index = _create_embeddings.generate_embeddings(
        logger, catalog_books_df, embeddings_model, BATCH_SIZE
    )

    # Example output: (n_catalog_books, 384) for all-MiniLM-L6-v2
    # Example output: (n_catalog_books, 768) for all-mpnet-base-v2


def run_build_interaction_matrix(logger):
    ratings_df = pd.read_feather(OUTPUT_RATINGS)
    logger.info(f"Ratings data size: {ratings_df.shape}")

    # Filtering once is not enough when constraints interact so we use a convergence loop
    # This guarantees that both constraints are simultaneously satisfied in the final dataframe.
    # Without a loop, it is likely that only one filter will stay true at a time.
    while True:
        prev_len = len(ratings_df)

        user_counts = ratings_df["user_id"].value_counts()
        ratings_df = ratings_df[
            ratings_df["user_id"].isin(
                user_counts[
                    (user_counts >= MIN_USER_INTERACTIONS)
                    & (user_counts <= MAX_USER_INTERACTIONS)
                ].index
            )
        ]

        book_counts = ratings_df["book_id"].value_counts()
        ratings_df = ratings_df[
            ratings_df["book_id"].isin(
                book_counts[book_counts >= MIN_BOOK_INTERACTIONS].index
            )
        ]

        if len(ratings_df) == prev_len:
            break

    logger.info(
        f"Ratings data size after filtering min/max interactions: {ratings_df.shape}"
    )

    # Create Index Mappings
    unique_users = ratings_df["user_id"].unique()
    unique_books = ratings_df["book_id"].unique()
    # Example: unique_books = [5, 12, 8, 100, 7, ...]  (in order of appearance)

    n_users = len(unique_users)
    n_cf_books = len(unique_books)

    logger.info("Users: %s", f"{n_users:,}")
    logger.info("CF-trainable books: %s", f"{n_cf_books:,}")

    # idx helps define the appearance order in ratings_df - hence can be used in reverse as well.
    user_to_cf_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    book_to_cf_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}

    with open(USER_IDX_PKL, "wb") as f:
        pickle.dump(user_to_cf_idx, f)

    with open(BOOK_IDX_PKL, "wb") as f:
        pickle.dump(book_to_cf_idx, f)

    logger.info("Index mappings saved.")

    # Step 2: Add Index Columns to DataFrame
    ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_cf_idx)
    ratings_df["book_idx"] = ratings_df["book_id"].map(book_to_cf_idx)

    assert ratings_df["user_idx"].notna().all(), "Some users failed to map to indices"
    assert ratings_df["book_idx"].notna().all(), "Some books failed to map to indices"

    logger.info("Index mapping complete.")

    # Step 3: Train / Val / Test Split
    # Split at interaction level (each row is one triplet)
    # Stratify by user to ensure each user has training data
    train_df, temp_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

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

    # Build matrices for train/val/test
    train_matrix = _train_cf.build_interaction_matrix(train_df, n_users, n_cf_books)
    val_matrix = _train_cf.build_interaction_matrix(val_df, n_users, n_cf_books)
    test_matrix = _train_cf.build_interaction_matrix(test_df, n_users, n_cf_books)

    logger.info("sUser Coverage Check ===")
    logger.info(f"Total unique users in filtered ratings: {n_users:,}")
    logger.info(f"User indices in train_matrix: 0 to {train_matrix.shape[0]-1}")
    logger.info(f"User indices in val_matrix: 0 to {val_matrix.shape[0]-1}")
    
    # Check if they match
    if train_matrix.shape[0] != val_matrix.shape[0]:
        logger.error("⚠️  MISMATCH: train and val matrices have different n_users!")
        logger.error(f"   train_matrix.shape[0] = {train_matrix.shape[0]}")
        logger.error(f"   val_matrix.shape[0] = {val_matrix.shape[0]}")
    else:
        logger.info(f"✓ Matrices aligned: both have {train_matrix.shape[0]:,} users")

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

    logger.info("=== Test Matrix ===")
    logger.info("Shape: %s", test_matrix.shape)
    logger.info("Non-zero entries: %s", f"{test_matrix.nnz:,}")

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

    sp.save_npz(OUTPUT_TRAIN_MATRIX, train_matrix)
    sp.save_npz(OUTPUT_VAL_MATRIX, val_matrix)
    sp.save_npz(OUTPUT_TEST_MATRIX, test_matrix)

    logger.info("Matrices saved successfully")


def run_train_cf_model(logger):
    # train_matrix = (user x book) sparse matrix
    train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)

    cf_model = _train_cf.model_initialization(
        logger, factors=64, regularization=0.001, iterations=15, alpha=40
    )
    cf_model = _train_cf.model_training(logger, cf_model, train_matrix)

    # Extract learned factors
    book_factors = cf_model.item_factors  # Shape: (n_cf_books, n_factors)
    user_factors = cf_model.user_factors  # Shape: (n_users, n_factors)

    # Save the trained model
    with open(OUTPUT_ALS_MODEL, "wb") as f:
        pickle.dump(cf_model, f)

    # Save factors
    np.save(OUTPUT_USER_FACTORS, user_factors)
    np.save(OUTPUT_BOOK_FACTORS, book_factors)

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

    # Random testing
    _train_cf.test_predictions(logger, user_factors, book_factors, user_cf_idx=0)
    _train_cf.test_predictions(logger, user_factors, book_factors, user_cf_idx=100)
    _train_cf.test_predictions(logger, user_factors, book_factors, user_cf_idx=200)


def analyze_recommendation_quality(logger, train_matrix, val_matrix, test_matrix):
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


def run_evaluation(logger):
    """
    Evaluate all recommendation strategies:
    - Pure CF (lambda=1.0)
    - Pure Embeddings (lambda=0.0)
    - Hybrid (various lambda values)
    """
    
    logger.info("Loading model artifacts for evaluation...")
    
    # Load trained model factors
    user_factors = np.load(OUTPUT_USER_FACTORS)
    book_factors = np.load(OUTPUT_BOOK_FACTORS)
    
    # Load interaction matrices
    train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)
    val_matrix = sp.load_npz(OUTPUT_VAL_MATRIX)
    test_matrix = sp.load_npz(OUTPUT_TEST_MATRIX)
    
    # Load embeddings
    catalog_embeddings = np.load(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)
    
    # Load mappings
    _, book_cf_idx_to_book_id = _hybrid_recommender.load_index_mappings(BOOK_IDX_PKL)
    cf_to_catalog_map = _hybrid_recommender.build_cf_to_catalog_mapping(
        book_cf_idx_to_book_id
    )
    
    logger.info(f"Loaded factors: user={user_factors.shape}, book={book_factors.shape}")
    logger.info(f"Loaded embeddings: {catalog_embeddings.shape}")
    logger.info(f"Loaded matrices: train={train_matrix.shape}, val={val_matrix.shape}, test={test_matrix.shape}")

    # Analyze data quality issues
    analyze_recommendation_quality(logger, train_matrix, val_matrix, test_matrix)
    
    # Evaluate multiple K values and lambda strategies
    # k_values = [5, 10, 20]
    # lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0=pure embedding, 1=pure CF
    k_values = [10] 
    lambda_values = [0.0, 0.5, 1.0]

    # Number of val users to evaluate: 14978
    # Users to evaluate: [    0     1     2 ... 25995 25997 25998]
    _evaluate.run_evaluation(
        logger=logger,
        user_factors=user_factors,
        book_factors=book_factors,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        test_matrix=test_matrix,
        catalog_embeddings=catalog_embeddings,
        cf_to_catalog_map=cf_to_catalog_map,
        book_cf_idx_to_book_id=book_cf_idx_to_book_id,
        k_values=k_values,
        lambda_values=lambda_values,
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
    user_to_cf_idx, user_cf_idx_to_user_id = _hybrid_recommender.load_index_mappings(USER_IDX_PKL)
    book_to_cf_idx, book_cf_idx_to_book_id = _hybrid_recommender.load_index_mappings(BOOK_IDX_PKL)

    # Build CF → catalog mapping
    book_cf_idx_to_catalog_map = _hybrid_recommender.build_cf_to_catalog_mapping(book_cf_idx_to_book_id)

    # Test different users and lambda values
    # lambda_weight = 0 is pure embedding, 1 is pure CF
    lambda_values = [0.5]
    test_users = [9393, 21756, 4570, 8260, 25932, 3818, 17023, 20445, 118, 18286]

    for user_idx in test_users:
        # Get the actual user_id from the index mapping
        user_id = user_cf_idx_to_user_id.get(user_idx)
        
        logger.info(f"\n{'#' * REPEATS}")
        logger.info(f"USER {user_idx}")
        logger.info(f"{'#' * REPEATS}")

        # Analyze user's rated books and their genres
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]

        if len(user_ratings) == 0:
            logger.warning(f"No ratings found for user {user_id}")
        else:
            rated_book_ids = user_ratings['book_id'].values
            rated_books = catalog_df[catalog_df['book_id'].isin(rated_book_ids)]
            
            # Extract and analyze genres from rated books
            all_genres = []
            for genres_str in rated_books['genres']:
                if isinstance(genres_str, str):
                    genres = ast.literal_eval(genres_str)
                    all_genres.extend(genres)
            
            user_preferred_genres = pd.Series(all_genres).value_counts().to_dict()
            
            # logger.info(f"Number of rated books: {len(rated_books)}")
            # logger.info(f"Genre Distribution in Rated Books (Top 10):")
            # for genre, count in list(user_preferred_genres.items())[:10]:
            #     logger.info(f"  {genre}: {count}")

        for lambda_w in lambda_values:
            print("Running recommendations for user=%s with lambda=%.2f", user_idx, lambda_w)
            logger.info(f"\n{'-' * REPEATS}")
            logger.info("Running recommendations for user=%s with lambda=%.2f", user_idx, lambda_w)

            indices, scores, sources = _hybrid_recommender.hybrid_recommender(
                user_idx=user_idx,
                user_factors=user_factors,
                book_factors=book_factors,
                train_matrix=train_matrix,
                catalog_embeddings=catalog_embeddings,
                cf_to_catalog_map=book_cf_idx_to_catalog_map,
                book_cf_idx_to_book_id=book_cf_idx_to_book_id,
                k=10,
                lambda_weight=lambda_w,
                # filter_rated=False
            )

            # _hybrid_recommender.display_recommendations(logger, indices, scores, sources, catalog_df)

            # Analyze recommended books genres
            if len(indices) > 0:
                recommended_books = catalog_df.iloc[indices]
                
                rec_genres = []
                for _, row in recommended_books.iterrows():
                    genres_str = row['genres']
                    if isinstance(genres_str, str):
                        genres = ast.literal_eval(genres_str)
                        rec_genres.extend(genres)
                
                rec_genre_counts = pd.Series(rec_genres).value_counts().to_dict()
                # logger.info(f"\nGenre Distribution in Recommended Books:")
                # for genre, count in list(rec_genre_counts.items())[:10]:
                #     logger.info(f"  {genre}: {count}")
                
                # Compare with user preferences
                if user_preferred_genres:
                    # logger.info(f"\nGenre Correlation Analysis:")
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
        FACTORS_DIR,
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
    if (
        skip_done
        and os.path.exists(OUTPUT_CATALOG_BOOKS_INDEX)
        and os.path.exists(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)
    ):
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
    # STAGE 5: Evaluate Recommendation Strategies
    # ============================================================
    logger.info("=" * REPEATS)
    logger.info("STAGE 5: Evaluate Recommendation Strategies")
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
    logger = setup_logging("pipeline", LOG_FILE)
    success = run_pipeline(logger, skip_done=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

