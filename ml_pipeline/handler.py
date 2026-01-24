import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from ml_pipeline.stages import stage_1_data_preprocessing as stage_1
from ml_pipeline.stages import stage_2_embeddings as stage_2
from ml_pipeline.stages import stage_3_matrices as stage_3
from ml_pipeline.stages import stage_4_training as stage_4
from ml_pipeline.stages import stage_5_evaluation as stage_5

from common.constants import PATHS, DATA_PREPROCESSING, EMBEDDINGS, INTERACTION_MATRIX, CF_MODEL_PARAMS
from common.logging import log_interaction_summary, log_model_statistics
from common.utils import setup_logging, safe_read_csv, safe_read_feather, save_pickle

logger = setup_logging(__name__, PATHS["eval_log_file"])

def run_stage_1_preprocessing():
    try:
        logger.info("Loading raw books data...")
        books_df = safe_read_csv(PATHS["books"], DATA_PREPROCESSING["input_cols_books"])
        logger.info(f"Loaded {len(books_df)} books")

        logger.info("Cleaning books data...")
        catalog_books_df = stage_1.clean_books_data(books_df)

        logger.info("Loading raw ratings data...")
        ratings_df = safe_read_csv(PATHS["ratings"], DATA_PREPROCESSING["input_cols_ratings"])
        logger.info(f"Loaded {len(ratings_df)} ratings")

        logger.info("Cleaning ratings data...")
        ratings_df = stage_1.clean_ratings_data(ratings_df, catalog_books_df)

        catalog_books_df[DATA_PREPROCESSING["output_cols_books"]].to_feather(PATHS["clean_books"])
        ratings_df[DATA_PREPROCESSING["output_cols_ratings"]].to_feather(PATHS["clean_ratings"])

        logger.info("✓ Stage 1 completed")
    except Exception as e:
        raise


def run_stage_2_embeddings():
    try:
        logger.info("Loading sentence transformer...")
        model = SentenceTransformer(EMBEDDINGS["embedding_model"])

        logger.info("Loading cleaned books...")
        catalog_df = safe_read_feather(PATHS["clean_books"])

        logger.info("Generating embeddings...")
        embeddings, index = stage_2.generate_embeddings(catalog_df, model, EMBEDDINGS["batch_size"])

        np.save(PATHS["catalog_books_embeddings"], embeddings)
        logger.info(f"✓ Stage 2 completed - embeddings shape: {embeddings.shape}")
    except Exception as e:
        raise


def run_stage_3_matrices():
    try:
        logger.info("Preparing data...")
        n_users, n_books, train_df, val_df, test_df = stage_3.prepare_data()

        logger.info("Building matrices...")
        train_matrix = stage_3.build_interaction_matrix(train_df, n_users, n_books, binary=True)
            
        val_matrix = stage_3.build_interaction_matrix(val_df, n_users, n_books, binary=True)
        test_matrix = stage_3.build_interaction_matrix(test_df, n_users, n_books, binary=True)

        log_interaction_summary(logger, n_users, n_books, train_matrix, val_matrix, test_matrix)

        sp.save_npz(PATHS["train_matrix"], train_matrix)
        sp.save_npz(PATHS["val_matrix"], val_matrix)
        sp.save_npz(PATHS["test_matrix"], test_matrix)

        logger.info("✓ Stage 3 completed")
    except Exception as e:
        raise


def run_stage_4_training():
    try:
        logger.info("Loading train matrix...")
        train_matrix = sp.load_npz(PATHS["train_matrix"])

        logger.info("Initializing ALS model...")
        cf_model = stage_4.model_initialization()

        logger.info("Model training...")
        cf_model = stage_4.model_training(cf_model, train_matrix)

        book_factors = cf_model.item_factors
        user_factors = cf_model.user_factors

        log_model_statistics(logger, user_factors, book_factors)

        save_pickle(cf_model, PATHS["als_model"])
        np.save(PATHS["user_factors"], user_factors)
        np.save(PATHS["book_factors"], book_factors)

        logger.info("✓ Stage 4 completed")
    except Exception as e:
        raise


def run_stage_5_evaluation():
    logger.info("Running evaluation...")
    stage_5.run_evaluation()
    logger.info("✓ Stage 5 completed")
    return True


# Stage registry - order and dependencies
STAGES = [
    ("stage_1_preprocessing", run_stage_1_preprocessing, []),
    ("stage_2_embeddings", run_stage_2_embeddings, ["stage_1_preprocessing"]),
    ("stage_3_matrices", run_stage_3_matrices, ["stage_1_preprocessing"]),
    ("stage_4_training", run_stage_4_training, ["stage_3_matrices"]),
    ("stage_5_evaluation", run_stage_5_evaluation, ["stage_2_embeddings", "stage_4_training"]),
]
