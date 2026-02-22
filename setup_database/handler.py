from pathlib import Path

from sentence_transformers import SentenceTransformer

from common import utils
from common.constants import *
from setup_database.data_pipeline import *
from db.connection import get_db
from setup_database.migrate_data_to_db import *
logger = setup_logging(__name__, PATHS["eval_log_file"])


def run_data_pipeline():
    try:
        # if data exists in the database, no need to run the pipeline again
        db = get_db()
        if db.check_db_has_data():
            logger.info("Database already has data. Skipping data pipeline.")
            return db

        for dir_path in [CLEAN_DATA_DIR, DATABASE_DIR, MODEL_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        logger.info("Loading raw books data...")
        books_df = utils.safe_read_csv(PATHS["books"], DATA_PREPROCESSING["input_cols_books"])
        logger.info(f"Loaded {len(books_df)} books")

        logger.info("Cleaning books data...")
        catalog_books_df = clean_books_data(books_df)

        logger.info("Loading raw ratings data...")
        ratings_df = utils.safe_read_csv(PATHS["ratings"], DATA_PREPROCESSING["input_cols_ratings"])
        logger.info(f"Loaded {len(ratings_df)} ratings")

        logger.info("Cleaning ratings data...")
        ratings_df = clean_ratings_data(ratings_df, catalog_books_df)

        catalog_books_df[DATA_PREPROCESSING["output_cols_books"]].to_feather(PATHS["clean_books"])
        ratings_df[DATA_PREPROCESSING["output_cols_ratings"]].to_feather(PATHS["clean_ratings"])

        logger.info("Starting data migration to database...")
        migrate_to_db()
        logger.info("✓ Migration to database completed")

        logger.info("Loading sentence transformer...")
        model = SentenceTransformer(EMBEDDINGS["embedding_model"])

        logger.info("Loading cleaned books...")
        catalog_df = utils.safe_read_feather(PATHS["clean_books"])

        logger.info("Generating embeddings...")
        embeddings, index = generate_embeddings(catalog_df, model, EMBEDDINGS["batch_size"])

        # save embeddings to database
        logger.info("Saving embeddings to database...")
        catalog_df = utils.safe_read_feather(PATHS["clean_books"])
        save_embeddings_to_database(catalog_df, embeddings)
        logger.info("✓ Embeddings saved to database")

        logger.info("✓ Data pipeline completed")

        return db
    except Exception as e:
        raise
