"""
Batch retraining pipeline for collaborative filtering model.

This script is designed to run as:
- A weekly cron job (Sunday 2 AM ET)
- Manual admin trigger via API endpoint

The pipeline:
1. Fetches recent swipes/interactions from the database
2. Merges swipes with original training data
3. Retrains ALS model with same hyperparameters
4. Saves updated user and book factors
5. Updates metadata with last training timestamp
6. Rolls back on failure (atomic operation)

Usage:
    python scripts/batch_retrain.py
"""

import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from common.constants import PATHS, MODEL_DIR, MATRICES_DIR
from common.utils import setup_logging, safe_read_feather, save_pickle, load_pickle
from db.connection import get_db

logger = setup_logging(__name__, PATHS["eval_log_file"])

# Best hyperparameters from grid search (see model_pipeline.py)
BEST_ALS_PARAMS = {
    "alpha": 80,
    "factors": 128,
    "iterations": 15,
    "random_state": 42,
    "regularization": 0.2,
}


def fetch_recent_swipes(last_training_date_str: str) -> pd.DataFrame:
    """
    Fetch user swipes (interactions) from database since last training date.
    
    Args:
        last_training_date_str: Date string in format YYYY-MM-DD
        
    Returns:
        DataFrame with columns: user_id, book_id, action, ts
    """
    try:
        db = get_db()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Parse date
        last_training_date = pd.to_datetime(last_training_date_str)
        
        cursor.execute(
            """
            SELECT user_id, book_id, action, ts 
            FROM interactions 
            WHERE ts > datetime(?)
            ORDER BY ts ASC
            """,
            (last_training_date.isoformat(),),
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            logger.info(f"No new swipes found since {last_training_date_str}")
            return pd.DataFrame(columns=["user_id", "book_id", "action", "ts"])
        
        # Convert to DataFrame
        swipes_df = pd.DataFrame(
            rows,
            columns=["user_id", "book_id", "action", "ts"]
        )
        
        # Convert action to confidence (like=1.0, dislike=0.0)
        swipes_df["confidence"] = (swipes_df["action"] == "like").astype(float)
        
        logger.info(f"Fetched {len(swipes_df)} new swipes since {last_training_date_str}")
        
        return swipes_df[["user_id", "book_id", "confidence"]].copy()
    
    except Exception as e:
        logger.error(f"Error fetching recent swipes: {str(e)}\n{traceback.format_exc()}")
        raise


def build_updated_training_matrix(
    original_ratings_df: pd.DataFrame,
    new_swipes_df: pd.DataFrame,
    user_id_to_cf: dict,
    book_id_to_cf: dict,
) -> sp.csr_matrix:
    """
    Merge original training data with new swipes and rebuild interaction matrix.
    
    For users/books new to the swipes, we expand the index mappings.
    
    Args:
        original_ratings_df: Original cleaned ratings data
        new_swipes_df: New interactions from database
        user_id_to_cf: Existing user ID to CF index mapping
        book_id_to_cf: Existing book ID to CF index mapping
        
    Returns:
        Updated sparse training matrix
    """
    try:
        # Combine datasets
        combined_df = pd.concat(
            [
                original_ratings_df[["user_id", "book_id", "confidence"]],
                new_swipes_df,
            ],
            ignore_index=True,
        )
        
        logger.info(f"Combined {len(original_ratings_df)} original + {len(new_swipes_df)} new swipes = {len(combined_df)} total interactions")
        
        # Handle new users
        unique_users = combined_df["user_id"].unique()
        for user_id in unique_users:
            if user_id not in user_id_to_cf:
                new_cf_idx = len(user_id_to_cf)
                user_id_to_cf[user_id] = new_cf_idx
                logger.debug(f"Added new user {user_id} at CF index {new_cf_idx}")
        
        # Handle new books
        unique_books = combined_df["book_id"].unique()
        for book_id in unique_books:
            if book_id not in book_id_to_cf:
                new_cf_idx = len(book_id_to_cf)
                book_id_to_cf[book_id] = new_cf_idx
                logger.debug(f"Added new book {book_id} at CF index {new_cf_idx}")
        
        # Map to CF indices
        combined_df["user_idx"] = combined_df["user_id"].map(user_id_to_cf)
        combined_df["book_idx"] = combined_df["book_id"].map(book_id_to_cf)
        
        # Build matrix
        n_users = len(user_id_to_cf)
        n_books = len(book_id_to_cf)
        
        row = combined_df["user_idx"].values
        col = combined_df["book_idx"].values
        data = combined_df["confidence"].values
        
        train_matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_books), dtype=np.float32)
        
        logger.info(f"Built training matrix: ({n_users} users, {n_books} books) with {train_matrix.nnz} interactions")
        
        return train_matrix
    
    except Exception as e:
        logger.error(f"Error building updated training matrix: {str(e)}\n{traceback.format_exc()}")
        raise


def retrain_cf_model(train_matrix: sp.csr_matrix) -> AlternatingLeastSquares:
    """
    Retrain ALS collaborative filtering model on updated training matrix.
    
    Args:
        train_matrix: Updated sparse interaction matrix
        
    Returns:
        Trained ALS model
    """
    try:
        logger.info(f"Retraining ALS model with params: {BEST_ALS_PARAMS}")
        
        model = AlternatingLeastSquares(**BEST_ALS_PARAMS)
        model.fit(train_matrix, show_progress=True)
        
        logger.info("✓ ALS model retraining complete")
        
        return model
    
    except Exception as e:
        logger.error(f"Error retraining CF model: {str(e)}\n{traceback.format_exc()}")
        raise


def backup_artifacts():
    """Create backup of current model artifacts for rollback."""
    try:
        backup_dir = Path(MODEL_DIR) / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup factors
        for artifact in ["user_factors.npy", "book_factors.npy"]:
            src = Path(MODEL_DIR) / artifact
            if src.exists():
                dst = backup_dir / artifact
                shutil.copy2(src, dst)
                logger.info(f"Backed up {artifact}")
        
        return backup_dir
    
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}\n{traceback.format_exc()}")
        raise


def save_artifacts(model: AlternatingLeastSquares, user_id_to_cf: dict, book_id_to_cf: dict):
    """
    Save updated model artifacts.
    
    Args:
        model: Trained ALS model
        user_id_to_cf: Updated user ID mapping
        book_id_to_cf: Updated book ID mapping
    """
    try:
        # Save factors
        np.save(PATHS["user_factors"], model.user_factors)
        logger.info(f"✓ Saved user factors: {PATHS['user_factors']}")
        
        np.save(PATHS["book_factors"], model.item_factors)
        logger.info(f"✓ Saved book factors: {PATHS['book_factors']}")
        
        # Save index mappings
        save_pickle(user_id_to_cf, PATHS["user_idx_pkl"])
        logger.info(f"✓ Saved user index mapping: {PATHS['user_idx_pkl']}")
        
        save_pickle(book_id_to_cf, PATHS["book_idx_pkl"])
        logger.info(f"✓ Saved book index mapping: {PATHS['book_idx_pkl']}")
    
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}\n{traceback.format_exc()}")
        raise


def rollback_artifacts(backup_dir: Path):
    """Restore artifacts from backup."""
    try:
        logger.warning("Rolling back to previous model artifacts...")
        
        for artifact in ["user_factors.npy", "book_factors.npy"]:
            src = backup_dir / artifact
            dst = Path(MODEL_DIR) / artifact
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Restored {artifact}")
        
        # Remove backup directory after successful rollback
        shutil.rmtree(backup_dir)
        logger.info("✓ Rollback complete")
    
    except Exception as e:
        logger.error(f"Error during rollback: {str(e)}\n{traceback.format_exc()}")
        raise


def update_last_training_date(new_date_str: str):
    """
    Update the last_training_date metadata in database.
    
    Args:
        new_date_str: Date string in format YYYY-MM-DD
    """
    try:
        db = get_db()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE metadata SET value = ? WHERE key = ?",
            (new_date_str, "last_training_date"),
        )
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Updated last_training_date to {new_date_str}")
    
    except Exception as e:
        logger.error(f"Error updating metadata: {str(e)}\n{traceback.format_exc()}")
        raise


def get_last_training_date() -> str:
    """Fetch last training date from metadata."""
    try:
        db = get_db()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM metadata WHERE key = ?", ("last_training_date",))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            logger.warning("last_training_date not found in metadata, using 2026-01-31")
            return "2026-01-31"
    
    except Exception as e:
        logger.error(f"Error fetching last_training_date: {str(e)}\n{traceback.format_exc()}")
        raise


def main():
    """Main batch retraining orchestration."""
    logger.info("=" * 80)
    logger.info("BATCH RETRAINING PIPELINE STARTED")
    logger.info("=" * 80)
    
    backup_dir = None
    
    try:
        # Step 1: Get last training date
        last_training_date = get_last_training_date()
        logger.info(f"Last training date: {last_training_date}")
        
        # Step 2: Fetch recent swipes
        new_swipes = fetch_recent_swipes(last_training_date)
        if len(new_swipes) == 0:
            logger.info("No new swipes to retrain on. Exiting.")
            return True
        
        # Step 3: Load original training data and index mappings
        logger.info("Loading original training data and mappings...")
        original_ratings_df = safe_read_feather(PATHS["clean_ratings"])
        user_id_to_cf = load_pickle(PATHS["user_idx_pkl"])
        book_id_to_cf = load_pickle(PATHS["book_idx_pkl"])
        logger.info(f"Loaded {len(original_ratings_df)} original ratings")
        
        # Step 4: Create backup before training
        backup_dir = backup_artifacts()
        
        # Step 5: Build updated training matrix
        updated_train_matrix = build_updated_training_matrix(
            original_ratings_df, new_swipes, user_id_to_cf, book_id_to_cf
        )
        
        # Step 6: Retrain model
        retrained_model = retrain_cf_model(updated_train_matrix)
        
        # Step 7: Save new artifacts
        save_artifacts(retrained_model, user_id_to_cf, book_id_to_cf)
        
        # Step 8: Update metadata
        today_str = datetime.now().strftime("%Y-%m-%d")
        update_last_training_date(today_str)
        
        # Step 9: Clean up backup after successful completion
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)
            logger.info("Backup directory cleaned up")
        
        logger.info("=" * 80)
        logger.info("✓ BATCH RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Batch retraining failed: {str(e)}\n{traceback.format_exc()}")
        
        # Rollback on failure
        if backup_dir and backup_dir.exists():
            try:
                rollback_artifacts(backup_dir)
            except Exception as rollback_err:
                logger.error(f"Rollback also failed: {str(rollback_err)}")
        
        logger.error("=" * 80)
        logger.error("✗ BATCH RETRAINING PIPELINE FAILED")
        logger.error("=" * 80)
        
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
