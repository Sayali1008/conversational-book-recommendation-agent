import logging
import os
import pickle
import sys
from typing import Optional

import pandas as pd


def setup_logging(stage_name: str, log_file: str, level=logging.INFO):
    """Configure logging for a pipeline stage.
    
    Args:
        stage_name: Name for the logger (typically __name__)
        log_file: Path to the log file to write to
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(stage_name)
    logger.handlers.clear()
    
    # Disable propagation to root logger to prevent duplicate logging
    logger.propagate = False

    logger.setLevel(level)
    
    # File Handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    return logger


def safe_read_csv(filepath: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Safely read CSV file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath).astype(str).fillna("")

        df.columns = df.columns.str.lower()
        if usecols:
            missing_cols = [c for c in usecols if c.lower() not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in input CSV: {missing_cols}")
            return df[usecols]
        return df
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing {filepath}: {e}")


def safe_read_feather(filepath: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Safely read Feather file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_feather(filepath).astype(str).fillna("")

        df.columns = df.columns.str.lower()
        if usecols:
            missing_cols = [c for c in usecols if c.lower() not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in input Feather: {missing_cols}")
            return df[usecols]
        return df
    except Exception as e:
        raise ValueError(f"Error reading or processing feather file {filepath}: {e}")


def load_index_mappings(pkl_file):
    """
    Load item index mappings from pickle files.

    Returns:
        item_id_to_cf: dict mapping item_id (book or user) → CF matrix column index
        cf_to_item_id: dict mapping CF matrix column index → item_id
    """
    with open(pkl_file, "rb") as f:
        item_id_to_cf = pickle.load(f)

    # Create reverse mapping: CF index → item_id
    cf_to_item_id = {cf_idx: item_id for item_id, cf_idx in item_id_to_cf.items()}
    return item_id_to_cf, cf_to_item_id


def map_book_cf_to_catalog_id(cf_to_book_id):
    """Build mapping from CF book indices to catalog indices."""
    cf_to_catalog_id = {}
    for cf, book_id in cf_to_book_id.items():
        catalog_id = book_id - 1  # book_id starts at 1, catalog indices start at 0
        cf_to_catalog_id[cf] = catalog_id

    return cf_to_catalog_id


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

