"""
Shared utilities for the recommendation system pipeline.
Provides common functions for data loading, logging, and validation.
"""

import logging
import os
import pickle
import sys
from typing import Optional

import pandas as pd
from pydantic import ConfigDict, validate_call


def setup_logging(stage_name: str, log_file: str, level=logging.INFO):
    """Configure logging for a pipeline stage."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        # format="%(message)s",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_file,
        filemode="w",
        encoding="utf-8",
    )
    logger = logging.getLogger(stage_name)

    # Prevent duplicate handlers if main() is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Explicitly ensure the stream is flushed after every write
    console_handler.flush = sys.stdout.flush
    # logger.addHandler(console_handler)

    return logger


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
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
        item_id_to_cf_idx: dict mapping item_id (book or user) → CF matrix column index
        cf_idx_to_item_id: dict mapping CF matrix column index → item_id
    """
    with open(pkl_file, "rb") as f:
        item_id_to_cf_idx = pickle.load(f)

    # Create reverse mapping: CF index → item_id
    cf_idx_to_item_id = {cf_idx: item_id for item_id, cf_idx in item_id_to_cf_idx.items()}
    return item_id_to_cf_idx, cf_idx_to_item_id


def map_book_cf_idx_to_catalog_index(cf_idx_to_book):
    """
    Build mapping from CF book indices to catalog indices.

    Args:
        cf_idx_to_book: Mapping from CF index to book_id

    Returns:
        dict: CF index → catalog index mapping
    """
    cf_idx_to_catalog_id_map = {}
    for cf_idx, book_id in cf_idx_to_book.items():
        catalog_idx = book_id - 1  # book_id starts at 1, catalog indices start at 0
        cf_idx_to_catalog_id_map[cf_idx] = catalog_idx

    return cf_idx_to_catalog_id_map


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
