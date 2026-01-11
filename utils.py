"""
Shared utilities for the recommendation system pipeline.
Provides common functions for data loading, logging, and validation.
"""

import logging
import os
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
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

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
def safe_read_feather(
    filepath: str, usecols: Optional[list[str]] = None
) -> pd.DataFrame:
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
