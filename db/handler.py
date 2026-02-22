"""
Database initialization and migration orchestration.

This module handles the complete database setup pipeline:
1. Create schema (tables, foreign keys, constraints)
2. Run one-time data migration (if needed, optional)
3. Initialize global database instance
"""

from common.constants import PATHS
from common.utils import setup_logging
from db.connection import get_db, reset_db_instance
from setup_database.handler import run_data_pipeline

logger = setup_logging(__name__, PATHS["app_log_file"])


def initialize_database():
    """
    Initialize the database with schema and optionally run migration.
    Call this once at app startup to ensure database is ready.
    """
    # Step 1: Create schema and get global instance
    db = get_db()

    # Step 2: Check if migration is needed (only if enabled)
    users_count = db.get_table_count("users")
    ratings_count = db.get_table_count("ratings")
    migration_needed = users_count == 0 or ratings_count == 0

    if migration_needed:
        logger.info("Running data migration (this may take 5-10 minutes)...")
        run_data_pipeline()
        logger.info("✓ Data migration complete")

    logger.info("✓ Database initialization complete")
    return db

def reset_database_for_testing():
    """
    Reset database instance for testing purposes.
    Useful for test isolation.
    """
    reset_db_instance()
    logger.info("Database instance reset for testing")
