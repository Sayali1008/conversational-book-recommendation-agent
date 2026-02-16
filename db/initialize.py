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

logger = setup_logging(__name__, PATHS["app_log_file"])


def initialize_database(run_migration_flag: bool = False):
    """
    Initialize the database with schema and optionally run migration.
    
    Call this once at app startup to ensure database is ready.
    
    Args:
        run_migration_flag: If True, run migration. Data migration is slow (1.2M+ rows),
                           so it's disabled by default. Enable only when needed.
                           Run separately via: python -m scripts.migrate_data_to_db
    """
    try:
        # Step 1: Create schema and get global instance
        db = get_db()

        # Step 2: Check if migration is needed (only if enabled)
        if run_migration_flag:
            from scripts.migrate_data_to_db import migrate_data
            
            users_count = db.get_table_count("users")
            ratings_count = db.get_table_count("ratings")

            migration_needed = users_count == 0 or ratings_count == 0

            if migration_needed:
                logger.info("Running data migration (this may take 5-10 minutes)...")
                migrate_data()
                logger.info("✓ Data migration complete")
        else:
            users_count = db.get_table_count("users")
            ratings_count = db.get_table_count("ratings")
            logger.info(f"Database status: {users_count} users, {ratings_count} ratings")
            if users_count == 0:
                logger.info("To run data migration, use: python -m scripts.migrate_data_to_db")

        logger.info("✓ Database initialization complete")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def reset_database_for_testing():
    """
    Reset database instance for testing purposes.
    Useful for test isolation.
    """
    reset_db_instance()
    logger.info("Database instance reset for testing")
