"""Database connection management and schema initialization."""

from typing import Optional

from common.constants import PATHS
from common.utils import setup_logging
from db.database import Database

logger = setup_logging(__name__, PATHS["app_log_file"])

# Global database instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.initialize_schema()
        _db_instance.initialize_metadata()
    return _db_instance


def reset_db_instance():
    """Reset the global database instance (for testing)."""
    global _db_instance
    _db_instance = None
