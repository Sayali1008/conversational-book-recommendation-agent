"""
Interactions data access layer.
Manages user interactions (likes/dislikes) with books.
"""

from typing import List, Optional

from db.connection import get_db


class Interactions:
    """Manager for user interactions (likes/dislikes) with books."""

    def __init__(self):
        """Initialize with global database instance."""
        self.db = get_db()

    def _get_connection(self):
        """Get database connection from global instance."""
        return self.db.get_connection()

    def insert_swipe(self, user_id: str, book_id: int, action: str, confidence: Optional[float]):
        """Insert a swipe interaction into the database."""
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO interactions (user_id, book_id, action, confidence) VALUES (?, ?, ?, ?)",
                (user_id, book_id, action, confidence),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user_swiped_books(self, user_id: str, actions: Optional[List[str]] = None, limit: Optional[int] = None):
        """Return list of interactions for a user, optionally filtered by action."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM interactions WHERE user_id = ?"
            params = [user_id]
            if actions:
                placeholders = ",".join(["?"] * len(actions))
                query += f" AND action IN ({placeholders})"
                params.extend(actions)
            query += " ORDER BY ts DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            conn.row_factory = __import__("sqlite3").Row
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def get_interaction_count(self, user_id: str):
        """Return number of interactions for a user."""
        conn = self._get_connection()
        try:
            query = "SELECT COUNT(*) as count FROM interactions WHERE user_id = ?"
            params = [user_id]
            conn.row_factory = __import__("sqlite3").Row
            cur = conn.cursor()
            cur.execute(query, params)
            result = cur.fetchone()
            return result["count"] if result else 0
        finally:
            conn.close()
