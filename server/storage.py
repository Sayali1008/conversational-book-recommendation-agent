import sqlite3
from typing import Optional

class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    book_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('like','dislike','superlike')),
                    confidence REAL,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def log_swipe(self, user_id: str, book_id: int, action: str, confidence: Optional[float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO interactions (user_id, book_id, action, confidence) VALUES (?, ?, ?, ?)",
                (user_id, book_id, action, confidence),
            )
            conn.commit()