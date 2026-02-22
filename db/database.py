import sqlite3
from pathlib import Path

import numpy as np

from common.constants import DATABASE_DIR, PATHS
from common.utils import setup_logging

logger = setup_logging(__name__, PATHS["app_log_file"])


class Database:
    def __init__(self, db_path: str = PATHS["database"]):
        """Initialize database connection."""
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure database file and directory exist."""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        if not db_file.exists():
            logger.info(f"Creating new database at {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection to the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize_schema(self):
        """Create all tables if they don't exist."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    login_attempt INTEGER DEFAULT 0,
                    last_login TIMESTAMP
                )
            """
            )

            # Genres table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS genres (
                    genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """
            )

            # Authors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS authors (
                    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                )
            """
            )

            # Books table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS books (
                    book_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    infolink TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Book embeddings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS book_embeddings (
                    book_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (book_id) REFERENCES books(book_id)
                )
            """
            )

            # Book-Authors junction table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS book_authors (
                    book_id INTEGER,
                    author_id INTEGER,
                    PRIMARY KEY (book_id, author_id),
                    FOREIGN KEY (book_id) REFERENCES books(book_id),
                    FOREIGN KEY (author_id) REFERENCES authors(author_id)
                )
            """
            )

            # Book-Genres junction table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS book_genres (
                    book_id INTEGER,
                    genre_id INTEGER,
                    PRIMARY KEY (book_id, genre_id),
                    FOREIGN KEY (book_id) REFERENCES books(book_id),
                    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
                )
            """
            )

            # User Genres table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_genres (
                    user_id TEXT,
                    genre_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, genre_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (genre_id) REFERENCES genres(genre_id)
                )
            """
            )

            # Interactions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    book_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('like','dislike')),
                    confidence REAL,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (book_id) REFERENCES books(book_id)
                )
            """
            )

            # Ratings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    user_id TEXT,
                    book_id INTEGER,
                    score REAL,
                    confidence REAL,
                    datetime TIMESTAMP,
                    review_summary TEXT,
                    review_text TEXT,
                    PRIMARY KEY (user_id, book_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (book_id) REFERENCES books(book_id)
                )
            """
            )

            # Metadata table for system configuration
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )

            conn.commit()
            logger.info("✓ Database schema initialized successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing database schema: {e}")
            raise
        finally:
            conn.close()

    def initialize_metadata(self):
        """Initialize metadata table with default values if not present."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Check if last_training_date exists
            cursor.execute("SELECT value FROM metadata WHERE key = ?", ("last_training_date",))
            result = cursor.fetchone()

            if result is None:
                # Insert initial training date
                cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("last_training_date", "2025-02-21"))
                conn.commit()
                logger.info("✓ Metadata initialized with last_training_date=2025-02-21")
            else:
                logger.info(f"Metadata already initialized: last_training_date={result[0]}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error initializing metadata: {e}")
            raise
        finally:
            conn.close()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            return cursor.fetchone() is not None
        finally:
            conn.close()

    def get_table_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def clear_table(self, table_name: str):
        """Clear all data from a table (use with caution)."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()
            logger.info(f"Cleared table: {table_name}")
        except sqlite3.OperationalError as e:
            logger.error(f"Error clearing table {table_name}: {e}")
            raise
        finally:
            conn.close()

    def close_all(self):
        """Close database connections (useful for cleanup)."""
        pass

    def check_db_has_data(self):
        """Check if the database tables already have data to determine if we need to run the data pipeline."""
        db_path = DATABASE_DIR / "system.db"
        if not db_path.exists():
            return False
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM books")
            books_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM book_embeddings")
            book_embeddings_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM users")
            users_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ratings")
            ratings_count = cursor.fetchone()[0]

            return books_count > 0 and book_embeddings_count > 0 and users_count > 0 and ratings_count > 0
        except Exception as e:
            # If there's an error (e.g., tables don't exist), assume no data
            return False
        finally:
            conn.close()

    def generate_single_book_embedding(self, title: str, genres: list, description: str, model) -> np.ndarray:
        """Generate embedding for a single book."""
        genres_str = ", ".join(genres) if isinstance(genres, list) else str(genres)
        text = f"Title: {title} | Genre: {genres_str} | Description: {description}"
        embedding = model.encode(text, show_progress_bar=False)
        return np.array(embedding)

    def save_embedding_to_db(self, book_id: int, embedding: np.ndarray) -> None:
        """Save book embedding to database as BLOB."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            embedding_blob = embedding.tobytes()
            embedding_dim = embedding.shape[0]

            cursor.execute(
                "INSERT OR REPLACE INTO book_embeddings (book_id, embedding, dim, created_at) VALUES (?, ?, ?, ?)",
                (book_id, embedding_blob, embedding_dim, __import__("datetime").datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def load_embedding_from_db(self, book_id: int):
        """Load book embedding from database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding, dim FROM book_embeddings WHERE book_id = ?", (book_id,))
            result = cursor.fetchone()

            if result:
                embedding_blob, dim = result
                return np.frombuffer(embedding_blob, dtype=np.float32).reshape(dim)
            return None
        finally:
            conn.close()

    def load_all_embeddings_from_db(self, catalog_df) -> np.ndarray:
        """Load all embeddings from database, ordered by catalog book_id."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT book_id, embedding, dim FROM book_embeddings ORDER BY book_id")
            results = cursor.fetchall()

            if not results:
                return None

            # Build embedding matrix in catalog order
            embeddings_dict = {}
            first_dim = None

            for book_id, embedding_blob, dim in results:
                if first_dim is None:
                    first_dim = dim
                embeddings_dict[int(book_id)] = np.frombuffer(embedding_blob, dtype=np.float32).reshape(dim)

            # Stack embeddings in catalog order (book_id 1-indexed)
            embeddings_list = []
            for _, row in catalog_df.iterrows():
                book_id = int(row["book_id"])
                if book_id in embeddings_dict:
                    embeddings_list.append(embeddings_dict[book_id])
                else:
                    # Fallback: zero vector if embedding not found
                    embeddings_list.append(np.zeros(first_dim, dtype=np.float32))

            return np.array(embeddings_list, dtype=np.float32)
        finally:
            conn.close()
