"""
Migration script to load data from FTR/CSV files into SQLite database.
Handles:
- Loading cleaned books catalog
- Loading ratings data
- Extracting and populating genres, authors, users
- Building junction tables for books-authors and books-genres
"""

import ast
import sqlite3

from common.constants import PATHS
from common.utils import safe_read_feather, setup_logging
from db.connection import get_db

logger = setup_logging(__name__, PATHS["app_log_file"])


class DataMigration:
    """Handles migration of data from FTR/CSV files to SQLite."""

    def __init__(self):
        """Initialize migration with database connection."""
        self.db = get_db()
        self.conn = self.db.get_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure connection is closed."""
        if self.conn:
            self.conn.close()

    def migrate_all(self):
        try:
            logger.info("Starting complete data migration...")

            # 1. Load genres
            self.migrate_genres()

            # 2. Load authors
            self.migrate_authors()

            # 3. Load books (separate from genres)
            self.migrate_books()

            # 4. Load book-genres junction
            self.migrate_book_genres()

            # 5. Load book_authors junction
            self.migrate_book_authors()

            # 6. Extract users from ratings first (before loading ratings)
            self.migrate_users()

            # 7. Load ratings (now that users exist)
            self.migrate_ratings()

            logger.info("✓ Migration completed successfully")
            return True

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise

    def migrate_genres(self):
        """Extract and load top-50 genres from books catalog."""
        logger.info("Migrating genres...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM genres")
        if cursor.fetchone()[0] > 0:
            logger.info("Genres already migrated, skipping")
            return
        
        # Load books catalog
        catalog_df = safe_read_feather(PATHS["clean_books"])
        logger.info(f"Loaded {len(catalog_df)} books from catalog")

        # Extract all genres
        all_genres = set()
        for genres_list in catalog_df["genres"]:
            if isinstance(genres_list, str):
                genres_list = ast.literal_eval(genres_list)
            if isinstance(genres_list, list):
                all_genres.update(genres_list)

        logger.info(f"Found {len(all_genres)} unique genres in catalog")

        # Insert genres (order doesn't matter, but sort for consistency)
        genres_sorted = sorted(all_genres)
        cursor.executemany("INSERT OR IGNORE INTO genres (name) VALUES (?)", [(g,) for g in genres_sorted])
        self.conn.commit()
        logger.info(f"✓ Inserted {len(genres_sorted)} genres")

    def migrate_authors(self):
        """Extract and load authors from books catalog."""
        logger.info("Migrating authors...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM authors")
        if cursor.fetchone()[0] > 0:
            logger.info("Authors already migrated, skipping")
            return

        # Load books catalog
        catalog_df = safe_read_feather(PATHS["clean_books"])

        # Extract all authors
        all_authors = set()
        for authors_list in catalog_df["authors"]:
            if isinstance(authors_list, str):
                authors_list = ast.literal_eval(authors_list)
            if isinstance(authors_list, list):
                all_authors.update(authors_list)

        logger.info(f"Found {len(all_authors)} unique authors in catalog")

        # Insert authors (order doesn't matter, but sort for consistency)
        authors_sorted = sorted(all_authors)
        cursor.executemany("INSERT OR IGNORE INTO authors (name) VALUES (?)", [(a,) for a in authors_sorted])
        self.conn.commit()
        logger.info(f"✓ Inserted {len(authors_sorted)} authors")

    def migrate_books(self):
        """Load books catalog."""
        logger.info("Migrating books...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM books")
        if cursor.fetchone()[0] > 0:
            logger.info("Books already migrated, skipping")
            return

        # Load books catalog
        catalog_df = safe_read_feather(PATHS["clean_books"])
        logger.info(f"Loading {len(catalog_df)} books from catalog")

        # Insert books
        books_data = []
        for _, row in catalog_df.iterrows():
            book_id = int(row["book_id"])
            title = row["title"]
            description = row.get("description", "")
            infolink = row.get("infolink", None)
            books_data.append((book_id, title, description, infolink))

        cursor.executemany(
            "INSERT OR IGNORE INTO books (book_id, title, description, infolink) VALUES (?, ?, ?, ?)", books_data
        )
        self.conn.commit()
        logger.info(f"✓ Inserted {len(books_data)} books")

    def migrate_book_genres(self):
        """Build book_genres junction."""
        logger.info("Migrating book-genres relationships...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM book_genres")
        if cursor.fetchone()[0] > 0:
            logger.info("Book-genres already migrated, skipping")
            return

        # Load books catalog
        catalog_df = safe_read_feather(PATHS["clean_books"])

        # Build genre lookup
        cursor.execute("SELECT genre_id, name FROM genres")
        genre_lookup = {name: genre_id for genre_id, name in cursor.fetchall()}

        # Build book_genres junction
        book_genres_data = []
        for _, row in catalog_df.iterrows():
            book_id = int(row["book_id"])

            # Parse genres for this book
            genres_list = row.get("genres", [])
            if isinstance(genres_list, str):
                genres_list = ast.literal_eval(genres_list)
            if isinstance(genres_list, list):
                for genre_name in genres_list:
                    genre_id = genre_lookup.get(genre_name)
                    if genre_id:
                        book_genres_data.append((book_id, genre_id))

        # Insert book-genres relationships
        cursor.executemany("INSERT OR IGNORE INTO book_genres (book_id, genre_id) VALUES (?, ?)", book_genres_data)

        self.conn.commit()
        logger.info(f"✓ Inserted {len(book_genres_data)} book-genre relationships")

    def migrate_book_authors(self):
        """Build book_authors junction from catalog."""
        logger.info("Migrating book-authors relationships...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM book_authors")
        if cursor.fetchone()[0] > 0:
            logger.info("Book-authors already migrated, skipping")
            return

        # Load books catalog
        catalog_df = safe_read_feather(PATHS["clean_books"])

        # Build author lookup
        cursor.execute("SELECT author_id, name FROM authors")
        author_lookup = {name: author_id for author_id, name in cursor.fetchall()}

        # Build junction
        book_authors_data = []
        for _, row in catalog_df.iterrows():
            book_id = int(row["book_id"])
            authors_list = row.get("authors", [])

            if isinstance(authors_list, str):
                authors_list = ast.literal_eval(authors_list)

            if isinstance(authors_list, list):
                for author_name in authors_list:
                    author_id = author_lookup.get(author_name)
                    if author_id:
                        book_authors_data.append((book_id, author_id))

        # Insert relationships
        cursor.executemany("INSERT OR IGNORE INTO book_authors (book_id, author_id) VALUES (?, ?)", book_authors_data)

        self.conn.commit()
        logger.info(f"✓ Inserted {len(book_authors_data)} book-author relationships")

    def migrate_users(self):
        """Extract users from ratings DataFrame and create user records."""
        logger.info("Migrating users...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] > 0:
            logger.info("Users already migrated, skipping")
            return

        # Load cleaned ratings to extract unique users
        ratings_df = safe_read_feather(PATHS["clean_ratings"])
        
        # Extract unique users from ratings DataFrame (not database)
        unique_users = ratings_df["user_id"].unique().tolist()
        logger.info(f"Found {len(unique_users)} unique users in ratings")

        # Create user_id to profilename mapping (use first occurrence)
        user_name_map = {}
        for _, row in ratings_df.iterrows():
            user_id = str(row["user_id"])
            if user_id not in user_name_map:
                user_name_map[user_id] = str(row["profilename"])

        # Insert users with name from profilename mapping
        users_data = []
        for user_id in unique_users:
            user_id_str = str(user_id)
            user_name = user_name_map.get(user_id_str, user_id_str)
            users_data.append((user_id_str, user_name, 0))

        cursor.executemany("INSERT OR IGNORE INTO users (user_id, name, login_attempt) VALUES (?, ?, ?)", users_data)

        self.conn.commit()
        logger.info(f"✓ Inserted {len(users_data)} users")

    def migrate_ratings(self):
        """Load ratings data into database."""
        logger.info("Migrating ratings...")

        # Check if already migrated
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ratings")
        if cursor.fetchone()[0] > 0:
            logger.info("Ratings already migrated, skipping")
            return

        # Load ratings
        ratings_df = safe_read_feather(PATHS["clean_ratings"])
        logger.info(f"Loaded {len(ratings_df)} ratings")

        # Get valid books and users to filter ratings
        cursor.execute("SELECT book_id FROM books")
        valid_books = set(str(row[0]) for row in cursor.fetchall())

        cursor.execute("SELECT user_id FROM users")
        valid_users = set(str(row[0]) for row in cursor.fetchall())

        logger.info(f"Validating against {len(valid_books)} books and {len(valid_users)} users")

        # Build ratings data, filtering for valid references
        ratings_data = []
        skipped = 0

        for _, row in ratings_df.iterrows():
            user_id = str(row["user_id"])
            book_id = str(int(row["book_id"]))

            # Skip if book or user doesn't exist
            if book_id not in valid_books or user_id not in valid_users:
                skipped += 1
                continue

            ratings_data.append(
                (
                    user_id,
                    int(book_id),
                    float(row["review/score"]),
                    float(row["confidence"]),
                    row["datetime"],
                    str(row.get("review/summary", "")),
                    str(row.get("review/text", "")),
                )
            )

        logger.info(f"Inserting {len(ratings_data)} valid ratings (skipped {skipped})")

        cursor.executemany(
            """INSERT OR IGNORE INTO ratings 
               (user_id, book_id, score, confidence, datetime, review_summary, review_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ratings_data,
        )

        self.conn.commit()
        logger.info(f"✓ Inserted {len(ratings_data)} ratings")

    def print_migration_summary(self):
        """Print summary of migrated data."""
        cursor = self.conn.cursor()

        tables = [
            ("users", "Users"),
            ("genres", "Genres"),
            ("authors", "Authors"),
            ("books", "Books"),
            ("book_authors", "Book-Author relationships"),
            ("book_genres", "Book-Genre relationships"),
            ("ratings", "Ratings"),
            ("interactions", "Interactions"),
            ("user_genres", "User Genres"),
        ]

        logger.info("=" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 80)

        for table_name, label in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"{label:.<40} {count:>10,}")
            except sqlite3.OperationalError:
                logger.info(f"{label:.<40} {'(table missing)':>10}")

        logger.info("=" * 80)


def migrate_to_db():
    """Main migration function to be called from command line or app initialization."""
    with DataMigration() as migrator:
        migrator.migrate_all()
        migrator.print_migration_summary()
