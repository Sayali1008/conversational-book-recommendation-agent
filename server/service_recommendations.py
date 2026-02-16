import ast
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common.constants import *
from common.utils import load_pickle, safe_read_feather, setup_logging
from db.connection import get_db
from model.model_pipeline import *
from recommenders import IndexMappings, RecommendationContext
from recommenders.handler import get_recommendations

logger = setup_logging(__name__, PATHS["app_log_file"])


class RecommendationService:
    def __init__(self):
        self.ready: bool = False
        self.init_error: Optional[str] = None

        # Check that all required artifacts exist before loading
        required_files = [
            PATHS["clean_books"],
            PATHS["catalog_books_embeddings"],
            PATHS["user_factors"],
            PATHS["book_factors"],
            PATHS["train_matrix"],
            PATHS["user_idx_pkl"],
            PATHS["book_idx_pkl"],
        ]

        missing = [p for p in required_files if not Path(p).exists()]
        if missing:
            self.init_error = f"Missing required artifacts: {missing}"
            logger.warning(self.init_error)
            return

        try:
            # Load Data Artifacts
            catalog_df = safe_read_feather(PATHS["clean_books"])
            catalog_embeddings = np.load(PATHS["catalog_books_embeddings"])

            user_factors = np.load(PATHS["user_factors"])
            book_factors = np.load(PATHS["book_factors"])
            train_matrix = sp.load_npz(PATHS["train_matrix"])

            # Build Index Mappings
            user_id_to_cf = load_pickle(PATHS["user_idx_pkl"])
            cf_to_user_id = {v: k for k, v in user_id_to_cf.items()}

            book_id_to_cf = load_pickle(PATHS["book_idx_pkl"])
            cf_to_book_id = {v: k for k, v in book_id_to_cf.items()}

            # Build catalog index mappings
            book_id_to_catalog_id = {int(row.book_id): i for i, row in catalog_df.reset_index(drop=True).iterrows()}

            book_cf_to_catalog_id = {}
            for cf_idx, book_id in cf_to_book_id.items():
                if book_id in book_id_to_catalog_id:
                    book_cf_to_catalog_id[cf_idx] = book_id_to_catalog_id[book_id]

            # Build RecommendationContext (immutable data)
            index_mappings: IndexMappings = {
                "book_cf_to_catalog_id": book_cf_to_catalog_id,
                "user_id_to_cf": user_id_to_cf,
                "cf_to_user_id": cf_to_user_id,
                "book_id_to_cf": book_id_to_cf,
                "cf_to_book_id": cf_to_book_id,
                "book_id_to_catalog_id": book_id_to_catalog_id,
            }

            self.context: RecommendationContext = {
                "catalog_df": catalog_df,
                "catalog_embeddings": catalog_embeddings,
                "index_mappings": index_mappings,
                "train_matrix": train_matrix,
                "user_factors": user_factors,
                "book_factors": book_factors,
            }

            logger.info("✓ RecommendationService initialized successfully")
            self.ready = True

        except Exception as e:
            self.init_error = str(e)
            logger.error(f"Failed to initialize RecommendationService: {self.init_error}")
            self.ready = False

    def reinitialize(self):
        self.__init__()

    def get_book_details(self, book_id: int) -> Optional[Dict]:
        """Retrieve full details for a specific book by book_id."""
        if not self.ready:
            raise ValueError("Recommendation artifacts are not available")
        # book_id is 1-indexed, catalog is 0-indexed
        catalog_idx = book_id - 1

        if catalog_idx < 0 or catalog_idx >= len(self.context["catalog_df"]):
            logger.warning(f"Book ID {book_id} not found in catalog")
            return None

        row = self.context["catalog_df"].iloc[catalog_idx]

        # Verify this is actually the correct book
        if int(row["book_id"]) != book_id:
            logger.error(f"Index mismatch: expected book_id={book_id}, got {row['book_id']}")
            return None

        authors = row["authors"]
        if isinstance(authors, str):
            authors = ast.literal_eval(authors)
        elif not isinstance(authors, list):
            authors = []

        genres = row["genres"]
        if isinstance(genres, str):
            genres = ast.literal_eval(genres)
        elif not isinstance(genres, list):
            genres = []

        return {
            "book_id": book_id,
            "title": str(row["title"]),
            "authors": authors,
            "description": str(row["description"]) if "description" in row else "",
            "genres": genres,
            "infolink": str(row["infolink"]) if "infolink" in row and row["infolink"] else None,
        }

    def get_book_details_from_db(self, book_id: int) -> Optional[Dict]:
        """Retrieve full details for a specific book from the database.
        
        This queries the actual database tables and is useful for newly added books
        that may not be in the recommendation catalog yet.
        """
        try:
            db = get_db()
            conn = db.get_connection()
            cursor = conn.cursor()

            # Get book info
            cursor.execute(
                "SELECT book_id, title, description, infolink FROM books WHERE book_id = ?",
                (book_id,)
            )
            book_row = cursor.fetchone()
            
            if not book_row:
                logger.warning(f"Book ID {book_id} not found in database")
                return None

            # Get authors
            cursor.execute(
                """
                SELECT a.name FROM authors a
                JOIN book_authors ba ON a.author_id = ba.author_id
                WHERE ba.book_id = ?
                ORDER BY a.name
                """,
                (book_id,)
            )
            author_rows = cursor.fetchall()
            authors = [row[0] for row in author_rows]

            # Get genres
            cursor.execute(
                """
                SELECT g.name FROM genres g
                JOIN book_genres bg ON g.genre_id = bg.genre_id
                WHERE bg.book_id = ?
                ORDER BY g.name
                """,
                (book_id,)
            )
            genre_rows = cursor.fetchall()
            genres = [row[0] for row in genre_rows]

            conn.close()

            return {
                "book_id": int(book_row[0]),
                "title": str(book_row[1]),
                "description": str(book_row[2]) if book_row[2] else "",
                "infolink": str(book_row[3]) if book_row[3] else None,
                "authors": authors,
                "genres": genres,
            }

        except Exception as e:
            logger.error(f"Error retrieving book details from database: {str(e)}")
            return None

    def get_user_cf(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index."""
        if not self.ready:
            raise ValueError("Recommendation artifacts are not available")
        if not user_id:
            return None
        return self.context["index_mappings"]["user_id_to_cf"].get(user_id)

    def user_has_history(self, user_cf: Optional[int]) -> bool:
        """Check if user has interaction history in training data."""
        return user_cf is not None and self.context["train_matrix"][user_cf].nnz > 0

    def book_ids_to_catalog_indices(self, items) -> List[int]:
        """Convert book IDs or interaction records to catalog row indices."""
        if not items:
            return []

        mappings = self.context["index_mappings"]
        book_id_to_catalog_id = mappings["book_id_to_catalog_id"]
        catalog_indices = []

        for item in items:
            # Handle both interaction records (with .book_id attribute) and raw book IDs
            book_id = item["book_id"] if isinstance(item, dict) or hasattr(item, "__getitem__") else item
            if book_id in book_id_to_catalog_id:
                catalog_indices.append(book_id_to_catalog_id[book_id])

        return catalog_indices

    def build_genre_based_profile(self, user_id: str, top_k_per_genre: int = 5) -> Optional[np.ndarray]:
        """
        Build a user profile embedding from their preferred genres.

        Steps:
        1. Fetch user's preferred genres
        2. Find popular books for each genre (by rating count)
        3. Get embeddings for those books
        4. Return weighted average embedding

        Returns None if user has no genres or if embedding cannot be built.
        """
        try:
            db = get_db()
            conn = db.get_connection()

            try:
                cursor = conn.cursor()
                cursor.execute("SELECT genre_id FROM user_genres WHERE user_id = ?", (user_id,))
                genre_rows = cursor.fetchall()
                genres = [row[0] for row in genre_rows]

                if not genres:
                    logger.info(f"[GB] User {user_id} has no genre preferences")
                    return None

                all_seed_book_ids = []
                for genre_id in genres:
                    cursor.execute(
                        """
                        SELECT bg.book_id, COUNT(r.book_id) as rating_count
                        FROM book_genres bg
                        LEFT JOIN ratings r ON bg.book_id = r.book_id
                        WHERE bg.genre_id = ?
                        GROUP BY bg.book_id
                        ORDER BY rating_count DESC
                        LIMIT ?
                        """,
                        (genre_id, top_k_per_genre),
                    )
                    genre_books = cursor.fetchall()
                    all_seed_book_ids.extend([row[0] for row in genre_books])

                if not all_seed_book_ids:
                    logger.info(f"[GB] No books found for user {user_id}'s genres")
                    return None

                book_id_to_catalog_id = self.context["index_mappings"]["book_id_to_catalog_id"]
                catalog_indices = []
                for book_id in all_seed_book_ids:
                    if book_id in book_id_to_catalog_id:
                        catalog_indices.append(book_id_to_catalog_id[book_id])

                if not catalog_indices:
                    logger.info(f"[GB] No catalog indices found for user {user_id}'s genre books")
                    return None

                logger.info(f"[GB] Catalog indices: {catalog_indices}")

                catalog_embeddings = self.context["catalog_embeddings"]
                logger.info(f"[REC] catalog embeddings {len(self.context['catalog_embeddings'])}")

                genre_embeddings = catalog_embeddings[catalog_indices]
                logger.info(f"[REC] genre embeddings {len(genre_embeddings)}")

                user_profile = genre_embeddings.mean(axis=0)
                logger.info(f"[REC] user profile {len(user_profile)}")

                logger.info(f"[GB] Built genre-based profile for user {user_id} from {len(catalog_indices)} books")
                return user_profile

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error building genre profile: {str(e)}")
            return None

    def build_interaction_based_profile(self, swiped_books: list) -> Optional[np.ndarray]:
        """Build a user profile embedding from books they have already interacted with."""
        try:
            if not swiped_books:
                return None

            like_boost = 1.0
            dislike_penalty = 0.5

            book_id_to_catalog_id = self.context["index_mappings"]["book_id_to_catalog_id"]
            catalog_embeddings = self.context["catalog_embeddings"]

            # Separate likes and dislikes
            liked_ids = [b["book_id"] for b in swiped_books if b["action"] == "like"]
            disliked_ids = [b["book_id"] for b in swiped_books if b["action"] == "dislike"]

            # Get liked embeddings
            liked_indices = [book_id_to_catalog_id[bid] for bid in liked_ids if bid in book_id_to_catalog_id]
            liked_profile = (
                catalog_embeddings[liked_indices].mean(axis=0)
                if liked_indices
                else np.zeros(catalog_embeddings.shape[1])
            )

            # Get disliked embeddings
            disliked_indices = [book_id_to_catalog_id[bid] for bid in disliked_ids if bid in book_id_to_catalog_id]
            disliked_profile = (
                catalog_embeddings[disliked_indices].mean(axis=0)
                if disliked_indices
                else np.zeros(catalog_embeddings.shape[1])
            )

            # Compute profile with negative feedback
            user_profile = (liked_profile * like_boost) - (dislike_penalty * disliked_profile)
            user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)  # Normalize

            logger.debug(
                f"[IB] Built interaction profile from {len(liked_indices)} likes and {len(disliked_indices)} dislikes."
            )
            return user_profile

        except Exception as e:
            logger.error(f"Error building interaction profile: {str(e)}")
            return None

    def recommend(self, user_id: Optional[str] = None, top_k=10, swiped_books=None):
        if not self.ready:
            raise ValueError("Recommendation artifacts are not available")

        user_cf = self.get_user_cf(user_id)
        in_matrix = self.user_has_history(user_cf)
        has_interactions = swiped_books is not None and len(swiped_books) > 0

        # "Warm" means they are in the matrix. "Semi-warm" means they have history we can use for Item-to-Item logic.
        is_warm = in_matrix
        is_semi_warm = not in_matrix and has_interactions
        best_rec_params = load_pickle(PATHS["best_rec_params"])

        logger.info(
            f"Generating recommendations for {'warm' if is_warm else 'cold'} user {f'({user_cf})' if user_cf else ''} using rec params: {best_rec_params}"
        )

        # Build user profile based on state
        user_profile = None
        if not is_warm:
            if is_semi_warm:
                # Item-to-Item logic: Build profile from their actual interactions
                user_profile = self.build_interaction_based_profile(swiped_books)
                logger.info(f"Using Item-to-Item logic for semi-warm user {user_id}")
            elif user_id:
                # Cold start logic: Build profile from onboarding genres
                user_profile = self.build_genre_based_profile(user_id)
                logger.info(f"Using Genre-based logic built user profile with shape {user_profile.shape}")

        # Get recommendations based on user state (warm/semi-warm/cold)
        indices, scores = get_recommendations(
            context=self.context,
            user_cf=user_cf,
            candidate_pool_size=best_rec_params["candidate_pool_size"],
            lambda_weight=best_rec_params["lambda"],
            is_warm_user=is_warm,
            top_k=top_k,
            swiped_books=swiped_books,
            user_profile=user_profile,
        )

        # Determine strategy used
        strategy = "hybrid" if is_warm and len(indices) > 0 else "content_based"
        logger.info(f"Returned {len(indices)} recommendations using strategy: {strategy}")

        # Format results with deduplication
        recs = []
        seen_catalog_ids = set()
        catalog_ids_log = []

        for idx, score in zip(indices, scores):
            if idx in seen_catalog_ids:
                logger.warning(f"Duplicate catalog_idx={idx} detected, skipping")
                continue

            seen_catalog_ids.add(idx)
            catalog_ids_log.append(idx)

            row = self.context["catalog_df"].iloc[idx]

            authors = row["authors"]
            if isinstance(authors, str):
                authors = ast.literal_eval(authors)
            elif not isinstance(authors, list):
                authors = []

            recs.append(
                {
                    "book_id": int(row["book_id"]),
                    "catalog_idx": int(idx),
                    "title": row["title"],
                    "authors": authors,
                    "score": float(score),
                }
            )

        logger.info(f"Final recommendations (after deduplication): {len(recs)} unique books")

        return recs, strategy
