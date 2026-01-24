import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from common.constants import PATHS, RECOMMEND
from common.utils import setup_logging, safe_read_feather, load_pickle
from recommenders import RecommendationConfig, IndexMappings, RecommendationContext, recommend

logger = setup_logging(__name__, PATHS["app_log_file"])


class RecommendationService:
    """Service for generating book recommendations."""

    def __init__(self):
        """Load all artifacts once at startup, build context and config."""
        logger.info(f"Initializing RecommendationService...")

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
            # ===================================================================
            # Load Data Artifacts
            # ===================================================================
            catalog_df = safe_read_feather(PATHS["clean_books"])
            catalog_embeddings = np.load(PATHS["catalog_books_embeddings"])

            user_factors = np.load(PATHS["user_factors"])
            book_factors = np.load(PATHS["book_factors"])
            train_matrix = sp.load_npz(PATHS["train_matrix"])

            logger.info(
                f"Loaded artifacts: user_factors={user_factors.shape}, embeddings={catalog_embeddings.shape}, catalog={catalog_df.shape}"
            )

            # ===================================================================
            # Build Index Mappings
            # ===================================================================
            user_to_cf_idx = load_pickle(PATHS["user_idx_pkl"])
            cf_idx_to_user = {v: k for k, v in user_to_cf_idx.items()}

            book_to_cf_idx = load_pickle(PATHS["book_idx_pkl"])
            cf_idx_to_book = {v: k for k, v in book_to_cf_idx.items()}

            # Build catalog index mappings
            book_id_to_catalog_idx = {int(row.book_id): i for i, row in catalog_df.reset_index(drop=True).iterrows()}

            cf_idx_to_catalog_id = {}
            for cf_idx, book_id in cf_idx_to_book.items():
                if book_id in book_id_to_catalog_idx:
                    cf_idx_to_catalog_id[cf_idx] = book_id_to_catalog_idx[book_id]

            logger.info(f"Built index mappings: {len(user_to_cf_idx)} users, {len(cf_idx_to_book)} books in CF")

            # Build RecommendationContext (immutable data)
            index_mappings: IndexMappings = {
                "cf_idx_to_catalog_id": cf_idx_to_catalog_id,
                "user_to_cf_idx": user_to_cf_idx,
                "cf_idx_to_user": cf_idx_to_user,
                "cf_idx_to_book": cf_idx_to_book,
                "book_id_to_catalog_idx": book_id_to_catalog_idx,
            }

            self.context: RecommendationContext = {
                "user_factors": user_factors,
                "book_factors": book_factors,
                "train_matrix": train_matrix,
                "catalog_embeddings": catalog_embeddings,
                "index_mappings": index_mappings,
                "catalog_df": catalog_df,
            }

            self.config: RecommendationConfig = {
                "norm": RECOMMEND["norm"],
                "norm_metadata": RECOMMEND["norm_metadata"],
                "lambda_weight": RECOMMEND["lambda_weight"],
                "k": RECOMMEND["k"],
                "candidate_pool_size": RECOMMEND["candidate_pool_size"],
                "filter_rated": RECOMMEND["filter_rated"],
                "recency_boost": RECOMMEND["recency_boost"],
            }

            logger.info("✓ RecommendationService initialized successfully")
            self.ready = True

        except Exception as e:
            self.init_error = str(e)
            logger.error(f"Failed to initialize RecommendationService: {self.init_error}")
            self.ready = False

    def reinitialize(self):
        """Re-attempt to initialize after artifacts may have been generated."""
        logger.info("Attempting to reinitialize RecommendationService...")
        self.__init__()

    def status(self) -> Dict[str, any]:
        """Return readiness status and any initialization errors."""
        missing = []
        required_files = [
            PATHS["clean_books"],
            PATHS["catalog_books_embeddings"],
            PATHS["user_factors"],
            PATHS["book_factors"],
            PATHS["train_matrix"],
            PATHS["user_idx_pkl"],
            PATHS["book_idx_pkl"],
        ]
        for p in required_files:
            if not Path(p).exists():
                missing.append(p)

        return {
            "ready": self.ready,
            "missing_artifacts": missing,
            "error": self.init_error,
        }

    def get_user_idx(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index."""
        if not self.ready:
            raise ValueError("Recommendation artifacts are not available")
        if not user_id:
            return None
        return self.context["index_mappings"]["user_to_cf_idx"].get(user_id)

    def user_has_history(self, user_cf_idx: Optional[int]) -> bool:
        """Check if user has interaction history in training data."""
        return user_cf_idx is not None and self.context["train_matrix"][user_cf_idx].nnz > 0

    def book_ids_to_catalog_indices(self, seed_book_ids: Optional[List[int]]) -> List[int]:
        """Convert book IDs to catalog row indices."""
        if not seed_book_ids:
            return []
        mappings = self.context["index_mappings"]
        return [mappings["book_id_to_catalog_idx"][b] for b in seed_book_ids if b in mappings["book_id_to_catalog_idx"]]

    def recommend(self, user_cf_idx, k=10, exclude_catalog_rows=None):
        if not self.ready:
            raise ValueError("Recommendation artifacts are not available")
        logger.info(f"in recommend()")

        warm = self.user_has_history(user_cf_idx)
        logger.info(f"Generating {k} recommendations for user_idx={user_cf_idx}, is_warm={warm}")

        # Convert exclusion list to numpy array if provided
        exclude_catalog_array = None
        if exclude_catalog_rows:
            exclude_catalog_array = np.array(exclude_catalog_rows)

        # Get recommendations
        indices, scores, sources = recommend(
            context=self.context,
            config=self.config,
            user_idx=user_cf_idx,
            is_warm_user=warm,
            k=k,
            exclude_catalog_rows=exclude_catalog_array,
        )

        # Determine strategy used
        if warm:
            strategy = "warm_hybrid" if len(indices) > 0 else "cold_embed"
        else:
            strategy = "cold_embed"

        logger.info(f"Returned {len(indices)} recommendations using strategy: {strategy}")

        # Format results with deduplication
        recs = []
        seen_catalog_ids = set()
        catalog_ids_log = []

        for idx, score, src in zip(indices, scores, sources):
            # Skip duplicates based on catalog index
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
                    "source": str(src),
                }
            )

        logger.info(f"Final recommendations (after deduplication): {len(recs)} unique books")
        logger.info(f"Catalog IDs: {catalog_ids_log}")

        return recs, strategy

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
