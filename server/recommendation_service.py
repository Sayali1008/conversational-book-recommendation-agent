import ast
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from common.constants import *
from common.utils import *
from recommenders import EvalConfig, IndexMappings, RecommendationContext, recommend

logger = setup_logging(__name__, APP_LOG_FILE)


class RecommendationService:
    """Service for generating book recommendations."""

    def __init__(self):
        """Load all artifacts once at startup, build context and config."""
        logger.info(f"Initializing RecommendationService...")

        # ===================================================================
        # Load Data Artifacts
        # ===================================================================
        catalog_df = safe_read_feather(PATHS["clean_books_path"])
        catalog_embeddings = np.load(PATHS["catalog_books_embeddings_path"])

        user_factors = np.load(PATHS["user_factors_path"])
        book_factors = np.load(PATHS["book_factors_path"])
        train_matrix = sp.load_npz(PATHS["train_matrix_path"])

        # Scale book factors to address tiny raw values (mean=0.003)
        # book_factors = book_factors * CF_MODEL_PARAMS["book_factor_scale"]

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
        index_mappings = IndexMappings = {
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

        self.config: EvalConfig = {
            "norm": RECOMMEND["norm"],
            "norm_metadata": RECOMMEND["norm_metadata"],
            "lambda_weight": RECOMMEND["lambda_weight"],
            "k": RECOMMEND["k"],
            "candidate_pool_size": RECOMMEND["candidate_pool_size"],
            "filter_rated": RECOMMEND["filter_rated"],
        }

        logger.info("✓ RecommendationService initialized successfully")

    def get_user_idx(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index."""
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

    def recommend(
        self,
        user_cf_idx: Optional[int],
        k: int = 10,
        seed_catalog_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Dict], str]:
        """Generate recommendations for a user."""
        logger.info(f"in recommend()")

        warm = self.user_has_history(user_cf_idx)
        logger.info(f"Generating {k} recommendations for user_idx={user_cf_idx}, is_warm={warm}")

        # Convert seed list to numpy array if provided
        seed_catalog_array = None
        if seed_catalog_indices:
            seed_catalog_array = np.array(seed_catalog_indices)

        # Get recommendations
        indices, scores, sources = recommend(
            context=self.context,
            config=self.config,
            user_idx=user_cf_idx,
            is_warm_user=warm,
            k=k,
            seed_catalog_indices=seed_catalog_array,
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
