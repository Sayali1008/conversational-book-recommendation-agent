import numpy as np
import scipy.sparse as sp
from typing import Optional

from config import *
from utils import load_pickle, safe_read_feather
import _hybrid_recommender


class RecommendationService:
    def __init__(self):
        # Load artifacts once
        self.catalog_df = safe_read_feather(OUTPUT_BOOKS)
        self.catalog_embeddings = np.load(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)
        
        # Load mappings from training stage and create reverse mappings
        self.user_to_cf_idx = load_pickle(USER_IDX_PKL)
        self.cf_idx_to_user = {v: k for k, v in self.user_to_cf_idx.items()}
        
        book_id_to_cf_idx = load_pickle(BOOK_IDX_PKL)
        self.cf_idx_to_book = {v: k for k, v in book_id_to_cf_idx.items()}
        
        # Build reverse mapping: user_id -> cf_idx
        # self.user_to_cf_idx = {v: k for k, v in self.cf_idx_to_user.items()}
        
        # Load model artifacts
        self.user_factors = np.load(OUTPUT_USER_FACTORS)
        self.book_factors = np.load(OUTPUT_BOOK_FACTORS)

        # Scale book factors 10x to address tiny raw factor values (mean=0.003)
        self.book_factors = self.book_factors * 10.0

        self.train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)
        
        # Build catalog mappings
        self.book_id_to_catalog_idx = {
            int(row.book_id): i for i, row in self.catalog_df.reset_index(drop=True).iterrows()
        }
        
        # Build cf_idx to catalog_idx mapping
        self.cf_idx_to_catalog_id_map = {}
        for cf_idx, book_id in self.cf_idx_to_book.items():
            if book_id in self.book_id_to_catalog_idx:
                self.cf_idx_to_catalog_id_map[cf_idx] = self.book_id_to_catalog_idx[book_id]
        
        print("✓ RecommendationService loaded successfully")

    def get_user_idx(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index."""
        if not user_id:
            return None
        return self.user_to_cf_idx.get(user_id)

    def user_has_history(self, user_cf_idx: Optional[int]) -> bool:
        return user_cf_idx is not None and self.train_matrix[user_cf_idx].nnz > 0

    def book_ids_to_catalog_indices(self, seed_book_ids: Optional[list[int]]) -> list[int]:
        if not seed_book_ids:
            return []
        return [self.book_id_to_catalog_idx[b] for b in seed_book_ids if b in self.book_id_to_catalog_idx]

    def recommend(
        self,
        user_cf_idx: Optional[int],
        k: int = 10,
        seed_catalog_indices: Optional[list[int]] = None,
        lambda_weight: float = 0.65,
    ) -> tuple[list[dict], str]:
        print("recommed: finding recommendations..")

        warm = self.user_has_history(user_cf_idx)
        print(f"user has history: {warm}")
        indices, scores, sources = _hybrid_recommender.recommend_with_cold_start(
            user_idx=user_cf_idx,
            is_warm_user=warm,
            user_factors=self.user_factors,
            book_factors=self.book_factors,
            train_matrix=self.train_matrix,
            catalog_embeddings=self.catalog_embeddings,
            cf_idx_to_catalog_id_map=self.cf_idx_to_catalog_id_map,
            cf_idx_to_book=self.cf_idx_to_book,
            k=k,
            lambda_weight=HYBRID_LAMBDA_WEIGHT,
            candidate_pool_size=HYBRID_CANDIDATE_POOL_SIZE,
            filter_rated=HYBRID_FILTER_RATED,
            seed_catalog_indices=seed_catalog_indices,
            norm=HYBRID_NORM,
            norm_metadata=HYBRID_NORM_METADATA
        )
        print(f"indices from hybrid={indices}, sources={sources}")

        strategy = "warm_hybrid" if warm and len(indices) else "cold_embed"
        print(f"strategy: {strategy}")
        if warm and seed_catalog_indices and len(indices):
            strategy = "mixed"

        recs = []
        for idx, score, src in zip(indices, scores, sources):
            row = self.catalog_df.iloc[idx]
            recs.append(
                {
                    "book_id": int(row["book_id"]),
                    "title": row["title"],
                    "score": float(score),
                    "source": str(src),
                }
            )
        return recs, strategy