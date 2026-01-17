import sys
import os
from pathlib import Path

# Add parent directory to path so we can import config, utils, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List

from server.schemas import RecommendResponse, SwipeRequest, SwipeResponse, BookRecommendation
from server.recommendation_service import RecommendationService
from server.storage import Storage
from server.user_registry import UserRegistry
from config import DATA_DIR

app = FastAPI(title="Book Recommender API", version="0.1.0")

# Add CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RecommendationService()
storage = Storage(str(DATA_DIR / "server.db"))
users = UserRegistry()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: Optional[str] = None,
    k: int = 10,
    seed_book_ids: Optional[str] = None,  # comma-separated book_id list
):
    try:
        print(f"user_id:{user_id} | type(user_id): {type(user_id)}")
        user_cf_idx = users.get_user_cf_idx(user_id)
        print(f"user_idx: {user_cf_idx} type: {user_cf_idx}")

        seeds = []
        if seed_book_ids:
            seeds = [int(s) for s in seed_book_ids.split(",") if s.strip()]
        seed_catalog_indices = service.book_ids_to_catalog_indices(seeds)
        recs, strategy = service.recommend(user_cf_idx=user_cf_idx, k=k, seed_catalog_indices=seed_catalog_indices)
        return RecommendResponse(
            recommendations=[BookRecommendation(**r) for r in recs],
            strategy=strategy,
            used_seeds=seed_catalog_indices,
        )
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in /recommend: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swipe", response_model=SwipeResponse)
def swipe(payload: SwipeRequest):
    try:
        storage.log_swipe(payload.user_id, payload.book_id, payload.action, payload.confidence)
        # Optionally prefetch a small next batch for snappy UX
        user_idx = users.get_user_cf_idx(payload.user_id)
        recs, _ = service.recommend(user_cf_idx=user_idx, k=5, seed_catalog_indices=None)
        return SwipeResponse(status="ok", next_recommendations=[BookRecommendation(**r) for r in recs])
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in /swipe: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))
