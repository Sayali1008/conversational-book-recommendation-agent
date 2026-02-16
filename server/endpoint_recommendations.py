from datetime import datetime

from fastapi import HTTPException

from common.constants import *
from common.utils import setup_logging
from db import Interactions
from db.connection import get_db
from server.schemas import *
from server.service_recommendations import RecommendationService

logger = setup_logging(__name__, PATHS["app_log_file"])


def recommend(service: RecommendationService, idb: Interactions, payload):
    if not service.ready:
        raise HTTPException(
            status_code=503, detail="Recommendation artifacts are not available. Run the pipeline to generate them."
        )

    logger.info(f"Called /recommend API with params: {payload['user_id']}, {payload['k']}")

    # Exclude any books the user has previously swiped (avoid repeats)
    swiped_books = []
    if payload["user_id"]:
        swiped_books = idb.get_user_swiped_books(payload["user_id"])

    recs, strategy = service.recommend(
        user_id=payload["user_id"],
        top_k=payload["k"],

        swiped_books=swiped_books,
    )
    return RecommendResponse(recommendations=[BookRecommendation(**r) for r in recs], strategy=strategy)


def swipe(service: RecommendationService, idb: Interactions, payload: SwipeRequest):
    if not service.ready:
        raise HTTPException(
            status_code=503, detail="Recommendation artifacts are not available. Run the pipeline to generate them."
        )

    # Normalize confidence based on action
    if payload.action == "like":
        confidence = 1.0
    else:  # payload.action == "dislike":
        confidence = 0.0

    logger.info(f"Swipe: user={payload.user_id}, book={payload.book_id}, action={payload.action}")

    # Log interaction to persistent storage
    idb.insert_swipe(payload.user_id, payload.book_id, payload.action, confidence)

    # Fetch all swiped books from database (used to build dynamic user profile)
    all_swiped = idb.get_user_swiped_books(payload.user_id)
    
    # Generate next batch of recommendations (profile built from all swipes)
    recs, strategy = service.recommend(user_id=payload.user_id, top_k=payload.k, swiped_books=list(all_swiped) if all_swiped else [])
    filtered_recs = [r for r in recs if int(r["book_id"]) != payload.book_id]

    logger.info(f"Strategy: {strategy} | Generated {len(filtered_recs)} recs after removing currently swiped book")

    # Return response with replacement batch (clears old recommendations)
    return SwipeResponse(status="ok", next_recommendations=[BookRecommendation(**r) for r in filtered_recs])
