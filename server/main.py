import traceback
from typing import Optional
import threading
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from common.constants import PATHS
from common.utils import setup_logging
from server.recommendation_service import RecommendationService
from server.schemas import BookRecommendation, BookDetails, RecommendResponse, SwipeRequest, SwipeResponse
from server.storage import Interactions
from ml_pipeline import handler as pipeline_handler

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
idb = Interactions(PATHS["database"])
logger = setup_logging(__name__, PATHS["app_log_file"])

# Simple pipeline state tracking
pipeline_state_lock = threading.Lock()
pipeline_state = {"status": "idle", "current_stage": None, "overall_progress": 0, "error": None, "pipeline_id": None}


def update_pipeline_state(
    status: str = None,
    current_stage: str = None,
    overall_progress: int = None,
    error: str = None,
    pipeline_id: str = None,
):
    """Thread-safe update of pipeline state."""
    global pipeline_state
    with pipeline_state_lock:
        if status is not None:
            pipeline_state["status"] = status
        if current_stage is not None:
            pipeline_state["current_stage"] = current_stage
        if overall_progress is not None:
            pipeline_state["overall_progress"] = overall_progress
        if error is not None:
            pipeline_state["error"] = error
        if pipeline_id is not None:
            pipeline_state["pipeline_id"] = pipeline_id


def get_pipeline_state():
    """Thread-safe read of pipeline state."""
    with pipeline_state_lock:
        return dict(pipeline_state)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "recommendations_ready": service.ready,
        "error": service.init_error,
    }


@app.get("/book/{book_id}", response_model=BookDetails)
def get_book_details(book_id: int):
    """
    Retrieve full details for a specific book.

    Called when user clicks on a book card to view detailed information.
    Returns: title, authors, description, genres, and info link.
    """
    try:
        if not service.ready:
            raise HTTPException(
                status_code=503, detail="Recommendation artifacts are not available. Run the pipeline to generate them."
            )

        details = service.get_book_details(book_id)
        if details is None:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        return BookDetails(**details)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /book/{book_id}: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendation/status")
def recommendation_status():
    """Report readiness of recommendation artifacts."""
    return service.status()


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: Optional[str] = None, k: int = 10, seed_book_ids: Optional[str] = None):
    try:
        if not service.ready:
            raise HTTPException(
                status_code=503, detail="Recommendation artifacts are not available. Run the pipeline to generate them."
            )

        user_cf = service.get_user_cf(user_id)

        # Exclude any books the user has previously swiped (avoid repeats)
        swiped_books = []
        if user_id:
            swiped_books = idb.get_user_swiped_books(user_id)

        # Seeds from query (positive intent)
        seed_ids = []
        if seed_book_ids:
            seed_ids = [int(s) for s in seed_book_ids.split(",") if s.strip()]

        recs, strategy = service.recommend(
            user_cf=user_cf,
            k=k,
            seed_book_ids=seed_ids,
            swiped_books=swiped_books,
        )
        return RecommendResponse(recommendations=[BookRecommendation(**r) for r in recs], strategy=strategy)
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /recommend: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swipe", response_model=SwipeResponse)
def swipe(payload: SwipeRequest):
    try:
        if not service.ready:
            raise HTTPException(
                status_code=503, detail="Recommendation artifacts are not available. Run the pipeline to generate them."
            )

        # Normalize confidence based on action
        if payload.action == "like":
            confidence = 1.0
        else:  # payload.action == "dislike":
            confidence = 0.0

        logger.info(
            f"Swipe: user={payload.user_id}, book={payload.book_id}, action={payload.action}, confidence={confidence}"
        )

        # Log interaction to persistent storage
        idb.insert_swipe(payload.user_id, payload.book_id, payload.action, confidence)

        # Generate fresh recommendations excluding all recent swipes
        user_idx = service.get_user_cf(payload.user_id)

        exclude_catalog_indices = None
        all_swiped = idb.get_user_swiped_books(payload.user_id)
        if all_swiped:
            exclude_catalog_indices = service.book_ids_to_catalog_indices(list(all_swiped))
            logger.info(f"Excluding {len(exclude_catalog_indices)} swiped items (DB) from recommendations")

        # Generate next batch of recommendations (full replacement, not prefetch)
        # Use k from payload to match user's slider setting
        # No seed_catalog_indices needed since they would be excluded anyway
        # Prefer passing book IDs as exclusions; we already have them
        exclude_book_ids = list(all_swiped) if all_swiped else []
        recs, strategy = service.recommend(
            user_cf=user_idx,
            k=payload.k,
            swiped_books=exclude_book_ids,
        )

        filtered_recs = [r for r in recs if int(r["book_id"]) != payload.book_id]

        logger.info(
            f"Generated {len(recs)} recommendations, {len(filtered_recs)} after removing currently swiped book, strategy={strategy}"
        )

        # Return response with replacement batch (clears old recommendations)
        return SwipeResponse(status="ok", next_recommendations=[BookRecommendation(**r) for r in filtered_recs])

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /swipe: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# PIPELINE ENDPOINTS
# ===================================================================


@app.get("/pipeline/status")
def get_pipeline_status():
    """
    Get current pipeline execution status and progress.

    Returns simplified status with overall progress and current stage.
    """
    try:
        return get_pipeline_state()
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /pipeline/status: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


def _run_pipeline_background():
    """Execute the full ML pipeline in background with progress updates."""
    try:
        logger.info("Starting full pipeline execution...")

        # Define stages with human-readable names
        stages = [
            ("stage_1_preprocessing", "Data Preprocessing", pipeline_handler.run_stage_1_preprocessing, 0),
            ("stage_2_embeddings", "Generating Embeddings", pipeline_handler.run_stage_2_embeddings, 20),
            ("stage_3_matrices", "Building Matrices", pipeline_handler.run_stage_3_matrices, 40),
            ("stage_4_training", "Training Model", pipeline_handler.run_stage_4_training, 60),
            ("stage_5_evaluation", "Evaluation", pipeline_handler.run_stage_5_evaluation, 80),
        ]

        for stage_id, stage_name, stage_func, progress_start in stages:
            try:
                logger.info(f"Executing {stage_id}...")
                update_pipeline_state(current_stage=stage_name, overall_progress=progress_start)
                stage_func()
                logger.info(f"✓ {stage_id} completed")
            except Exception as e:
                error_msg = f"Stage {stage_id} failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                update_pipeline_state(status="failed", error=str(e))
                raise

        # Pipeline completed successfully
        update_pipeline_state(status="completed", overall_progress=100, current_stage="All stages complete")
        logger.info("✓ Full pipeline completed successfully")

        # Reinitialize recommendation service with newly generated artifacts
        service.reinitialize()
        logger.info("✓ RecommendationService reinitialized with new artifacts")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}")


@app.post("/pipeline/run")
def start_pipeline():
    """Start the full ML pipeline in background."""
    try:
        state = get_pipeline_state()
        if state["status"] == "running":
            raise HTTPException(status_code=409, detail="Pipeline is already running")

        pipeline_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        update_pipeline_state(status="running", overall_progress=0, current_stage="stage_1_preprocessing", error=None)
        update_pipeline_state(pipeline_id=pipeline_id)

        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=_run_pipeline_background, daemon=True)
        pipeline_thread.start()

        logger.info(f"Pipeline {pipeline_id} started in background")

        return {
            "pipeline_id": pipeline_id,
            "status": "running",
            "message": "Pipeline started, executing stages...",
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /pipeline/run: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))
