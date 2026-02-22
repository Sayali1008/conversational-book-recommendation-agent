import threading
import traceback
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

import server.endpoint_authentication as auth
import server.endpoint_genre as genre
import server.endpoint_recommendations as rec
from common.constants import *
from common.utils import setup_logging
from db import Interactions
from model import handler as model_handler
from server.schemas import *
from server.service_recommendations import RecommendationService
from setup_database import handler as database_handler

app = FastAPI(title="Book Recommender API", version="0.1.0")

# Add CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
db = database_handler.run_data_pipeline()
model = model_handler.run_model_pipeline()

service = RecommendationService()
idb = Interactions()

logger = setup_logging(__name__, PATHS["app_log_file"])

pipeline_state_lock = threading.Lock()
pipeline_state = {"status": "idle", "current_stage": None, "overall_progress": 0, "error": None, "pipeline_id": None}


@app.get("/status")
def recommendation_status():
    logger.info(f"Retrieving recommendation status..")
    return {"ready": service.ready}


# region AUTHENTICATION ENDPOINTS
@app.post("/users", response_model=CreateUserResponse)
def create_user(payload: CreateUserRequest):
    try:
        response = auth.create_user(payload)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login", response_model=LoginResponse)
def login_user(payload: LoginRequest):
    try:
        response = auth.login_user(payload)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
def check_user_exists(user_id: str):
    """Check if a user ID exists (for registration form validation)."""
    try:
        response = auth.check_user_exists(user_id)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking user: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region GENRE ENDPOINTS
@app.get("/genres", response_model=GenresResponse)
def get_genres():
    """Get all available genres."""
    try:
        response = genre.get_genres()
        return response
    except Exception as e:
        logger.error(f"Error fetching genres: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/genres", response_model=CreateGenreResponse)
def create_genre(payload: CreateGenreRequest):
    """Create a new genre."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        # Check if genre already exists
        cursor.execute("SELECT genre_id FROM genres WHERE name = ?", (payload.name,))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            raise HTTPException(status_code=409, detail=f"Genre '{payload.name}' already exists")

        # Insert new genre
        cursor.execute("INSERT INTO genres (name) VALUES (?)", (payload.name,))
        conn.commit()

        # Get the created genre ID
        genre_id = cursor.lastrowid
        conn.close()

        logger.info(f"Genre created: {genre_id} - {payload.name}")
        return CreateGenreResponse(genre_id=genre_id, name=payload.name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating genre: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preferred-genres", response_model=UserGenresResponse)
def set_user_genres(payload: UserGenresRequest):
    """Save user's preferred genres."""
    try:
        response = genre.set_user_genres(payload)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving user genres: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region AUTHOR ENDPOINTS
@app.get("/authors", response_model=AuthorsResponse)
def get_authors(q: Optional[str] = None):
    """Get all authors, optionally filtered by search query."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        if q:
            cursor.execute("SELECT author_id, name FROM authors WHERE name LIKE ? ORDER BY name ASC", (f"%{q}%",))
        else:
            cursor.execute("SELECT author_id, name FROM authors ORDER BY name ASC")

        authors_data = cursor.fetchall()
        conn.close()

        authors = [Author(author_id=row["author_id"], name=row["name"]) for row in authors_data]
        logger.info(f"Fetched {len(authors)} authors")
        return AuthorsResponse(authors=authors)

    except Exception as e:
        logger.error(f"Error fetching authors: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/authors", response_model=CreateAuthorResponse)
def create_author(payload: CreateAuthorRequest):
    """Create a new author."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        # Check if author already exists
        cursor.execute("SELECT author_id FROM authors WHERE name = ?", (payload.name,))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            raise HTTPException(status_code=409, detail=f"Author '{payload.name}' already exists")

        # Insert new author
        cursor.execute("INSERT INTO authors (name) VALUES (?)", (payload.name,))
        conn.commit()

        # Get the created author ID
        author_id = cursor.lastrowid
        conn.close()

        logger.info(f"Author created: {author_id} - {payload.name}")
        return CreateAuthorResponse(author_id=author_id, name=payload.name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating author: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region BOOK ENDPOINTS
@app.get("/book/{book_id}", response_model=BookDetails)
def get_book_details(book_id: int):
    """Retrieve full details for a specific book."""
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


@app.get("/books/search", response_model=SearchBooksResponse)
def search_books(title: str, author_ids: list[int] = None):
    """Search for books by title and author IDs."""
    try:
        logger.info(f"Search books - Title: {title}, Author IDs: {author_ids}")

        conn = db.get_connection()
        cursor = conn.cursor()

        # Build query to search for books with matching title
        if author_ids and len(author_ids) > 0:
            # If author IDs provided, join with book_authors and filter by author IDs
            author_ids_placeholders = ",".join(["?"] * len(author_ids))
            query = f"""
                SELECT DISTINCT b.book_id, b.title, b.description, b.infolink
                FROM books b
                JOIN book_authors ba ON b.book_id = ba.book_id
                WHERE LOWER(b.title) LIKE LOWER(?)
                AND ba.author_id IN ({author_ids_placeholders})
            """
            params = [f"%{title}%"] + author_ids
        else:
            # Search only by title if no authors provided
            query = """
                SELECT DISTINCT b.book_id, b.title, b.description, b.infolink
                FROM books b
                WHERE LOWER(b.title) LIKE LOWER(?)
            """
            params = [f"%{title}%"]

        cursor.execute(query, params)
        results = cursor.fetchall()

        # Fetch full details for each book from database
        books = []
        for row in results:
            book_id = row["book_id"]
            details = service.get_book_details_from_db(book_id)
            if details:
                books.append(BookDetails(**details))

        conn.close()
        logger.info(f"Book search: title='{title}', author_ids={author_ids}, found={len(books)}")
        return SearchBooksResponse(books=books)

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /books/search: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/books", response_model=AddBookResponse)
def add_book(payload: AddBookRequest):
    """Add a new book to the catalog."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()

        # Get the next book_id
        cursor.execute("SELECT MAX(book_id) as max_id FROM books")
        result = cursor.fetchone()
        next_book_id = (result["max_id"] or 0) + 1

        # Insert book
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO books (book_id, title, description, infolink, created_at) VALUES (?, ?, ?, ?, ?)",
            (next_book_id, payload.title, payload.description, payload.infolink, now),
        )

        # Link authors
        for author_id in payload.authors:
            cursor.execute("INSERT INTO book_authors (book_id, author_id) VALUES (?, ?)", (next_book_id, author_id))

        # Link genres
        for genre_id in payload.genres:
            cursor.execute("INSERT INTO book_genres (book_id, genre_id) VALUES (?, ?)", (next_book_id, genre_id))

        conn.commit()
        conn.close()

        try:
            # Fetch genre names
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT g.name FROM genres g JOIN book_genres bg ON g.genre_id = bg.genre_id WHERE bg.book_id = ?",
                (next_book_id,),
            )
            genres = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Generate embedding
            model = SentenceTransformer(EMBEDDINGS["embedding_model"])
            embedding = db.generate_single_book_embedding(payload.title, genres, payload.description, model)

            # Save to database
            db.save_embedding_to_db(next_book_id, embedding)
            logger.info(f"✓ Embedding generated for book {next_book_id}")
        except Exception as e:
            logger.warning(f"Failed to generate embedding for book {next_book_id}: {str(e)}")

        logger.info(f"Book added: {next_book_id} - {payload.title}")
        return AddBookResponse(book_id=next_book_id, title=payload.title, created_at=now)

    except Exception as e:
        logger.error(f"Error adding book: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region RECOMMENDATION ENDPOINTS
@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: Optional[str] = None, k: int = 10):
    try:
        payload = {"user_id": user_id, "k": k}
        response = rec.recommend(service, idb, payload)
        return response
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /recommend: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swipe", response_model=SwipeResponse)
def swipe(payload: SwipeRequest):
    try:
        response = rec.swipe(service, idb, payload)
        return response
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /swipe: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region PIPELINE ENDPOINTS
@app.get("/pipeline/status")
def get_pipeline_status():
    """Get current pipeline execution status and progress."""
    try:
        return _get_pipeline_state()
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /pipeline/status: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/run")
def start_pipeline():
    """Start the full ML pipeline in background."""
    try:
        state = _get_pipeline_state()
        if state["status"] == "running":
            raise HTTPException(status_code=409, detail="Pipeline is already running")

        pipeline_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

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


# endregion


# region RETRAIN ENDPOINTS
@app.post("/retrain/trigger")
def trigger_batch_retrain():
    """Manually trigger batch retraining of the CF model."""
    try:
        from scripts.batch_retrain import main as batch_retrain_main

        # Start retrain in background thread
        retrain_thread = threading.Thread(target=_run_batch_retrain_background, daemon=True)
        retrain_thread.start()

        logger.info("Batch retraining triggered from admin endpoint")

        return {
            "status": "started",
            "message": "Batch retraining started in background",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"ERROR in /retrain/trigger: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region HELPERS
def _get_pipeline_state():
    """Thread-safe read of pipeline state."""
    with pipeline_state_lock:
        return dict(pipeline_state)


def _update_pipeline_state(
    current_stage: str = None, status: str = None, overall_progress: int = None, error: str = None
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


def _run_batch_retrain_background():
    """Execute batch retraining in background."""
    try:
        from scripts.batch_retrain import main as batch_retrain_main

        logger.info("Starting batch retraining in background...")
        success = batch_retrain_main()

        if success:
            logger.info("✓ Batch retraining completed successfully")
            # Reinitialize recommendation service with updated factors
            service.reinitialize()
            logger.info("✓ RecommendationService reinitialized with updated factors")
        else:
            logger.error("Batch retraining failed")

    except Exception as e:
        logger.error(f"Batch retraining background execution failed: {str(e)}\n{traceback.format_exc()}")


def _run_pipeline_background():
    """Execute the full ML pipeline in background with progress updates."""
    try:
        logger.info("Starting full pipeline execution...")

        stages = [
            ("stage_1", "Data pipeline", database_handler.run_data_pipeline, 0),
            ("stage_2", "Model pipeline", model_handler.run_model_pipeline, 50),
        ]

        for stage_id, stage_name, stage_func, progress_start in stages:
            try:
                logger.info(f"Executing {stage_id}...")
                _update_pipeline_state(current_stage=stage_name, overall_progress=progress_start)
                stage_func()
                logger.info(f"✓ {stage_id} completed")
            except Exception as e:
                error_msg = f"Stage {stage_id} failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                _update_pipeline_state(status="failed", error=str(e))
                raise

        # Pipeline completed successfully
        _update_pipeline_state(status="completed", overall_progress=100, current_stage="All stages complete")
        logger.info("✓ Full pipeline completed successfully")

        # Reinitialize recommendation service with newly generated artifacts
        service.reinitialize()
        logger.info("✓ RecommendationService reinitialized with new artifacts")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}")


# endregion
