from datetime import datetime

from fastapi import HTTPException

from common.constants import *
from common.utils import setup_logging
from db.connection import get_db
from server.schemas import *

logger = setup_logging(__name__, PATHS["app_log_file"])


def get_genres():
    db = get_db()
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT genre_id, name FROM genres ORDER BY name ASC")
    genres_data = cursor.fetchall()
    conn.close()

    genres = [Genre(genre_id=row["genre_id"], name=row["name"]) for row in genres_data]
    logger.info(f"Fetched {len(genres)} genres")
    return GenresResponse(genres=genres)


def set_user_genres(payload: UserGenresRequest):
    user_id = payload.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    db = get_db()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Delete existing preferences
    cursor.execute("DELETE FROM user_genres WHERE user_id = ?", (user_id,))

    # Insert new preferences
    for genre_id in payload.genre_ids:
        cursor.execute(
            "INSERT INTO user_genres (user_id, genre_id, created_at) VALUES (?, ?, ?)",
            (user_id, genre_id, datetime.now().isoformat()),
        )

    conn.commit()
    conn.close()

    logger.info(f"User {user_id} saved {len(payload.genre_ids)} genre preferences")
    return UserGenresResponse(user_id=user_id, saved_genres=payload.genre_ids)
