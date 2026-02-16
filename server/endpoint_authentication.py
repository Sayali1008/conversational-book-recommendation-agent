from datetime import datetime

from fastapi import HTTPException

from common.constants import *
from common.utils import setup_logging
from db.connection import get_db
from server.schemas import *

logger = setup_logging(__name__, PATHS["app_log_file"])


def create_user(payload: CreateUserRequest):
    db = get_db()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Check if user already exists
    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (payload.user_id,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="User ID already exists")

    # Create user
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO users (user_id, name, created_at, login_attempt, last_login) VALUES (?, ?, ?, ?, ?)",
        (payload.user_id, payload.name, now, 0, now),
    )
    conn.commit()
    conn.close()

    logger.info(f"User created: {payload.user_id}")
    return CreateUserResponse(user_id=payload.user_id, name=payload.name, created_at=now)


def login_user(payload: LoginRequest):
    db = get_db()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Look up user
    cursor.execute("SELECT user_id, name, login_attempt FROM users WHERE user_id = ?", (payload.user_id,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User ID not found. Please create an account first.")

    # Check if user has no genre preferences (first_login = no genres in user_genres table)
    cursor.execute("SELECT COUNT(*) as count FROM user_genres WHERE user_id = ?", (payload.user_id,))
    genre_count = cursor.fetchone()["count"]
    is_first_login = genre_count == 0

    # Update login tracking
    now = datetime.now().isoformat()
    cursor.execute(
        "UPDATE users SET login_attempt = login_attempt + 1, last_login = ? WHERE user_id = ?",
        (now, payload.user_id),
    )
    conn.commit()
    conn.close()

    logger.info(f"User logged in: {payload.user_id} {user['name']} (first_login={is_first_login})")
    return LoginResponse(user_id=user["user_id"], name=user["name"], first_login=is_first_login)


def check_user_exists(user_id: str):
    db = get_db()
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone() is not None
    conn.close()

    if exists:
        return {"exists": True}
    else:
        raise HTTPException(status_code=404, detail="User not found")
