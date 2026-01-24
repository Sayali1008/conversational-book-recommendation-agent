from collections import defaultdict
from datetime import datetime

from common.constants import PATHS
from common.utils import setup_logging

logger = setup_logging(__name__, PATHS["app_log_file"])


# ===================================================================
# Lightweight User Session State
# Tracks recent swipes per user without model updates
# ===================================================================
class UserSessionState:
    """In-memory session state for active users. Cleared on server restart."""

    def __init__(self):
        # user_id -> {"recent_likes": [book_ids], "recent_dislikes": [book_ids], "ts": datetime}
        self.sessions = defaultdict(lambda: {"recent_likes": [], "recent_dislikes": [], "ts": None})

    def record_swipe(self, user_id: str, book_id: int, action: str):
        """Update session state with new swipe (like or dislike)."""
        if action == "like":
            self.sessions[user_id]["recent_likes"].insert(0, book_id)
        elif action == "dislike":
            self.sessions[user_id]["recent_dislikes"].insert(0, book_id)
        self.sessions[user_id]["ts"] = datetime.now()
        logger.debug(
            f"Updated session state for {user_id}: likes={len(self.sessions[user_id]['recent_likes'])}, dislikes={len(self.sessions[user_id]['recent_dislikes'])}"
        )

    def get_recent_likes(self, user_id: str) -> list:
        """Get list of recently liked book_ids."""
        return self.sessions[user_id]["recent_likes"]

    def get_recent_dislikes(self, user_id: str) -> list:
        """Get list of recently disliked book_ids."""
        return self.sessions[user_id]["recent_dislikes"]

    def get_all_recent_swiped(self, user_id: str) -> set:
        """Get all recently swiped books (both likes and dislikes) to exclude from next batch."""
        return set(self.sessions[user_id]["recent_likes"]) | set(self.sessions[user_id]["recent_dislikes"])
