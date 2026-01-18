from typing import Optional

from common.constants import PATHS
from common.utils import *


class UserRegistry:
    def __init__(self):
        # Load mapping: cf_idx -> user_id
        self.user_to_cf_idx, self.cf_idx_to_user = load_index_mappings(PATHS["user_idx_pkl"])

    def get_user_cf_idx(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index. Returns None if unknown."""
        if not user_id:
            return None
        return self.user_to_cf_idx.get(user_id)