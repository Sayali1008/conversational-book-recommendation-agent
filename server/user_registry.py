from pathlib import Path
from typing import Optional

from common.constants import PATHS
from common.utils import load_index_mappings


class UserRegistry:
    def __init__(self):
        # Load mapping: cf_idx -> user_id, tolerate missing artifacts until pipeline runs
        idx_path = Path(PATHS["user_idx_pkl"])
        if idx_path.exists():
            self.user_to_cf_idx, self.cf_idx_to_user = load_index_mappings(str(idx_path))
        else:
            self.user_to_cf_idx, self.cf_idx_to_user = {}, {}

    def get_user_cf_idx(self, user_id: Optional[str]) -> Optional[int]:
        """Convert external user_id to CF matrix index. Returns None if unknown."""
        if not user_id:
            return None
        return self.user_to_cf_idx.get(user_id)