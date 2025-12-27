from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import shutil

def snapshot_raw(input_path: str, snapshot_dir: str):
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Path(snapshot_dir) / f"raw_snapshot_{timestamp}.csv"
    shutil.copy2(input_path, target)
    return str(target)

def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)  # read everything as str to avoid dtype surprises
    df = df.fillna("")  # unify missing as empty string for now
    return df