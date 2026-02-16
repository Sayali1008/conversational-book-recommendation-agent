# Batch Retraining Setup Guide

## Overview

The batch retraining system automatically retrains the collaborative filtering (ALS) model with new user swipes on a weekly schedule (Sunday 2 AM ET) or manually via API.

## Quick Start

### Option 1: Automated Cron Job Setup (Recommended)

Run the setup script to configure the cron job automatically:

```bash
bash scripts/setup_cron.sh
```

This will:
- Create a wrapper script for cron execution
- Add a cron job that runs every Sunday at 7 AM UTC (2 AM ET)
- Configure logging to `logs/eval_logs/batch_retrain_cron.log`

**Verify the cron job was added:**
```bash
crontab -l | grep retrain
```

**Remove the cron job:**
```bash
crontab -e
# Then remove the line containing "batch_retrain"
```

---

### Option 2: Manual Cron Configuration

If you prefer to set up cron manually:

1. **Create/edit your crontab:**
   ```bash
   crontab -e
   ```

2. **Add this line:**
   ```
   0 7 * * 0 /path/to/project/scripts/retrain_cron_wrapper.sh
   ```

   Where:
   - `0 7 * * 0` = Every Sunday at 7 AM UTC (= 2 AM ET during EDT, 3 AM EST during winter)
   - Path should be absolute path to `retrain_cron_wrapper.sh`

3. **Save and exit** (usually Ctrl+X in nano, :wq in vim)

---

### Option 3: Manual Trigger via API

Trigger retraining anytime by calling the HTTP endpoint:

```bash
curl -X POST http://localhost:8000/retrain/trigger
```

Response:
```json
{
  "status": "started",
  "message": "Batch retraining started in background",
  "timestamp": "2026-02-15T14:30:00"
}
```

---

### Option 4: Manual Trigger via Command Line

Run batch retraining directly:

```bash
python3 scripts/batch_retrain.py
```

---

## How Batch Retraining Works

1. **Fetch New Interactions**: Queries all swipes since the last training date from the `interactions` table
2. **Merge Data**: Combines new swipes with original training ratings
3. **Handle New Users/Books**: Extends index mappings for users/books not in original model
4. **Retrain ALS**: Retrains the collaborative filtering model using the same hyperparameters:
   - `factors=128`, `alpha=80`, `iterations=15`, `regularization=0.2`
5. **Backup & Save**: Creates backup before saving, with atomic rollback on failure
6. **Update Metadata**: Records the new training date in the database
7. **Reinitialize Service**: RecommendationService picks up new factors automatically

---

## Monitoring Retraining

### Check Logs

The retraining process logs to: `logs/eval_logs/YYYYMMDD.log`

Example grep to see retrain events:
```bash
grep -i "batch retraining\|retrain" logs/eval_logs/*.log
```

### Check Last Training Date

Query the database directly:
```sql
SELECT * FROM metadata WHERE key = 'last_training_date';
```

Or via Python:
```python
from db.connection import get_db
db = get_db()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT value FROM metadata WHERE key = 'last_training_date'")
result = cursor.fetchone()
print(f"Last training date: {result[0]}")
conn.close()
```

---

## Troubleshooting

### Cron job not running?

1. **Check if cron is enabled:**
   ```bash
   sudo launchctl list | grep cron
   ```

2. **Check system logs:**
   ```bash
   log stream --predicate 'process == "cron"'
   ```

3. **Verify wrapper script is executable:**
   ```bash
   ls -l scripts/retrain_cron_wrapper.sh
   # Should have 'x' in permissions like: -rwxr-xr-x
   ```

### Retraining fails?

1. **Check eval logs:**
   ```bash
   tail -f logs/eval_logs/*.log
   ```

2. **Verify database connectivity:**
   ```bash
   ls -la data/database/system.db
   ```

3. **Check disk space:**
   ```bash
   df -h data/
   ```

4. **Model artifacts automatically rollback** on failure, so old factors are restored

---

## Timing Notes

- **Cron Time Format**: `minute hour day-of-month month day-of-week`
- **Sunday 2 AM ET**: 
  - During EDT (March-November): `0 7 * * 0` (7 AM UTC)
  - During EST (November-March): `0 8 * * 0` (8 AM UTC)
- **Current setup uses 7 AM UTC** (accounts for EDT, adjust for winter if needed)

---

## File Locations

- **Batch retrain script**: `scripts/batch_retrain.py`
- **Cron wrapper script**: `scripts/retrain_cron_wrapper.sh`
- **Setup script**: `scripts/setup_cron.sh`
- **Logs**: `logs/eval_logs/batch_retrain_cron.log` (cron-specific)
- **Model artifacts**: `data/model/user_factors.npy`, `data/model/book_factors.npy`
- **Index mappings**: `data/pkl/user_to_idx.pkl`, `data/pkl/book_to_idx.pkl`
- **Training metadata**: Database `metadata` table

---

## API Endpoint

**POST /retrain/trigger**

Manually trigger batch retraining from any client.

**Request:**
```bash
POST http://localhost:8000/retrain/trigger
```

**Response:**
```json
{
  "status": "started",
  "message": "Batch retraining started in background",
  "timestamp": "2026-02-15T14:30:00.123456"
}
```

**Notes:**
- Retraining runs in background thread
- Returns immediately while training completes asynchronously
- Check logs or RecommendationService readiness to verify completion
- No authentication required (implement if needed)

---
