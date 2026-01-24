# Swipe API Design & Implementation

## Overview

The `/swipe` API is a **pure event logger** that records user interactions without triggering expensive model updates. It captures user feedback (like/dislike) and uses that feedback to guide immediate recommendations via lightweight session state, not by retraining models.

## Architecture

### Three Layers

1. **Persistent Layer (SQLite)**
   - Every swipe is logged to `data/server.db` with timestamp
   - Provides audit trail for analytics and future batch retraining
   - Schema: `(id, user_id, book_id, action, confidence, ts)`

2. **Session Layer (In-Memory)**
   - `UserSessionState` class tracks recent likes/dislikes per user
   - Stored in `session_state` global dict (cleared on server restart)
   - No persistence; ephemeral per session
   - Used to guide next batch recommendations

3. **Recommendation Layer**
   - Pre-trained CF factors + embeddings (static, never updated)
   - Recent swipes used as seed items for content-based scoring
   - No model retraining or global updates

### Key Design Decisions

#### 1. Confidence Normalization
- **Frontend sends:** action = "like" or "dislike" (optionally with confidence)
- **Server normalizes:** 1.0 for like, 0.0 for dislike
- **Why?** Binary signals are cleaner for analytics and future retraining than continuous values

#### 2. Session State Over Model Updates
- ❌ Don't update CF factors on each swipe → too slow, risk overfitting
- ✅ Track recent likes/dislikes in memory → fast, session-scoped
- ✅ Use recent likes as seed items → guides content-based recommendations

#### 3. Prefetch with Recent Seeds
```
Swipe 1: Like Book A
  → recent_likes = [A]
  → Prefetch next batch using seed=[A]
  → User sees next card immediately

Swipe 2: Dislike Book B
  → recent_dislikes = [B]
  → Prefetch using seed=[A], exclude=[A, B]
  
Swipe 3: Like Book C
  → recent_likes = [A, C]
  → Prefetch using seed=[A, C], exclude=[A, B, C]
```

This creates a virtuous loop: each swipe biases the next batch toward user's current preferences.

#### 4. No Live Model Updates
Swipes do NOT:
- Update `user_factors` or `book_factors`
- Recompute embeddings
- Rebuild FAISS index
- Trigger background retraining

Why? These are expensive operations that can:
- Cause high latency (user waits 5+ seconds per swipe)
- Lead to staleness if updates race with recommendations
- Risk inconsistency across concurrent requests
- Make rollback/debugging hard

Instead: use offline batch retraining (weekly/monthly) with accumulated swipe data.

## Code Changes

### 1. New Session State Class (`server/main.py`)

```python
class UserSessionState:
    def __init__(self, max_recent=20):
        self.sessions = defaultdict(lambda: {...})
    
    def record_swipe(self, user_id, book_id, action):
        """Track recent like/dislike."""
        if action == "like":
            self.sessions[user_id]["recent_likes"].insert(0, book_id)
        elif action == "dislike":
            self.sessions[user_id]["recent_dislikes"].insert(0, book_id)
    
    def get_recent_likes(self, user_id):
        """Return [book_id, ...]"""
        
    def get_all_recent_swiped(self, user_id):
        """Return set of all swiped (likes + dislikes)"""

session_state = UserSessionState(max_recent=20)
```

**Key parameters:**
- `max_recent=20`: Keep track of last 20 likes, last 20 dislikes (prevents memory bloat)

### 2. Enhanced `/swipe` Endpoint (`server/main.py`)

```python
@app.post("/swipe", response_model=SwipeResponse)
def swipe(payload: SwipeRequest):
    # Step 1: Normalize confidence
    confidence = 1.0 if payload.action == "like" else 0.0
    
    # Step 2: Log to SQLite
    storage.log_swipe(payload.user_id, payload.book_id, payload.action, confidence)
    
    # Step 3: Update session state
    session_state.record_swipe(payload.user_id, payload.book_id, payload.action)
    
    # Step 4: Prefetch next batch using recent likes as seeds
    recent_likes = session_state.get_recent_likes(payload.user_id)
    seed_catalog_indices = service.book_ids_to_catalog_indices(recent_likes)
    recs, _ = service.recommend(user_cf_idx, k=5, seed_catalog_indices=seed_catalog_indices)
    
    # Step 5: Filter out recently swiped items
    recently_swiped = session_state.get_all_recent_swiped(payload.user_id)
    filtered_recs = [r for r in recs if r["book_id"] not in recently_swiped]
    
    # Step 6: Return
    return SwipeResponse(status="ok", next_recommendations=filtered_recs)
```

### 3. Updated Schema Docstrings (`server/schemas.py`)

Added clarifications:
- `SwipeRequest.confidence` is optional; API normalizes it
- `SwipeResponse` includes prefetched recommendations for UX

## Flow Diagram

```
User swipes (right=like, left=dislike)
    ↓
POST /swipe {user_id, book_id, action}
    ↓
Normalize: action → confidence (1.0 or 0.0)
    ↓
Log to SQLite (persistent)
    ↓
Update session_state (in-memory)
    ↓
Get recent_likes from session_state
    ↓
Convert to seed_catalog_indices
    ↓
recommend(user_cf_idx, k=5, seed_catalog_indices)
    ├─ If warm user: use CF + seeds for hybrid scoring
    └─ If cold user: use seeds or catalog mean
    ↓
Filter out recently_swiped items
    ↓
Return next_recommendations
    ↓
Frontend: swap card, user doesn't wait
```

## What "Not Feeding Back to Live Scoring" Means

### Current (This Design)
Swipes are **logged but not immediately incorporated into model scoring**:
- Pre-trained CF factors remain static
- Pre-computed embeddings remain static
- Recent swipes used only as seed items for content-based fallback

**Result:** User preference drift is captured in seed selection, but ranking still relies on pre-trained patterns.

### Future (Batch Retraining)
When you want swipes to affect the model:
1. Collect swipes over time (e.g., 1 week)
2. Append swipes to training matrix
3. Recompute CF factors offline (no live impact)
4. Deploy new factors → immediately used for all requests
5. Clear session state; cycle continues

**Key insight:** Separate fast session-level feedback (seeds, filtering) from slow model-level learning (batch retraining).

## Why This Approach?

### Latency
- ✅ Swipe → log (< 50ms) + prefetch (~ 100ms) + return = ~150ms total
- ❌ Swipe → recompute factors (10+ seconds) + return = unacceptable

### Consistency
- ✅ Pre-trained factors are immutable; all users see consistent behavior
- ❌ Live factor updates can race with recommendations; hard to reason about

### Simplicity
- ✅ Session state is ephemeral; no cleanup needed
- ✅ SQLite logs are append-only; easy to replay or migrate
- ❌ Live model updates require careful version management and rollback logic

### Scalability
- ✅ Stateless recommendation engine scales horizontally
- ❌ Live factor updates require coordination across servers

## Testing the Swipe API

### Minimal Frontend Integration

```javascript
// Vue/Frontend
async function swipe(book, action) {
  const response = await fetch('http://localhost:8000/swipe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: currentUserId,
      book_id: book.book_id,
      action: action,  // "like" or "dislike"
      confidence: null  // API will normalize
    })
  });
  
  const data = await response.json();
  
  // data.next_recommendations is ready to display
  displayNextCard(data.next_recommendations[0]);
}
```

### Verify SQLite Logging

```bash
sqlite3 data/server.db "SELECT * FROM interactions LIMIT 10;"
```

Expected output:
```
1|user123|101|like|1.0|2026-01-18 10:30:45
2|user123|202|dislike|0.0|2026-01-18 10:31:12
3|user123|303|like|1.0|2026-01-18 10:32:08
```

## Future Extensions

### 1. Session Timeout
Clear old session state after inactivity:
```python
def cleanup_stale_sessions(max_age_minutes=60):
    for user_id, state in session_state.sessions.items():
        if (datetime.now() - state["ts"]).total_seconds() > max_age_minutes * 60:
            del session_state.sessions[user_id]
```

### 2. Batch Retraining Pipeline
Process accumulated swipes:
```python
# Weekly job
swipes = storage.get_swipes_since(last_retraining_date)
# Append to training matrix
# Recompute CF factors
# Deploy new service with updated factors
```

### 3. User Feedback Metrics
Track swipe patterns for analytics:
```python
def get_user_stats(user_id):
    likes = len(session_state.get_recent_likes(user_id))
    dislikes = len(session_state.get_recent_dislikes(user_id))
    return {"likes": likes, "dislikes": dislikes}
```

### 4. Confidence Calibration
If frontend can provide fine-grained confidence:
```python
# Store as-is for future ML
storage.log_swipe(user_id, book_id, action, confidence=0.8)
# But still use binary for immediate seeding
binary_confidence = 1.0 if confidence > 0.5 else 0.0
```

## Summary

The `/swipe` API is designed for:
- ✅ Fast response times (prefetch next batch)
- ✅ Lightweight session state (no model changes)
- ✅ Rich audit trail (SQLite logging for future retraining)
- ✅ Smooth UX (next card already loaded)
- ❌ NOT real-time model updates (use batch retraining instead)

This keeps the system simple, scalable, and maintainable while providing a responsive user experience.
