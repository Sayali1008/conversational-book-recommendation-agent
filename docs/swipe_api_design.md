# Swipe API Design & Implementation

## Overview

The `/swipe` API logs user interactions to SQLite. It captures user feedback (like/dislike) and returns a fresh batch of recommendations, naturally adapting to their preferences through dynamic user profile building.

## Architecture

### Two Layers

1. **Persistent Layer (SQLite)**
   - Every swipe is logged to `data/server.db` with timestamp
   - Provides audit trail for analytics and future batch retraining
   - Schema: `(id, user_id, book_id, action, confidence, ts)`

2. **Recommendation Layer**
   - Pre-trained CF factors + embeddings (static, never updated)
   - User profile dynamically built from logged swipes
   - No model retraining or global updates on swipe

### Key Design Decisions

#### 1. Confidence Normalization
- **Frontend sends:** action = "like" or "dislike"
- **Server stores:** 1.0 for like, 0.0 for dislike
- **Why?** Binary signals are cleaner for analytics and future retraining

#### 2. Database-Backed User History
- ✅ Every swipe logged to SQLite
- ✅ User profile built dynamically from logged swipes
- ✅ No model retraining on swipe (remains offline)
- ✅ All historical data available for batch retraining

#### 3. No Live Model Updates
Swipes do NOT:
- Update `user_factors` or `book_factors`
- Recompute embeddings
- Rebuild FAISS index
- Trigger background retraining

Why? These are expensive operations that would cause high latency. Instead: use offline batch retraining (weekly/monthly) with accumulated swipe data.

## Implementation

### `/swipe` Endpoint

```python
@app.post("/swipe", response_model=SwipeResponse)
def swipe(payload: SwipeRequest):
    # Step 1: Normalize confidence
    confidence = 1.0 if payload.action == "like" else 0.0
    
    # Step 2: Log to SQLite (persistent)
    idb.insert_swipe(payload.user_id, payload.book_id, payload.action, confidence)
    
    # Step 3: Fetch all historical swipes from database
    all_swiped = idb.get_user_swiped_books(payload.user_id)
    exclude_catalog_indices = service.book_ids_to_catalog_indices(all_swiped)
    
    # Step 4: Generate fresh recommendations
    # - Cold users: profile from genres + swiped books
    # - Warm users: profile from swiped books + CF factors
    recs, strategy = service.recommend(
        user_id=payload.user_id,
        top_k=k,
        swiped_books=all_swiped,
    )
    
    # Step 5: Filter out the book just swiped
    filtered_recs = [r for r in recs if r["book_id"] != payload.book_id]
    
    # Step 6: Return next batch
    return SwipeResponse(status="ok", next_recommendations=filtered_recs)
```

### How Profile Building Works

**User profile** is dynamically built based on their interaction history:

- **Cold user** (no history): genres + now has 1 swipe
  - Profile = weighted average of recommended genre books + current swipe
  
- **Semi-warm user** (a few swipes): 
  - Profile = weighted average of swiped books (likes boosted, dislikes penalized)
  
- **Warm user** (in training data):
  - Uses pre-trained CF factors + swiped books as additional signal

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
Fetch all swiped books from database
    ↓
Build dynamic user profile:
  - likes weighted positive
  - dislikes weighted negative
    ↓
Generate recommendations:
  - If warm user: blend CF factors + embeddings + swipes
  - If cold user: embeddings + swipes + genres
    ↓
Filter out recently swiped book
    ↓
Return next_recommendations (fresh batch)
    ↓
Frontend: swap card, user sees next recommendation
```

## Why This Approach?

### Simplicity
- ✅ Single source of truth: SQLite database
- ✅ User profile computed fresh on each request from database
- ✅ No in-memory state management or cleanup needed

### Latency
- ✅ Swipe → log (< 50ms) + fetch swiped history (< 10ms) + generate recs (~ 100ms) = ~160ms total
- ❌ Swipe → recompute CF factors (10+ seconds) unacceptable

### Correctness
- ✅ All swipe history available in database for debugging
- ✅ Easy to replay or audit user interactions
- ✅ Portable across server restarts

## Future Enhancements

### 1. Batch Retraining Pipeline
Process accumulated swipes weekly/monthly:
```python
# Weekly job
swipes = db.get_swipes_since(last_retraining_date)
# Append swipes to training matrix
# Recompute CF factors offline
# Deploy new service with updated factors
```

### 2. Analytics & Insights
Track user behavior patterns:
```python
def get_user_stats(user_id):
    interactions = db.get_user_swiped_books(user_id)
    likes = sum(1 for i in interactions if i["action"] == "like")
    dislikes = sum(1 for i in interactions if i["action"] == "dislike")
    return {"likes": likes, "dislikes": dislikes, "engagement_rate": likes / (likes + dislikes)}
```

### 3. Swipe Export for Retraining
Export swipe data for offline ML pipeline:
```bash
# Export swipes from last 7 days
sqlite3 data/server.db "SELECT * FROM interactions WHERE ts > datetime('now', '-7 days')" > swipes_export.csv
```

## Summary

The `/swipe` API is designed for:
- ✅ Fast response times (fresh recommendations from database)
- ✅ Simple implementation (no session state, single source of truth)
- ✅ Rich audit trail (SQLite logging for future retraining)
- ✅ Adaptive recommendations (user profile built from swipes)
- ❌ NOT real-time model updates (use batch retraining instead)
