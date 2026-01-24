# API Flow Diagrams

This document traces the complete execution flow for API requests, from frontend click to response.

---

## GET /recommend

### High-Level Flow

```
User clicks "Get Recommendations" button
    ↓
Frontend HTTP GET /recommend?user_id=...&k=10&seed_book_ids=...
    ↓
server/main.py: @app.get("/recommend")
    ↓
RecommendationService.recommend()
    ↓
_hybrid_recommender.recommend_with_cold_start()
    ↓
Determine: Is user warm or cold?
    ├─ WARM: Call hybrid_recommender() (CF + embeddings)
    └─ COLD: Use seed embeddings or catalog mean
    ↓
Merge & rank results
    ↓
Convert to BookRecommendation objects
    ↓
Return RecommendResponse to frontend
```

---

### Detailed Step-by-Step Flow

#### **Step 1: Frontend Request**

```
User Input:
  user_id = "A14OJS0VWMOSWO"  (optional, for cold-start)
  k = 10
  seed_book_ids = "101,202,303"  (optional, for cold users)

Frontend Code (App.vue):
  GET http://localhost:8000/recommend?user_id=A14OJS0VWMOSWO&k=10&seed_book_ids=101,202,303
```

---

#### **Step 2: API Entry Point**

```
server/main.py: @app.get("/recommend")

Input validation:
  user_id: Optional[str] = "A14OJS0VWMOSWO"
  k: int = 10
  seed_book_ids: Optional[str] = "101,202,303"

Parsing:
  seeds = [101, 202, 303]  (split comma-separated string)
```

---

#### **Step 3: Convert user_id to user_cf_idx**

```
UserRegistry.get_user_idx("A14OJS0VWMOSWO")
  ↓
Lookup in self.user_to_cf_idx dictionary (loaded from USER_IDX_PKL)
  user_to_cf_idx = {
    "A14OJS0VWMOSWO": 42,
    "AFVQZQ8PW0L": 15,
    ...
  }
  ↓
user_cf_idx = 42  (or None if user not in training data)
```

**Variable mapping:**
- Input: `user_id` (string from frontend)
- Output: `user_cf_idx` (integer, row in user_factors matrix)

---

#### **Step 4: Convert seed book_ids to catalog_indices**

```
RecommendationService.book_ids_to_catalog_indices([101, 202, 303])
  ↓
For each book_id, lookup in self.book_id_to_catalog_idx:
  book_id_to_catalog_idx = {
    101: 5,    (book_id 101 is at row 5 in catalog_df and embeddings)
    202: 18,
    303: 42,
    ...
  }
  ↓
seed_catalog_indices = [5, 18, 42]
```

**Variable mapping:**
- Input: `seed_book_ids` = [101, 202, 303] (book IDs from user)
- Output: `seed_catalog_indices` = [5, 18, 42] (rows in embeddings array)

---

#### **Step 5: Check if User is Warm**

```
RecommendationService.recommend(user_cf_idx=42, k=10, seed_catalog_indices=[5,18,42])
  ↓
self.user_has_history(user_cf_idx=42)
  ↓
Check: self.train_matrix[42].nnz > 0  (does row 42 have any interactions?)
  ↓
Result:
  - If YES → user is WARM (has rated books before)
  - If NO → user is COLD (new user or not in training data)
```

---

#### **Step 6A: WARM User Path**

If `user_cf_idx=42` is warm (has history in train_matrix):

```
_hybrid_recommender.recommend_with_cold_start(
  user_idx=42,
  is_warm_user=True,
  ...
)
  ↓
Call hybrid_recommender() for warm path:
  ├─ Get user vector: user_factors[42]  (shape: 64,)
  ├─ Compute CF scores: book_factors.dot(user_factors[42])  (shape: 14762,)
  ├─ Get embedding profile: Average embeddings of user's rated books
  ├─ Compute embedding scores: Cosine similarity to all items
  ├─ Normalize both:
  │  ├─ CF scores via minmax → [0, 1]
  │  └─ Embedding scores via minmax → [0, 1]
  ├─ Hybrid blend: 0.5 * cf_norm + 0.5 * emb_norm
  ├─ Mask already-rated items (set to -inf)
  └─ Return: top_k_indices, top_k_scores, ["hybrid"]*k

Result:
  warm_indices = [42, 15, 89, ...] (catalog_indices)
  warm_scores = [0.95, 0.87, 0.82, ...] (hybrid scores)
  warm_sources = ["hybrid", "hybrid", ...]
```

---

#### **Step 6B: COLD User Path**

If `user_cf_idx=None` or user is cold:

```
_hybrid_recommender.recommend_with_cold_start(
  user_idx=None,
  is_warm_user=False,
  seed_catalog_indices=[5, 18, 42],  (from Step 4)
  ...
)
  ↓
Build user profile:
  IF seed_catalog_indices provided:
    user_profile = Mean of embeddings[5], embeddings[18], embeddings[42]
  ELSE IF no seeds:
    user_profile = Mean of all embeddings (very weak prior)
  ↓
Call embedding_only_recommender():
  ├─ Get embeddings for all catalog items
  ├─ Compute cosine similarity: user_profile · all_embeddings.T
  ├─ Normalize scores via minmax → [0, 1]
  └─ Return: top_k_indices, top_k_scores

Result:
  cold_indices = [23, 67, 51, ...] (catalog_indices)
  cold_scores = [0.92, 0.85, 0.78, ...] (embedding scores only)
  cold_sources = ["embedding_only", "embedding_only", ...]
```

---

#### **Step 7: Merge Warm & Cold Results**

```
recommend_with_cold_start() merges both paths:
  ↓
all_indices = concatenate(warm_indices, cold_indices)
all_scores = concatenate(warm_scores, cold_scores)
all_sources = concatenate(warm_sources, cold_sources)
  ↓
Sort by score descending and take top-k:
  order = argsort(all_scores)[::-1][:k]
  final_indices = all_indices[order]
  final_scores = all_scores[order]
  final_sources = all_sources[order]
  ↓
Return: (final_indices, final_scores, final_sources)
```

**Key insight:** If user is warm, warm results come first. Cold items fill the remaining slots.

---

#### **Step 8: Convert Indices to Book Details**

```
Back in server/main.py:

For each catalog_idx in final_indices:
  row = catalog_df.iloc[catalog_idx]  (get row from cleaned books dataframe)
  ↓
  Extract:
    book_id = int(row["book_id"])  (original book ID from raw data)
    title = row["title"]
    score = final_scores[i]  (hybrid or embedding score)
    source = final_sources[i]  ("hybrid" or "embedding_only")
  ↓
  Create BookRecommendation object:
    {
      "book_id": 101,
      "title": "The Great Gatsby",
      "score": 0.95,
      "source": "hybrid"
    }
```

**Variable mapping:**
- Input: `catalog_idx` = 5 (row in embeddings/catalog_df)
- Lookup: `catalog_df.iloc[5]["book_id"]` → 101
- Output: Recommendation with `book_id=101, title=...`

---

#### **Step 9: Return Response**

```
Assemble RecommendResponse:
  {
    "recommendations": [
      {"book_id": 101, "title": "The Great Gatsby", "score": 0.95, "source": "hybrid"},
      {"book_id": 202, "title": "To Kill a Mockingbird", "score": 0.87, "source": "hybrid"},
      {"book_id": 303, "title": "Pride and Prejudice", "score": 0.82, "source": "embedding_only"},
      ...
    ],
    "strategy": "warm_hybrid",  (or "cold_embed" or "mixed")
    "used_seeds": [5, 18, 42]
  }
  ↓
HTTP 200 with JSON response
  ↓
Frontend receives and displays results
```

---

### Complete Variable Mapping Chain

```
Frontend Input:
  user_id = "A14OJS0VWMOSWO"
  seed_book_ids = [101, 202, 303]
  
  ↓ Step 3
  
  user_cf_idx = 42  (from user_to_cf_idx lookup)
  
  ↓ Step 4
  
  seed_catalog_indices = [5, 18, 42]  (from book_id_to_catalog_idx lookup)
  
  ↓ Steps 5-7
  
  final_catalog_indices = [5, 23, 89, ...]  (reranked results)
  final_scores = [0.95, 0.87, 0.82, ...]
  
  ↓ Step 8
  
  results = [
    {book_id: 101, title: "...", catalog_idx: 5, score: 0.95},
    {book_id: 202, title: "...", catalog_idx: 23, score: 0.87},
    ...
  ]
  
  ↓ Step 9
  
  RecommendResponse (HTTP 200)
```

---

## POST /swipe

### Purpose & Design Philosophy

The `/swipe` API is a **pure event logger** that records user interactions without triggering retraining, embedding rebuilds, or CF factor updates. Its goals are:

1. **Log interactions**: Record user feedback (like/dislike) with confidence scores (1.0 for like, 0.0 for dislike)
2. **Build audit trail**: Enable future batch retraining and analytics
3. **Update lightweight user state**: Maintain in-session user preferences to inform immediate recommendations (no model changes)
4. **Prefetch next batch**: Deliver smooth UX by proactively fetching recommendations while user swipes

**Key constraint:** Swipes do NOT immediately alter pre-trained CF factors or trigger retraining. Instead:
- Recommendations continue to use pre-computed factors (`user_factors`, `book_factors`)
- Recent swipes are leveraged as **seed items** to guide content-based / cold-start recommendations
- This keeps latency low and avoids model staleness issues

---

### High-Level Flow

```
User swipes right (👍) or left (👎) on a book
    ↓
Frontend HTTP POST /swipe
  {
    "user_id": "A14OJS0VWMOSWO",
    "book_id": 101,
    "action": "like",  (or "dislike")
    "confidence": null  (will be normalized by API)
  }
    ↓
server/main.py: @app.post("/swipe")
    ├─ Normalize confidence: 1.0 for "like", 0.0 for "dislike"
    ├─ Log interaction to SQLite
    ├─ Update lightweight user state (recent swipes)
    └─ Prefetch next batch using recent swipes as seeds
    ↓
Return SwipeResponse with next recommendations for seamless UX
```

---

### Detailed Flow

#### **Step 1: Frontend Payload**

```
User swipes on a book card:
  - Swipe right (👍) → action = "like"
  - Swipe left (👎) → action = "dislike"
  
Frontend normalizes to:
  payload = {
    "user_id": "A14OJS0VWMOSWO",
    "book_id": 101,
    "action": "like",
    "confidence": null  (optional; API will set it)
  }
    ↓
POST http://localhost:8000/swipe
  with Content-Type: application/json
```

---

#### **Step 2: Confidence Normalization**

```
server/main.py: @app.post("/swipe")
  ↓
Validate SwipeRequest (action must be "like" or "dislike")
  ↓
Normalize confidence:
  IF action == "like":
    confidence = 1.0  (strong positive signal)
  ELSE IF action == "dislike":
    confidence = 0.0  (strong negative signal)
  ELSE IF action == "superlike":
    confidence = 1.0  (treat same as like for now)
  ↓
Result: confidence is always 0.0 or 1.0 (binary signal)
```

**Why binary confidence?**
- Swipes are discrete gestures (yes/no), not gradients
- Binary signals are more stable for downstream analytics
- Reduces noise in future batch retraining

---

#### **Step 3: Log Interaction to SQLite**

```
Storage.log_swipe(user_id, book_id, action, confidence)
  ↓
SQLite INSERT:
  INSERT INTO interactions (user_id, book_id, action, confidence, ts)
  VALUES ("A14OJS0VWMOSWO", 101, "like", 1.0, NOW())
  ↓
Record persisted with timestamp
  ↓
Schema:
  id (INTEGER PRIMARY KEY AUTOINCREMENT)
  user_id (TEXT) → User identifier
  book_id (INTEGER) → Book ID from catalog
  action (TEXT) → "like" or "dislike"
  confidence (REAL) → 1.0 or 0.0
  ts (DATETIME) → Auto-timestamp of interaction
```

**Purpose of logging:**
- Build a complete audit trail of user behavior
- Enable offline batch retraining when sufficient data accumulates
- Support analytics and A/B testing (future)
- No immediate effect on live recommendations

---

#### **Step 4: Update Lightweight User State**

```
Create or update in-memory user session state:
  user_session_state[user_id] = {
    "recent_likes": [101, 202, 303],     (last N liked books)
    "recent_dislikes": [404, 505],        (last N disliked books)
    "last_updated": timestamp,
    "like_count": 3,
    "dislike_count": 2
  }
  ↓
Lightweight tracking without model changes:
  - Does NOT update CF factors
  - Does NOT recompute any vectors
  - Purely tracks session feedback
  ↓
Benefits:
  - Guides next recommendations via recent_likes as seed items
  - Enables simple filtering (exclude recent_dislikes)
  - Stays in memory; cleared on server restart
```

**Rationale:** Instead of retraining on every swipe, we use recent user feedback to bias the content-based recommendations. This is fast and reflects user's current session preferences without staleness.

---

#### **Step 5: Prefetch Next Batch Using Recent Swipes**

```
user_idx = UserRegistry.get_user_cf_idx(user_id)
  ↓
Retrieve recent_likes from user_session_state
  ↓
Convert recent_likes (book_ids) to seed_catalog_indices:
  seed_catalog_indices = [5, 18, 42]  (recent liked books as seeds)
  ↓
Call service.recommend(
  user_cf_idx=user_idx,
  k=5,  (small batch for next card)
  seed_catalog_indices=seed_catalog_indices
)
  ↓
Recommendation flow:
  IF user has CF history (warm):
    - Use CF + seed embeddings for hybrid scoring
    - Exclude recently swiped items (both likes and dislikes)
  ELSE (cold user):
    - Use seed embeddings or recent_likes as user profile
    - Fallback to catalog mean if no seeds yet
  ↓
Return top-5 recommendations
```

**Key behavior:** The prefetch uses recent swipes to guide recommendations but does NOT incorporate them into CF factors. This means:
- User preference drift is captured in seed items
- Pre-trained CF patterns still drive ranking
- No cold-start latency from model recomputation

---

#### **Step 6: Filter & Deduplicate**

```
Before returning results, filter out:
  - Recently swiped items (exclude from both recent_likes and recent_dislikes)
  - Duplicates (in case of prefetch overlap with previous batch)
  ↓
Ensure distinct, fresh recommendations
```

---

#### **Step 7: Return SwipeResponse**

```
Assemble response:
  {
    "status": "ok",
    "next_recommendations": [
      {
        "book_id": 202,
        "title": "To Kill a Mockingbird",
        "authors": ["Harper Lee"],
        "score": 0.87,
        "source": "hybrid",
        "catalog_idx": 23
      },
      ...  (4 more items)
    ]
  }
  ↓
HTTP 200 with JSON
  ↓
Frontend receives and immediately swaps in next card
```

**UX benefit:** User does not wait for recommendation fetch after each swipe; next card is already available.

---

### Interaction Flow Diagram

```
Session Start:
  user_session_state = {}

Swipe 1: Like Book A
  ↓ Log to DB: (user_id, book_A, "like", 1.0)
  ↓ Update state: recent_likes = [A]
  ↓ Prefetch using seed=[A]
  ↓ Return next batch
  ↓ User sees next card (already loaded)

Swipe 2: Dislike Book B
  ↓ Log to DB: (user_id, book_B, "dislike", 0.0)
  ↓ Update state: recent_dislikes = [B], recent_likes = [A]
  ↓ Prefetch using seed=[A], exclude=[B]
  ↓ Return next batch
  ↓ User sees next card

Swipe 3: Like Book C
  ↓ Log to DB: (user_id, book_C, "like", 1.0)
  ↓ Update state: recent_likes = [A, C]
  ↓ Prefetch using seed=[A, C], exclude=[B]
  ↓ Return next batch
  ↓ User sees next card
  
... (repeat as needed)

At end of session:
  - All interactions logged to SQLite
  - Session state cleared
  - Swipes available for batch retraining (offline, later)
```

---

### What "Not Feeding Back to Live Scoring" Means

**Current state:** Swipes are logged but do NOT immediately alter live recommendations via:
- ❌ Updating CF factors (`user_factors`, `book_factors`)
- ❌ Retraining CF model
- ❌ Recomputing embeddings
- ❌ Updating FAISS index

**Why?** These operations are expensive and risky:
- Retraining on every swipe → high latency, staleness, overfitting to noise
- Updating global embeddings → affects all users, hard to rollback
- Live factor updates → inconsistency across requests

**What we DO instead:** Use swipes as lightweight signals:
- ✅ Log to persistent storage (SQLite)
- ✅ Track in-session preferences (recent_likes, recent_dislikes)
- ✅ Seed content-based recommendations with recent likes
- ✅ Filter out recent dislikes from candidate pool

**Future work (batch retraining):**
When you want to "feed back" swipes into the model:
1. Collect swipes over time (e.g., 1-2 weeks of data)
2. Run offline batch retraining: incorporate swipes into training matrix
3. Recompute CF factors with new training data
4. Redeploy updated factors with zero downtime
5. Clear session state, start fresh cycle

This decouples fast session-level feedback (seeds, filtering) from slow model-level learning (batch retraining).

---

### Summary: /swipe Responsibility Matrix

| Responsibility              | Does It          | Notes                                           |
| --------------------------- | ---------------- | ----------------------------------------------- |
| Log interaction             | ✅ Yes            | Persists to SQLite with timestamp               |
| Normalize confidence        | ✅ Yes            | 1.0 for like, 0.0 for dislike                   |
| Update CF factors           | ❌ No             | Would require retraining                        |
| Rebuild embeddings          | ❌ No             | Would affect all users, high latency            |
| Track session state         | ✅ Yes            | In-memory recent_likes, recent_dislikes         |
| Use swipes as seeds         | ✅ Yes            | Guides content-based recommendations            |
| Filter recent swipes        | ✅ Yes            | Excludes them from next batch                   |
| Prefetch next batch         | ✅ Yes            | k=5 for snappy UX                               |
| Return next recommendations | ✅ Yes            | Allows seamless card transition                 |
| Enable batch retraining     | ✅ Yes (indirect) | Logs provide data for future offline retraining |

---

## Error Handling & Fallbacks

```
If user_id not found:
  user_cf_idx = None
  ↓ Triggers cold-start path
  ↓ Uses seed_book_ids or catalog mean

If seed_book_ids invalid:
  Skips unknown book_ids
  ↓ Uses valid ones, or falls back to catalog mean

If no candidates remain:
  Return empty recommendations list
  ↓ Frontend shows "No recommendations available"

If database error:
  HTTP 500 with error detail
  ↓ Frontend shows error message
```

---

## Summary: Key Variable Transformations

| Stage          | Input                     | Process                                  | Output                               |
| -------------- | ------------------------- | ---------------------------------------- | ------------------------------------ |
| Frontend → API | `user_id` string          | Lookup in `user_to_cf_idx`               | `user_cf_idx` int                    |
| Frontend → API | `seed_book_ids` list[int] | Lookup in `book_id_to_catalog_idx`       | `seed_catalog_indices` list[int]     |
| CF Matrix      | `user_cf_idx`             | Index into `user_factors[cf_idx]`        | User vector (dim=64)                 |
| CF Matrix      | `book_cf_idx`             | Index into `book_factors[cf_idx]`        | Book vector (dim=64)                 |
| Embeddings     | `catalog_idx`             | Index into `catalog_embeddings[idx]`     | Embedding vector (dim=384)           |
| Catalog        | `catalog_idx`             | Row access `catalog_df.iloc[idx]`        | Book metadata (title, authors, etc.) |
| Results        | `catalog_idx`             | Lookup `catalog_df.iloc[idx]["book_id"]` | Final API response with `book_id`    |
