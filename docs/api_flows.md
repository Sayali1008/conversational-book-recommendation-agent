# API Flow Diagrams

This document traces the complete execution flow for API requests, from frontend click to response.

---

## GET /recommend

### High-Level Flow

```
User clicks "Get Recommendations" button
    ↓
Frontend HTTP GET /recommend?user_id=...&k=10
    ↓
server/main.py: @app.get("/recommend")
    ↓
RecommendationService.recommend()
    ↓
Determine user state: Is user warm/semi-warm/cold?
    ├─ WARM: in CF training matrix
    ├─ SEMI-WARM: has swiped books but not in matrix
    └─ COLD: no history, use genre preferences
    ↓
Build dynamic user profile based on state
    ↓
Get recommendations from handler.get_recommendations()
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

#### **Step 2: API Entry Point & Payload**

```
server/main.py: @app.get("/recommend")

Input validation:
  user_id: Optional[str] = "A14OJS0VWMOSWO"  (optional, for registered users)
  k: int = 10

Assemble payload:
  payload = {"user_id": user_id, "k": k}
  
Forward to endpoint handler: endpoint_recommendations.recommend(service, idb, payload)
```

---

#### **Step 3: Fetch Swiped Books from Database**

```
Endpoint calls idb.get_user_swiped_books(user_id)
  ↓
Queries interactions table: SELECT * FROM interactions WHERE user_id = ?
  ↓
Returns: list of dicts with {book_id, action, confidence, ts}
  ↓
Passes swiped_books to service.recommend()
```

---

#### **Step 4: Determine User State**

```
RecommendationService.recommend(user_id, top_k, swiped_books)
  ↓
user_cf = self.get_user_cf(user_id)
  → Lookup in index_mappings["user_id_to_cf"] (from CF training data)
  
in_matrix = self.user_has_history(user_cf)
  → Check if user_cf row in train_matrix has interactions
  
has_interactions = len(swiped_books) > 0
  → Check if user has swiped any books this session

Determine state:
  ├─ is_warm = in_matrix (user in CF training data)
  ├─ is_semi_warm = not in_matrix AND has_interactions (swiped but not in training)
  └─ is_cold = not in_matrix AND not has_interactions (no history at all)
```

---

#### **Step 5: Build User Profile**

```
User profile building depends on user state:

  IF warm user:
    user_profile = None  (None signals CF factors will be used)
    
  ELSE IF semi_warm user:
    user_profile = self.build_interaction_based_profile(swiped_books)
    → For each swiped book, get its embedding
    → Likes: weighted positive (+1.0)
    → Dislikes: weighted negative (-0.5)
    → Average to single profile vector
    
  ELSE (cold user):
    user_profile = self.build_genre_based_profile(user_id)
    → Query user_genres table
    → Find popular books in those genres (by rating count)
    → Get embeddings of those books
    → Average to single profile vector
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
---

#### **Step 6: Get Recommendations**

```
Call recommenders.handler.get_recommendations(
  context=self.context,
  user_cf=user_cf,
  candidate_pool_size=best_rec_params["candidate_pool_size"],
  lambda_weight=best_rec_params["lambda"],
  is_warm_user=is_warm,
  top_k=top_k,
  swiped_books=swiped_books,
  user_profile=user_profile
)
  ↓
Inside get_recommendations():
  
  # Build exclusion set (items to filter out)
  exclusions = _build_exclusions(user_cf, swiped_books)
    → Items from user's CF history + recent swipes
  
  # Cold user path: content-based only
  IF not is_warm_user:
    Get content-based scores using user_profile
    Return top_k items
    
  # Warm user path: hybrid (CF + embeddings)
  ELSE:
    Get CF scores: book_factors.dot(user_factors[user_cf])
    Build user profile from recent swipes (likes boosted, dislikes penalized)
    Get content-based scores using profile
    Normalize both scores
    Hybrid blend: (lambda * cf_norm) + ((1-lambda) * cb_norm)
    Return top_k items
  ↓
Final: indices (catalog indices), scores (final scores)
```

---

#### **Step 7: Format Recommendations**

```
For each (index, score) pair:
  row = self.context["catalog_df"].iloc[index]
  
  Extract:
    book_id = int(row["book_id"])
    title = row["title"]
    authors = row["authors"]  (parse if string)
    score = float(score)
  
  Create dict:
    {
      "book_id": book_id,
      "catalog_idx": index,
      "title": title,
      "authors": authors,
      "score": score
    }
  ↓
Return: recs (list of dicts), strategy ("hybrid" or "content_based")
```

---

#### **Step 8: Return Response**

```
Convert recs to BookRecommendation objects:
  recommendations = [BookRecommendation(**r) for r in recs]
  
Assemble RecommendResponse:
  {
    "recommendations": [
      {
        "book_id": 101,
        "catalog_idx": 5,
        "title": "The Great Gatsby",
        "authors": ["F. Scott Fitzgerald"],
        "score": 0.95
      },
      ...  (more items)
    ],
    "strategy": "hybrid"  or "content_based"
  }
  ↓
HTTP 200 with JSON response
  ↓
Frontend receives and displays results
```

---

### Complete User State Flow

```
Warm User (in CF training matrix):
  swiped_books = [interactions from DB]
  user_profile = profile from recent swipes
  ↓
  Hybrid: CF factors dominate, swipes refine
  Strategy: "hybrid"

Semi-Warm User (not in matrix, but has swipes):
  swiped_books = [recent swipes]
  user_profile = weighted average of swap embeddings
  ↓
  Content-Based: pure semantic similarity from swipes
  Strategy: "content_based"

Cold User (no history):
  swiped_books = []
  user_profile = genre-based embeddings
  ↓
  Content-Based: semantic similarity from genres
  Strategy: "content_based"
```
  
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

#### **Step 4: Fetch Historical Swipes & Build User Profile**

```
Retrieve all swiped books from database:
  all_swiped = db.get_user_swiped_books(user_id)
  ↓
Dynamically build user profile:
  - If warm user: use CF factors + embeddings from swipes
  - If semi-warm user: use embeddings from swipes + genres
  - If cold user: use genre embeddings + now 1 swipe
```

**Rationale:** User profile evolves naturally with each swipe. No model updates, no session state, just fresh profile generation from stored interactions.

---

#### **Step 5: Generate Fresh Recommendations**

```
Call service.recommend(
  user_id=user_id,
  top_k=k,
  swiped_books=all_swiped
)
  ↓
Inside recommend():
  - Build dynamic user profile from database
  - Generate candidates while excluding swiped items
  - Return top-k per strategy (hybrid or content-based)
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
Swipe 1: Like Book A
  ↓ Log to DB: (user_id, book_A, "like", 1.0)
  ↓ Fetch all swipes from DB: [A]
  ↓ Build profile from A + genres
  ↓ Generate recommendations
  ↓ Return next batch
  ↓ User sees next card

Swipe 2: Dislike Book B
  ↓ Log to DB: (user_id, book_B, "dislike", 0.0)
  ↓ Fetch all swipes from DB: [A, B]
  ↓ Build profile: like(A) - dislike(B)
  ↓ Generate recommendations
  ↓ Return next batch
```
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
