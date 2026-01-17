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

### High-Level Flow

```
User clicks 👍 or 👎 on a book
    ↓
Frontend HTTP POST /swipe
  {
    "user_id": "A14OJS0VWMOSWO",
    "book_id": 101,
    "action": "like",
    "confidence": 1.0
  }
    ↓
server/main.py: @app.post("/swipe")
    ↓
Storage.log_swipe()  (write to SQLite)
    ↓
Fetch next recommendations via service.recommend()
    ↓
Return SwipeResponse with optional next batch
```

---

### Detailed Flow

#### **Step 1: User Interaction**

```
Frontend (App.vue):
  swipe(rec, action):
    payload = {
      "user_id": "A14OJS0VWMOSWO",
      "book_id": 101,
      "action": "like",  (or "dislike" or "superlike")
      "confidence": 1.0
    }
    
    POST http://localhost:8000/swipe
```

---

#### **Step 2: Log Interaction**

```
server/main.py: @app.post("/swipe")
  ↓
Storage.log_swipe(user_id, book_id, action, confidence)
  ↓
SQLite INSERT:
  INSERT INTO interactions (user_id, book_id, action, confidence, ts)
  VALUES ("A14OJS0VWMOSWO", 101, "like", 1.0, NOW())
  ↓
Recorded for future training/analytics
```

---

#### **Step 3: Prefetch Next Batch (Optional)**

```
Get user_cf_idx from user_id
  ↓
Call service.recommend(user_cf_idx, k=5)  (small batch for snappy UX)
  ↓
Follow same flow as GET /recommend
  ↓
Return SwipeResponse:
  {
    "status": "ok",
    "next_recommendations": [
      {"book_id": 202, "title": "...", "score": 0.87, "source": "..."},
      ...
    ]
  }
  ↓
Frontend displays next batch without requiring another button click
```

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

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| Frontend → API | `user_id` string | Lookup in `user_to_cf_idx` | `user_cf_idx` int |
| Frontend → API | `seed_book_ids` list[int] | Lookup in `book_id_to_catalog_idx` | `seed_catalog_indices` list[int] |
| CF Matrix | `user_cf_idx` | Index into `user_factors[cf_idx]` | User vector (dim=64) |
| CF Matrix | `book_cf_idx` | Index into `book_factors[cf_idx]` | Book vector (dim=64) |
| Embeddings | `catalog_idx` | Index into `catalog_embeddings[idx]` | Embedding vector (dim=384) |
| Catalog | `catalog_idx` | Row access `catalog_df.iloc[idx]` | Book metadata (title, authors, etc.) |
| Results | `catalog_idx` | Lookup `catalog_df.iloc[idx]["book_id"]` | Final API response with `book_id` |
