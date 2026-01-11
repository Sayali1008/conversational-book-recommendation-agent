# Phase 2 - Personalization with Collaborative Filtering

## 1. Goal
Users who liked X also liked Y, even when X and Y aren't semantically similar.

---

## 2. Steps Involved

- Create synthetic or simple real implicit feedback data.
- Train a matrix factorization model using an “implicit” library.
- Build a hybrid score function combining:
    - CF score
    - embedding similarity
- Add a fallback strategy for new users and new books.

---

## 3. Data Preparation

### 3.1. Basic Statistics

Key observations from the raw dataset:
- ~2.44M reviews
- ~1.01M users
- ~212k books raw, ~136k usable
- Ratings are discrete 1–5
- Strong positive skew (mean 4.22)
- Users and books both long-tailed
- No critical missing values in core columns

### 3.2. Data Analysis from Ratings Data

- Review scores have a strong positive skew.
- Users are far more likely to leave reviews when they like a book
- A “5-star” does not necessarily mean exceptional, just “not bad”
- Treating scores as absolute truth can bias recommendations toward popularity
- Most users have 1–2 reviews
- Most titles have very few reviews

### 3.3. Data Manipulation Strategies for `ratings_df` (Ratings)
1. **Mandatory Field Validation:** Drops records missing critical data in the `title`, `user_id`, or `review/score` columns.
2. **Catalog Alignment:** Normalizes rating titles and performs an inner join with the cleaned catalog to ensure interaction data matches valid book entries.
3. **Temporal Standardization:** Converts Unix epoch timestamps into UTC-aware datetime objects.
4. **Recency Deduplication:** Sorts reviews chronologically to keep only the most recent interaction for any unique user-book pair.
5. **Confidence Weighting:** Transforms 1-5 star ratings into a confidence scale where scores ≤ 3 are ignored, 4 is mapped to 1, and 5 is mapped to 2.

---

## 4. Implicit Interaction Definition

An interaction represents: A user engaging with a book enough to complete and rate it on Amazon.

Implicit interaction is:
- **Event**: A user rated a book
- **Signal strength**: Confidence score derived from star rating
  - 5 stars $\rightarrow$ confidence = 2 (strong positive)
  - 4 stars $\rightarrow$ confidence = 1 (moderate positive)
  - ≤3 stars $\rightarrow$ filtered out (no signal)

Absent interactions: No rating does NOT mean dislike; it means unknown. The user may not have read the book, or read but didn't rate it.

This definition treats all interactions as positive signals with varying strength, which aligns with implicit feedback assumptions.

Ratings are treated as implicit positive feedback with varying intensity. The implicit assumption is: "If a user rated a book 4-5 stars, they engaged with it positively." Neutral/negative ratings (3 or below) are ignored - it keeps the signal clean and avoids the "implicit library doesn't handle negatives well" problem as well.

**One consideration for later**: The implicit library typically expects confidence values in a wider range (like 1-10) because it uses the formula `1 + alpha * confidence` internally. Values (1, 2) are quite compressed. This isn't wrong, but we might want to scale them up:
- 5 stars → confidence = 10
- 4 stars → confidence = 5
- ≤3 stars → filtered out

This gives the ALS algorithm more room to differentiate between "liked" and "loved." But you can experiment with both and see what works better.

---

## 5. Defining the Universe

The "universe" in collaborative filtering refers to **the set of all possible (user, item) pairs that could theoretically receive a recommendation**.

### 5.1. Universe Components

**Users ($U$)**
1. **Definition:** All unique users within `cf_ratings_df` with at least one valid interaction.
2. **Filter Criteria:** Restricted to users who have provided a rating of 4–5 stars.

**Items ($I$)**
1. Two potential definitions for the item universe:
   1. **Option A: CF-Trainable Items Only**
      1. **Scope:** Limited to books present in `cf_ratings_df` (minimum of one 4–5 star rating).
      2. **Constraint:** Model restricted to items with established interaction data.
      3. **Universe:** $U \times I_{cf\_trainable}$
   2. **Option B: Full Catalog**
      1. **Scope:** Comprehensive set of all books in `catalog_books_df`.
      2. **Constraint:** Includes cold-start items (items with zero historical ratings).
      3. **Universe:** $U \times I_{full\_catalog}$
        ```text
        Universe = U × I_full_catalog
        Where:
            - I_cf_trainable ⊂ I_full_catalog (books with ratings)
            - I_cold_start = I_full_catalog - I_cf_trainable (books without ratings)
        ```

### 5.2. Recommended Approach: Option B with a nuanced approach

**Why this matters:**
- **For CF training**: You only train on I_cf_trainable (books with interaction data)
- **For recommendation**: You can recommend from I_full_catalog by using hybrid scoring:
  - Books in I_cf_trainable: Get CF score + embedding score
  - Books in I_cold_start: Get only embedding score (fallback)

This architecture handles the cold start problem naturally - new books without ratings can still be recommended via book embeddings.

```text
Universe size: 204,985 users x 136,001 items
CF-trainable items: 50,776 (37.3% of catalog)
Cold-start items: 85,225
```

---

## 6. Building the Interaction Matrix

The interaction matrix is a **sparse matrix** of shape `(n_users, n_items_cf)` where:
- Rows = users
- Columns = books (CF-trainable only)
- Values = confidence scores
- Most entries = 0 (implicit: "no known interaction")

### 6.1. Matrix Structure
```
           book_0  book_1  book_2  ...  book_n
user_0        0       2       0    ...     0
user_1        1       0       0    ...     2
user_2        0       0       1    ...     0
...
user_m        2       0       0    ...     0
```

---

## 7. The Training Algorithm: Alternating Least Squares (ALS)

### 7.1. Matrix Factorization Goal

Decompose your sparse interaction matrix `R` (n_users × n_items) into two lower-rank matrices:
- `U` (n_users × k): User latent factors
- `V` (n_items × k): Item latent factors

Such that: `R ≈ U × V^T`

Where `k` is the number of latent factors (typically 50-200).

#### 7.1.1. What are latent factors?

Think of them as hidden dimensions that explain user preferences and item characteristics. For books, latent factors might capture:
- Factor 1: "Fantasy vs. Realism"
- Factor 2: "Literary vs. Commercial"
- Factor 3: "Character-driven vs. Plot-driven"
- Factor 4: "Light vs. Heavy themes"
- ... and so on

The model learns these factors automatically from the data - you don't define them.

### 7.2. Hyperparameters

**1. Number of factors (k):**
- Too few: Can't capture preference complexity
- Too many: Overfitting on sparse data
- Typical range: 50-200
- More dimensions give the model more expressiveness, which can help with sparse data.

**2. Regularization (λ):**
- Controls overfitting
- Penalizes large factor values
- Typical range: 0.001 - 0.1
- Less regularization allows the model to learn larger factor values, especially for users with few interactions.

**3. Number of iterations:**
- More iterations = better fit but diminishing returns
- Typical range: 15-30

**4. Confidence function (alpha):**
- The `implicit` library uses: `confidence = 1 + alpha * rating_value`
- Alpha controls how much weight to give to observed interactions
- Typical range: 10-40
- Lower alpha reduces the confidence weighting, which can help prevent overfitting to the few strong signals and allow weaker signals to contribute more.

### 7.4. What Happens During Training?

**User factors learn:**
- Which latent dimensions this user cares about
- Example: User vector `[0.8, -0.3, 0.5, ...]` means they strongly prefer fantasy (dim 1), dislike heavy themes (dim 2), moderately like character-driven stories (dim 3)

**Item factors learn:**
- How much each book exhibits each latent dimension
- Example: Book vector `[0.9, -0.2, 0.6, ...]` means it's highly fantasy, not too heavy, quite character-driven

**Prediction:**
To predict user u's affinity for item i: `score = u_u · v_i` (dot product)

High dot product = user's preferences align with book's characteristics = good recommendation.

### 7.5. Implicit Feedback Special Handling

Unlike explicit ratings (Netflix-style), implicit feedback treats **all unobserved entries as negative**:
- Observed interaction (4-5 stars): Positive signal with confidence
- Unobserved (no rating): Weak negative signal (user didn't interact)

The confidence weighting makes observed interactions much stronger than unobserved ones, but the model still learns from "what users didn't interact with."

### 7.6. Convergence

ALS is guaranteed to converge (loss decreases each iteration) but not necessarily to global optimum. In practice:
- Loss drops rapidly in first 5-10 iterations
- Flattens out after 15-20 iterations
- Stopping early is fine if loss plateaus

---

## 8. Hybrid Architecture

```text
For a given user and candidate book:

Step 1: Get CF score
  - If book is in CF-trainable set (has training data):
      cf_score = user_factors[user_idx] · item_factors[book_idx]
  - Else (cold-start book):
      cf_score = None

Step 2: Get embedding similarity score
  - Compute similarity between:
      - Book's embedding (from Phase 1)
      - User's profile (average/weighted embedding of books they've rated)

Step 3: Combine scores
  - If cf_score exists:
      final_score = λ * normalize(cf_score) + (1-λ) * normalize(embedding_score)
  - Else:
      final_score = embedding_score  # Fallback to pure content-based
```

```text
Hybrid Recommendation Pipeline:

Input: user_id, number of recommendations (k)
    ↓
Step 1: Generate CF scores for CF-trainable books (11,591 books)
Step 2: Create user embedding profile from rated books  
Step 3: Generate embedding similarity scores for ALL books (136,001 books)
Step 4: Normalize both score types to [0, 1]
Step 5: Blend scores with weight λ (for CF-trainable books)
Step 6: Use pure embedding scores for cold-start books
Step 7: Filter out already-rated books
Step 8: Return top-k recommendations
    ↓
Output: book_ids, scores, explanations (CF/embedding/hybrid)
```

---

## Summary
```text
cf_ratings_df (interactions)
   ↓
Create index mappings (user_to_idx, book_to_idx)
   ↓
Split into train/val/test matrices (80/10/10, stratified by user)
   ↓
Build sparse interaction matrix R (n_users × n_items_cf)
   ↓
Train the ALS model (k=64, λ=0.01, alpha=40, iterations=15)
   ↓
Learned Output: user_factors (n_users × 64), book_factors (n_books × 64)
   ↓
Evaluation → Hybrid scoring → Recommendation pipeline
```


## 9. Running Recommendations

<!-- prompt -->
Now we're done building the pipeline and ready to perform some testing. When we test, can we figure out from the books users hae rated - the common genre of books they like? So when they get recommended new books we'll know what genre they are from and whether there is any correlation. Also it wouldbe nice to have the genres of recommended books in that case.

<!-- prompt -->
Another thing to note is that when i set `filter_rated = False` books already rated by the user also show up. But the output showsw a few books multiple times with different scores. Why is that?