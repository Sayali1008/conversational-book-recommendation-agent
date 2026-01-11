# Index Stability Analysis: Is `idx` Guaranteed to Never Change?

## Quick Answer

**YES, `idx` is guaranteed to never change** — but ONLY because:
1. It's created ONCE in `run_build_interaction_matrix()` and immediately persisted to disk (pickled)
2. All downstream stages (model training, embedding, recommendations) RELOAD the same pickle files
3. No code ever regenerates `user_to_idx` or `book_to_idx` — they use the saved versions

---

## Where idx is Created: The Single Source of Truth

### Location: [`pipeline.py` lines 91-99](pipeline.py#L91-L99)

```python
# Step 1: Extract unique IDs from filtered ratings_df (in order of appearance)
unique_users = ratings_df["user_id"].unique()      # Order depends on DataFrame.unique()
unique_books = ratings_df["book_id"].unique()      # Order depends on DataFrame.unique()

# Step 2: Create mappings (idx is assigned by enumerate order)
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
book_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}

# Step 3: PERSIST to disk immediately
with open(USER_IDX_PKL, "wb") as f:
    pickle.dump(user_to_idx, f)

with open(BOOK_IDX_PKL, "wb") as f:
    pickle.dump(book_to_idx, f)
```

**Critical observation**: `idx` values are determined by:
- The **order of appearance** in `unique_users` and `unique_books`
- This order comes from `.unique()` on the filtered `ratings_df`

---

## The Idx Flow Through the Pipeline

### Stage 1: Build Interaction Matrix (Creates idx)

```
ratings_df (filtered)
    ↓
unique_users, unique_books = .unique()
    ↓
user_to_idx, book_to_idx (created via enumerate)
    ↓
SAVED TO DISK: USER_IDX_PKL, BOOK_IDX_PKL  ← SINGLE SOURCE OF TRUTH
    ↓
Add columns: ratings_df["user_idx"], ratings_df["book_idx"]
    ↓
Build matrices: train_matrix, val_matrix, test_matrix
    │  (uses user_idx and book_idx columns)
    ↓
SAVED TO DISK: OUTPUT_TRAIN_MATRIX, OUTPUT_VAL_MATRIX, OUTPUT_TEST_MATRIX
```

**Key point**: The matrices use `user_idx` and `book_idx` directly from the dataframe:

From [`_train_cf.build_interaction_matrix()`](_train_cf.py#L1-L25):

```python
def build_interaction_matrix(df, n_users, n_cf_books):
    row = df["user_idx"].values           # These are from the mapping
    col = df["book_idx"].values           # These are from the mapping
    data = df["confidence"].values
    matrix = sp.csr_matrix(
        (data, (row, col)), 
        shape=(n_users, n_cf_books),      # Shape defined by n_users, n_cf_books
        dtype=np.float32
    )
    return matrix
```

**Result**: 
- train_matrix is (n_users × n_cf_books)
- Row indices are `user_idx` values (0 to n_users-1)
- Column indices are `book_idx` values (0 to n_cf_books-1)

### Stage 2: Train CF Model (Uses idx via saved matrices)

```python
# From pipeline.py run_train_cf_model()
train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)  # ← Uses saved idx values
cf_model = _train_cf.model_initialization(...)
cf_model = _train_cf.model_training(cf_model, train_matrix)

# Extract factors
user_factors = cf_model.user_factors  # Shape: (n_users, n_factors)
                                       # Row i = user with user_idx=i
book_factors = cf_model.item_factors  # Shape: (n_cf_books, n_factors)
                                       # Row i = book with book_idx=i
```

**Critical guarantee**: 
- `user_factors[i]` corresponds to user with `user_idx = i`
- `book_factors[j]` corresponds to book with `book_idx = j`
- These relationships are FIXED because the matrix row/column indices were fixed

### Stage 3: Hybrid Recommender (Loads saved idx)

```python
# From pipeline.py run_hybrid_recommender()
# RELOAD the SAME pickle files
user_to_idx, idx_to_user_id = _hybrid_recommender.load_index_mappings(USER_IDX_PKL)
book_to_idx, idx_to_book_id = _hybrid_recommender.load_index_mappings(BOOK_IDX_PKL)

# These are the EXACT SAME MAPPINGS that were used to create the matrices
# So idx values MUST match
```

---

## Proof: Why idx Can Never Change

### The Immutable Chain

```
1. run_build_interaction_matrix()
   ├─ Creates user_to_idx, book_to_idx via enumerate()
   ├─ SAVES to disk (pickle files)
   └─ Uses these indices to build matrices
   
2. run_train_cf_model()
   ├─ LOADS matrices (which used original idx)
   ├─ Trains model with shape determined by original matrices
   ├─ user_factors and book_factors inherit the same idx relationships
   └─ Model structure depends on original matrix indices
   
3. run_hybrid_recommender()
   ├─ RELOADS the SAME pickle files
   ├─ Gets the EXACT SAME user_to_idx, book_to_idx
   └─ Queries model factors using original indices
```

**Why it cannot change**:

| Why Change Would Break | Consequence |
|---|---|
| If you regenerate `user_to_idx` in stage 2 | Model factors would be indexed differently; predictions would be wrong |
| If you change the order of `unique_users` | Indices wouldn't match what's in the matrices |
| If you filter ratings again | You'd need to retrain everything from scratch |
| If a pickle file is corrupted | Code would crash immediately (no silent error) |

---

## Concrete Example of Idx Stability

### After Stage 1: What's Saved

```python
# ratings_df after filtering (suppose it comes in this order):
#   user_id  book_id  confidence  user_idx  book_idx
#   -------  -------  ----------  --------  --------
#    101      5001     4.0          0         0
#    105      5002     5.0          1         1
#    101      5003     3.0          0         2
#    112      5001     2.0          2         0
#    ...

# These are created:
# USER_IDX_PKL: {101: 0, 105: 1, 112: 2, ...}
# BOOK_IDX_PKL: {5001: 0, 5002: 1, 5003: 2, ...}

# train_matrix created with shape (3, 3) for this example
# train_matrix[0, 0] = 4.0  (user 101 → user_idx 0, book 5001 → book_idx 0)
# train_matrix[1, 1] = 5.0  (user 105 → user_idx 1, book 5002 → book_idx 1)
# train_matrix[0, 2] = 3.0  (user 101 → user_idx 0, book 5003 → book_idx 2)
# train_matrix[2, 0] = 2.0  (user 112 → user_idx 2, book 5001 → book_idx 0)
```

### In Stage 2: Training

```python
# Load the saved matrix
train_matrix = sp.load_npz(OUTPUT_TRAIN_MATRIX)
# Shape: (3, 3) with the exact same indices

# Train model
cf_model.fit(train_matrix)

# Result:
# user_factors[0] ← factor for user_idx=0 (user_id=101)
# user_factors[1] ← factor for user_idx=1 (user_id=105)
# user_factors[2] ← factor for user_idx=2 (user_id=112)
# book_factors[0] ← factor for book_idx=0 (book_id=5001)
# book_factors[1] ← factor for book_idx=1 (book_id=5002)
# book_factors[2] ← factor for book_idx=2 (book_id=5003)
```

### In Stage 3: Recommendations

```python
# Load the SAME mappings
user_to_idx, idx_to_user_id = load_index_mappings(USER_IDX_PKL)
# {101: 0, 105: 1, 112: 2, ...}

book_to_idx, idx_to_book_id = load_index_mappings(BOOK_IDX_PKL)
# {5001: 0, 5002: 1, 5003: 2, ...}

# Get recommendations for user 101
user_id = 101
user_idx = user_to_idx[user_id]  # = 0
user_vec = user_factors[user_idx]  # = user_factors[0] ✓ CORRECT

# Compute scores
cf_scores = book_factors.dot(user_vec)
# cf_scores[0] = book_factors[0].dot(user_factors[0])
#              = (factor for book 5001) · (factor for user 101) ✓ CORRECT
```

---

## Where We Know the User_id/Book_id → CF Index Mapping

### The Mapping Happens in Three Places:

#### 1. **Creation Time** (Stage 1: [`pipeline.py` lines 91-99](pipeline.py#L91-L99))
```python
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
book_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
```

#### 2. **Storage Time** (Stage 1: [`pipeline.py` lines 95-99](pipeline.py#L95-L99))
```python
with open(USER_IDX_PKL, "wb") as f:
    pickle.dump(user_to_idx, f)

with open(BOOK_IDX_PKL, "wb") as f:
    pickle.dump(book_to_idx, f)
```

#### 3. **Inference Time** (Stage 3: `_hybrid_recommender.load_index_mappings()`)
```python
def load_index_mappings(pkl_file):
    with open(pkl_file, "rb") as f:
        item_to_idx = pickle.load(f)  # ← Same dict from creation time
    
    idx_to_item_id = {cf_idx: item_id for item_id, cf_idx in item_to_idx.items()}
    return item_to_idx, idx_to_item_id
```

**The guarantee**: The pickle file preserves the EXACT same dictionary across all stages.

---

## How Idx Values Become Matrix Positions

Here's the explicit flow:

### Before Matrix Creation
```
ratings_df (has: user_id, book_id, confidence)
```

### After Mapping
```
ratings_df now has: user_id, book_id, confidence, user_idx, book_idx
         ↑ Added by: ratings_df["user_idx"] = ratings_df["user_id"].map(user_to_idx)
         ↑ Added by: ratings_df["book_idx"] = ratings_df["book_id"].map(book_to_idx)
```

### Into the Matrix
```
From _train_cf.build_interaction_matrix():
    row = df["user_idx"].values         # ← These are CF matrix ROW indices
    col = df["book_idx"].values         # ← These are CF matrix COL indices
    
    matrix[row[i], col[i]] = confidence[i]
```

### Result
```
matrix shape: (n_users, n_cf_books)
  where n_users = max(user_idx) + 1
  where n_cf_books = max(book_idx) + 1

user_factors shape: (n_users, n_factors)
  → user_factors[user_idx] = factor for user with that user_idx

book_factors shape: (n_cf_books, n_factors)
  → book_factors[book_idx] = factor for book with that book_idx
```

---

## Potential Risks (If idx Changed)

| Scenario | What Would Happen |
|----------|-------------------|
| **Rebuild mappings** (e.g., call `run_build_interaction_matrix()` again without retraining) | Indices could change if order of `.unique()` differs. Model would break because factors are misaligned. |
| **Load wrong pickle** | Indices from one model version wouldn't match another model's factors. Predictions would be randomly wrong. |
| **Manually edit pickle file** | Same as above. Silent corruption risk. |
| **Run only partial pipeline** | If you skip matrix building but try to use model training, there are no mappings to load. Would error. |

---

## Verification: Is idx Really Stable?

You can verify this by checking if the same keys appear in the mapping files at different stages:

```python
# Stage 1: After creation
import pickle

with open(USER_IDX_PKL, 'rb') as f:
    mapping_1 = pickle.load(f)

print("User mapping keys:", sorted(mapping_1.keys())[:10])
print("User idx values:", sorted(mapping_1.values())[:10])

# Stage 3: After loading
with open(USER_IDX_PKL, 'rb') as f:
    mapping_3 = pickle.load(f)

print("Mappings are identical:", mapping_1 == mapping_3)
```

These would be identical because the pickle file is never modified after stage 1.

---

## Summary

| Question | Answer |
|----------|--------|
| **Is idx guaranteed to never change?** | YES, because it's persisted in pickle files immediately after creation and reloaded in all downstream stages |
| **Where is idx created?** | [`pipeline.py` line 94-95](pipeline.py#L94-L95) via `enumerate(unique_users/unique_books)` |
| **Where is the mapping stored?** | [`pipeline.py` line 96-99](pipeline.py#L96-L99) as `USER_IDX_PKL` and `BOOK_IDX_PKL` pickle files |
| **Where do we know user_id → idx?** | In the pickle files (created once, reloaded always) |
| **Where do we know idx → matrix position?** | In the interaction matrices themselves (row/col indices match idx values) |
| **What guarantees consistency?** | No code regenerates mappings; all stages reload the same pickles |
| **What could break it?** | Manually editing pickles, running stages out of order, or deleting/corrupting the pickle files |
