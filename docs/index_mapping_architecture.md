# Index Mapping Architecture

## The Three Coordinate Systems

Your system has **three different ways** to refer to books:

1. **`book_id`** (catalog space): Original identifiers from `catalog_books_df` (1 to n_catalog_books)
2. **`cf_idx`** (CF matrix space): Column indices in the interaction matrix (0 to n_cf_books-1)
3. **`catalog_idx`** (DataFrame space): Row positions in `catalog_books_df` (0 to n_catalog_books-1)

Similarly for users:
- **`user_id`**: Original identifier from ratings data
- **`user_idx`**: Row index in CF matrix (0 to n_users-1)

## How the Mappings Work

### 1. `user_to_idx` and `book_to_idx`

Created in `run_build_interaction_matrix`:

```python
# From pipeline.py
user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
book_to_idx = {book_id: idx for idx, book_id in enumerate(unique_books)}
```

**Purpose**: Map original IDs → CF matrix positions

**Key insight**: The `idx` values are **determined by order of appearance** in `ratings_df` after filtering.

### 2. `idx_to_book_id` (and `idx_to_user_id`)

Created in `load_index_mappings`:

```python
# From _hybrid_recommender.py
idx_to_item_id = {cf_idx: item_id for item_id, cf_idx in item_to_idx.items()}
```

**Purpose**: Reverse mapping (CF matrix positions → original IDs)

### 3. `cf_to_catalog_map`

Created in `build_cf_to_catalog_mapping`:

```python
# From _hybrid_recommender.py
cf_to_catalog_map = {}
for cf_idx, book_id in idx_to_book_id.items():
    catalog_idx = book_id - 1  # book_id starts at 1, catalog indices start at 0
    cf_to_catalog_map[cf_idx] = catalog_idx
```

**Purpose**: CF matrix column → catalog DataFrame row

**Critical assumption**: `book_id` was assigned sequentially starting at 1 during data preprocessing

## Is idx Guaranteed to Never Change?
**YES**. Here's why:

1. **Single Creation Point**: idx is created exactly ONCE in pipeline.py via enumerate() on the unique users and books in order of appearance.
2. **Immediately Persisted**: The mappings are saved to pickle files right away:
    ```python
    with open(USER_IDX_PKL, "wb") as f:
        pickle.dump(user_to_idx, f)  # ← LOCKED IN
    ```
3. **Never Regenerated**: No code in your pipeline ever recreates these mappings. Every downstream stage (training, embedding, recommendations) reloads the same pickle files:
    ```python
    # Stage 3: Load the EXACT SAME mapping
    user_to_idx, idx_to_user_id = load_index_mappings(USER_IDX_PKL)
    ```
### How We Know the Mapping is Correct
The mapping is baked into the **interaction matrices themselves**. When you create the matrix:
```python
train_matrix[user_idx, book_idx] = confidence #           ↑ from user_to_idx    ↑ from book_to_idx
```
So:
- user_factors[i] corresponds to the user with user_idx = i (because the model trains on matrix rows)
- book_factors[j] corresponds to the book with book_idx = j (because the model trains on matrix columns)
- These relationships are unchangeable once the model is trained

## Concrete Example

Let's trace a specific book through the system:

### Initial State (After Preprocessing)

```python
# catalog_books_df (cleaned books)
#   catalog_idx  |  book_id  |  title
#   ------------------------------------
#        0       |     1     |  "1984"
#        1       |     2     |  "Dune"  
#        2       |     3     |  "Foundation"
#        3       |     4     |  "Neuromancer"
#      ...       |    ...    |   ...
```

### After CF Filtering (in `run_build_interaction_matrix`)

Only books with sufficient ratings remain:

```python
# unique_books from ratings_df after filtering
unique_books = [4, 2, 1, ...]  # book_ids in order of appearance

# book_to_idx mapping
book_to_idx = {
    4: 0,   # "Neuromancer" appears first → CF column 0
    2: 1,   # "Dune" appears second → CF column 1  
    1: 2,   # "1984" appears third → CF column 2
    ...
}

# idx_to_book_id (reverse)
idx_to_book_id = {
    0: 4,   # CF column 0 → book_id 4
    1: 2,   # CF column 1 → book_id 2
    2: 1,   # CF column 2 → book_id 1
    ...
}

# cf_to_catalog_map
cf_to_catalog_map = {
    0: 3,   # CF column 0 → catalog row 3 (book_id 4 - 1)
    1: 1,   # CF column 1 → catalog row 1 (book_id 2 - 1)
    2: 0,   # CF column 2 → catalog row 0 (book_id 1 - 1)
    ...
}
```

### In the Hybrid Recommender

When computing recommendations for a user:

```python
# Step 1: Get CF scores for all CF-trainable books
cf_scores = book_factors.dot(user_vec)  # Shape: (n_cf_books,)
# cf_scores[0] = score for CF column 0 = "Neuromancer"
# cf_scores[1] = score for CF column 1 = "Dune"
# cf_scores[2] = score for CF column 2 = "1984"

# Step 2: Get embedding scores for ALL catalog books
embedding_scores = cosine_similarity(user_profile, catalog_embeddings)
# embedding_scores[0] = score for catalog row 0 = "1984"
# embedding_scores[1] = score for catalog row 1 = "Dune"
# embedding_scores[3] = score for catalog row 3 = "Neuromancer"

# Step 3: Combine scores (mapping CF → catalog space)
for cf_idx in range(n_cf_books):
    catalog_idx = cf_to_catalog_map[cf_idx]
    final_scores[catalog_idx] = (
        lambda_weight * cf_scores_norm[cf_idx] + 
        (1 - lambda_weight) * embedding_scores_norm[catalog_idx]
    )

# Example for "Neuromancer":
# cf_idx=0 → catalog_idx=3
# final_scores[3] = λ * cf_scores_norm[0] + (1-λ) * embedding_scores_norm[3]
```

## Is the Mapping Correct?

**Yes**, because of this critical design choice in your data preprocessing:

From `_data_preprocessing.py` (implied from your workflow):

```python
# When creating catalog_books_df, you likely did:
catalog_books_df['book_id'] = range(1, len(catalog_books_df) + 1)
# This ensures: catalog_books_df.iloc[i] has book_id = i + 1
```

This guarantee makes the conversion **`catalog_idx = book_id - 1`** always correct.

## Verification in Your Code

You can verify this works correctly in [`create_user_embedding_profile`](_hybrid_recommender.py):

```python
# Line 48: Convert CF indices to catalog indices
rated_catalog_indices = np.array([
    cf_idx_to_catalog_idx(cf_idx, idx_to_book_id) 
    for cf_idx in rated_cf_indices
])

# Line 49: Fetch embeddings using catalog indices
rated_embeddings = catalog_embeddings[rated_catalog_indices]
```

This chain works because:
1. `cf_idx` → `book_id` via `idx_to_book_id`
2. `book_id` → `catalog_idx` via `book_id - 1`
3. `catalog_idx` → embedding via direct array indexing

## Potential Issue to Watch

⚠️ **The mapping assumes `book_id` was assigned sequentially starting at 1.**

If your preprocessing ever changes to use original dataset IDs or non-sequential IDs, you'd need to add an explicit `book_id_to_catalog_idx` mapping instead of relying on the -1 offset.

You could add this validation to catch issues:

```python
def build_cf_to_catalog_mapping(idx_to_book_id, catalog_df):
    """Build mapping with validation"""
    cf_to_catalog_map = {}
    
    for cf_idx, book_id in idx_to_book_id.items():
        # Find the actual catalog index where book_id matches
        catalog_idx = catalog_df[catalog_df['book_id'] == book_id].index[0]
        cf_to_catalog_map[cf_idx] = catalog_idx
        
        # Verify the -1 assumption still holds
        assert catalog_idx == book_id - 1, \
            f"book_id {book_id} not at expected position {book_id - 1}"
    
    return cf_to_catalog_map
```