# Variable Definitions & Mappings

This document clarifies the key variables and their relationships throughout the recommendation system.

---

## Core Variable Types

### `user_id` (string)
- **What:** External, human-readable user identifier
- **Source:** Raw data or user input (e.g., "user_12345")
- **Scope:** Global, unique per user
- **Used in:** Frontend, API requests, database logs
- **Example:** `"A14OJS0VWMOSWO"`, `"AFVQZQ8PW0L"`, `"AHD101501WCN1"`

### `user_cf_idx` (integer, 0-indexed)
- **What:** Collaborative Filtering matrix row index for a user
- **Source:** Assigned during `prepare_data()` in training
- **Scope:** Training/serving only, maps to CF user_factors matrix
- **Range:** 0 to (n_users - 1)
- **Used in:** CF model scoring, user_factors indexing
- **Example:** `user_cf_idx=42` → row 42 in user_factors array (shape: (n_users, 64))

### `book_id` (integer)
- **What:** Catalog book identifier from raw data
- **Source:** Raw books dataset (e.g., from books_data.csv)
- **Scope:** Global, unique per book across entire system
- **Used in:** Database, API responses, user ratings
- **Example:** `book_id=5`, `book_id=101`, `book_id=9999`

### `book_cf_idx` (integer, 0-indexed)
- **What:** Collaborative Filtering matrix column index for a book
- **Source:** Assigned during `prepare_data()` in training
- **Scope:** Training/serving only, maps to CF item_factors matrix
- **Range:** 0 to (n_cf_books - 1)
- **Used in:** CF model scoring, book_factors indexing
- **Example:** `book_cf_idx=150` → column 150 in book_factors array (shape: (n_cf_books, 64))

### `catalog_idx` (integer, 0-indexed)
- **What:** Row index in the cleaned books catalog (catalog_df)
- **Source:** Position in the feather file after cleaning/deduplication
- **Scope:** Indexing into catalog_embeddings and catalog_df
- **Range:** 0 to (len(catalog_df) - 1)
- **Used in:** Embedding lookups, catalog DataFrame access, recommendation results
- **Example:** `catalog_idx=23` → catalog_df.iloc[23], catalog_embeddings[23]

---

## Mapping Relationships

### Training Stage (prepare_data)

```
Raw ratings.csv
    ↓
Unique users extracted → assign user_cf_idx
    ↓
user_to_cf_idx: {user_id: user_cf_idx}
    Saved as: USER_IDX_PKL ("data/pkl/user_to_idx.pkl")
    
    
Raw ratings.csv
    ↓
Filter by min/max interactions
    ↓
Unique books extracted → assign book_cf_idx
    ↓
book_to_cf_idx: {book_id: book_cf_idx}
    Saved as: BOOK_IDX_PKL ("data/pkl/book_to_idx.pkl")
```

### Serving Stage (recommendation_service.py)

```
Load USER_IDX_PKL → get user_to_cf_idx: {user_id: user_cf_idx}
Reverse it → cf_idx_to_user: {user_cf_idx: user_id}
Build reverse → user_to_cf_idx: {user_id: user_cf_idx}

Load BOOK_IDX_PKL → get book_to_cf_idx: {book_id: book_cf_idx}
Reverse it → cf_idx_to_book: {book_cf_idx: book_id}

Load catalog_df → extract book_id column
Build → book_id_to_catalog_idx: {book_id: catalog_idx}

Combine → cf_idx_to_catalog_id_map: {book_cf_idx: catalog_idx}
    Links CF matrix columns to embedding/catalog rows
```

---

## Mapping Examples

### Example User Flow

**Scenario:** User "john_doe" likes books 101, 202, 303

```
Frontend Input:
  user_id = "john_doe"
  seed_book_ids = [101, 202, 303]

↓

API receives and processes:
  user_id = "john_doe"
  
  UserRegistry.get_user_idx("john_doe")
    → user_to_cf_idx["john_doe"] = 15
    → user_cf_idx = 15
  
  Service.book_ids_to_catalog_indices([101, 202, 303])
    → book_id_to_catalog_idx[101] = 5
    → book_id_to_catalog_idx[202] = 18
    → book_id_to_catalog_idx[303] = 42
    → seed_catalog_indices = [5, 18, 42]

↓

Recommendation:
  Average embeddings[5], embeddings[18], embeddings[42]
    → user_profile vector
  
  Score all items, rank by similarity
  
  Return top 10 with:
    - catalog_idx (for display)
    - book_id (from catalog_df.iloc[catalog_idx])
    - title, score, source
```

### Example CF Matrix Indexing

**Scenario:** Computing predictions for user_cf_idx=15

```
user_factors shape: (n_users, 64)
user_factors[15] → user vector (length 64)

book_factors shape: (n_cf_books, 64)
book_factors.T @ user_factors[15] → scores for all books

scores[150] = CF score for book_cf_idx=150

cf_idx_to_book[150] = 202
  → This book's book_id is 202

book_id_to_catalog_idx[202] = 18
  → This book's catalog row is 18

catalog_df.iloc[18] → get title, authors, etc.
catalog_embeddings[18] → get embedding
```

---

## Key Distinctions

| Variable | Created In | Used In | Scope | Mutable |
|----------|-----------|---------|-------|---------|
| `user_id` | Raw data / User input | API, Database, Frontend | Global | No |
| `user_cf_idx` | `prepare_data()` | CF scoring, user_factors indexing | Training/Serving | No |
| `book_id` | Raw data | API, Database, Catalog | Global | No |
| `book_cf_idx` | `prepare_data()` | CF scoring, book_factors indexing | Training/Serving | No |
| `catalog_idx` | Cleaned catalog | Embeddings, Display | Serving | No |

---

## Common Pitfalls

### ❌ Mistake 1: Mixing book_id with book_cf_idx
```python
# WRONG: Using book_id as array index
score = book_factors[book_id]  # Will fail if book_id != index

# CORRECT: Convert to CF index first
book_cf_idx = book_to_cf_idx[book_id]
score = book_factors[book_cf_idx]
```

### ❌ Mistake 2: Confusing catalog_idx with book_cf_idx
```python
# WRONG: Using catalog_idx for CF matrix
embedding = catalog_embeddings[book_cf_idx]  # May be wrong book!

# CORRECT: Use catalog_idx
embedding = catalog_embeddings[catalog_idx]

# To go from CF → catalog:
catalog_idx = cf_idx_to_catalog_id_map[book_cf_idx]
embedding = catalog_embeddings[catalog_idx]
```

### ❌ Mistake 3: Unpacking pickles incorrectly
```python
# WRONG: Expecting tuple
cf_idx_to_user, cf_idx_to_book = load_pickle(USER_IDX_PKL)
# ValueError: too many values to unpack

# CORRECT: Load the forward mapping, reverse if needed
user_to_cf_idx = load_pickle(USER_IDX_PKL)
cf_idx_to_user = {v: k for k, v in user_to_cf_idx.items()}
```

---

## Saved Artifacts

| File | Contains | Key Content |
|------|----------|-------------|
| `USER_IDX_PKL` | user_to_cf_idx dict | `{user_id: user_cf_idx}` |
| `BOOK_IDX_PKL` | book_to_cf_idx dict | `{book_id: book_cf_idx}` |
| `user_factors.npy` | NumPy array | Shape (n_users, 64) |
| `book_factors.npy` | NumPy array | Shape (n_cf_books, 64) |
| `train_matrix.npz` | Sparse CSR matrix | Shape (n_users, n_cf_books) |
| `cleaned_books_data.ftr` | Feather (catalog_df) | Index = catalog_idx, columns include book_id, title, etc. |
| `catalog_books_384.npy` | Embeddings array | Shape (len(catalog_df), 384) |

---

## Summary Table: Where to Use What

```
If you have:        Use this mapping:           To get:
─────────────────────────────────────────────────────────
user_id             user_to_cf_idx           user_cf_idx
user_cf_idx         cf_idx_to_user      user_id
book_id             book_id_to_cf_idx           book_cf_idx
book_cf_idx         cf_idx_to_book      book_id
book_id             book_id_to_catalog_idx      catalog_idx
book_cf_idx         cf_idx_to_catalog_id_map    catalog_idx
catalog_idx         catalog_df.iloc[idx]        Row (title, authors, etc.)
catalog_idx         catalog_embeddings[idx]     Embedding vector
```
