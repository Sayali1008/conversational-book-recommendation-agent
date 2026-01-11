# Phase 1 - Semantic Book Representation & Retrieval

## 1. Goal
Transform raw book metadata into a structured, searchable semantic space where books can be retrieved by meaning rather than exact keyword overlap.

At the end of this phase, every book is represented as a dense vector embedding, and similar books can be retrieved efficiently using vector similarity search.

---

## 2. Steps Involved

- Ingests and load raw book metadata (titles, authors, descriptions, genres).
- Clean and normalize fields (strip HTML, fix encoding, normalize genre tags).
- Generate embeddings for each book with a pre-trained SentenceTransformer.
- Stores vectors in a searchable index (FAISS or similar) and metadata in a small DB.
- Write a small semantic-search module that returns top-K similar books (semantic similarity > keyword match).

---

## 3. Data Preparation

### 3.1. Data Source
- **Dataset:** Goodreads Books 100k (Kaggle)
- **Reason for selection:**
  - Rich natural language descriptions
  - Genre tags and author metadata
  - Large enough scale to observe realistic retrieval behavior

This dataset is well-suited for semantic search but does not include user interaction data, which is intentionally deferred to later phases.

### 3.2. Basic Statistics
Key observations from the raw dataset:
- No fully duplicated rows
- Partial duplicates based on `(title + author)` pairs
- Missing data present in:
  - Descriptions (~6.7%)
  - Genres (~10.4%)
- High genre cardinality (1,000+ unique genre strings)
- Text noise detected:
  - HTML tags
  - Escaped characters
  - Encoding artifacts
  - Control / non-printable Unicode characters

These findings directly informed the preprocessing strategy.

### 3.3. Data Manipulation Strategies for `catalog_books_df` (Books)
1. **Text Sanitization:** Removes HTML tags, unescapes entities, and strips non-printable control characters from titles and descriptions.
2. **Author Normalization:** Removes titles (e.g., "editor"), splits by delimiters (`;`, `&`, `and`), and standardizes to sorted Title Case lists.
3. **Genre Categorization:** Normalizes strings to lowercase, filters short/invalid entries, and maps infrequent genres to an "other" category based on a top-N threshold.
4. **Whitespace Collapsing:** Replaces tabs, newlines, and multiple consecutive spaces with a single space across all string fields.
5. **Quality Constraint Filtering:** Retains only "usable" rows containing a valid title, at least one author, at least one genre, and a minimum description length.

---

## 4. Data Embedding

### 4.1. Text Construction for Embedding
For each book, a single input string was constructed by concatenating:
1. Title
2. Genre list (space-joined)
3. Description

This balances:
- High-level intent (title)
- Thematic context (genres)
- Detailed semantics (description)

### 4.2. Embedding Generation

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** One dense vector per book
- **Format:** NumPy array aligned with dataset rows

This model was chosen for its strong performance on sentence-level semantic similarity tasks.

### 4.3. Vector Storage & Semantic Search

- Embeddings stored using **FAISS**
- Index type: `IndexFlatIP`
- Vectors L2-normalized to approximate cosine similarity
- Top-K semantic search implemented

This enables fast, in-memory semantic retrieval.

---

## 5. Limitations

### 5.1. No Personalization
- Results are identical for all users
- No modeling of user preferences or behavior

### 5.2. Query Intent Ambiguity
- Subjective queries (e.g. “sad books that make you cry”) may:
  - Overweight lexical cues
  - Miss deeper thematic intent

This is a known limitation of pure embedding-based retrieval.

### 5.3. Metadata Bias
- Titles and genre text can overly influence embeddings
- Metaphorical or misleading titles may rank highly

### 5.4. Genre Noise
- Genre labels are inconsistent and overly granular
- No weighting or taxonomy enforcement yet

Deferred to later hybrid ranking and reranking stages.

### 5.5. Edition-Level Duplication
- Multiple editions of the same book may appear as separate rows
- Acceptable for Phase 1 and revisitable later
