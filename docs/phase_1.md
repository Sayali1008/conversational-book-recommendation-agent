# Phase 1 — Semantic Book Representation & Retrieval

## Goal
Transform raw book metadata into a structured, searchable semantic space where books can be retrieved by meaning rather than exact keyword overlap.

At the end of this phase, every book is represented as a dense vector embedding, and similar books can be retrieved efficiently using vector similarity search.

---

## Data Source

- **Dataset:** Goodreads Books 100k (Kaggle)
- **Reason for selection:**
  - Rich natural language descriptions
  - Genre tags and author metadata
  - Large enough scale to observe realistic retrieval behavior

This dataset is well-suited for semantic search but does not include user interaction data, which is intentionally deferred to later phases.

---

## Selected Schema

The following fields were retained and normalized for Phase 1:

| Column        | Type              | Notes |
|--------------|-------------------|------|
| title        | string            | Normalized text |
| author       | list of strings   | One or more authors |
| desc         | string            | Long-form description |
| genre_union  | list of strings   | Normalized, unioned genres |
| isbn         | string            | May be missing |
| link         | string            | Always present |
| pages        | int64             | May be zero or missing |

---

## Exploratory Data Analysis Summary

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

---

## Text Normalization & Cleaning

A consistent normalization pipeline was applied to **titles**, **descriptions**, **authors**, and **genres**.

Normalization steps:
- Remove HTML tags
- Replace escaped characters (`\n`, `\t`, `\r`) with whitespace
- Collapse repeated whitespace
- Decode HTML entities
- Remove non-printable Unicode characters
- Preserve original casing and punctuation where possible

The intent was **noise removal**, not aggressive linguistic normalization.

---

## Author Normalization

- Split multi-author fields into lists
- Trim whitespace inside list items
- Remove empty entries and placeholder values (e.g. `"Unknown"`)
- Preserve author name formatting

---

## Genre Normalization

- Remove HTML tags
- Remove escaped characters
- Split comma-separated genre strings into lists
- Remove duplicates within each row
- Preserve fine-grained genre labels

No controlled genre taxonomy was enforced at this stage to avoid premature semantic loss.

---

## Duplicate Handling Strategy

### Definition
Partial duplicates were defined as rows sharing the same:
- Normalized title
- Normalized author list

### Approach
- Rows were **not merged or dropped**
- A **union of all genres** across duplicates was computed
- The unioned genre list was assigned back to **all duplicate rows**

This preserves:
- Multiple ISBNs or editions
- Potentially different descriptions
- Maximum genre coverage for embedding

---

## Missing & Incomplete Data Handling

Final filtering decisions:
- Dropped rows where:
  - `author` was empty **and**
  - `genre_union` was empty
- Dropped rows with placeholder or meaningless titles (e.g. titles containing `"unknown"`)
- Retained rows with missing descriptions
  - Empty descriptions are treated as weaker but valid signals

The resulting dataset is structurally consistent and semantically usable.

---

## Text Construction for Embedding

For each book, a single input string was constructed by concatenating:
1. Title
2. Genre list (space-joined)
3. Description

This balances:
- High-level intent (title)
- Thematic context (genres)
- Detailed semantics (description)

---

## Embedding Generation

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** One dense vector per book
- **Format:** NumPy array aligned with dataset rows

This model was chosen for its strong performance on sentence-level semantic similarity tasks.

---

## Vector Storage & Semantic Search

- Embeddings stored using **FAISS**
- Index type: `IndexFlatIP`
- Vectors L2-normalized to approximate cosine similarity
- Top-K semantic search implemented

This enables fast, in-memory semantic retrieval.

---

## What Phase 1 Enables

At the end of Phase 1, the system can:
- Retrieve books based on semantic similarity
- Handle vague or abstract natural language queries
- Surface thematically related books even without keyword overlap
- Serve as a foundation for personalization and hybrid recommendation logic

---

## Known Limitations

### No Personalization
- Results are identical for all users
- No modeling of user preferences or behavior

---

### Query Intent Ambiguity
- Subjective queries (e.g. “sad books that make you cry”) may:
  - Overweight lexical cues
  - Miss deeper thematic intent

This is a known limitation of pure embedding-based retrieval.

---

### Metadata Bias
- Titles and genre text can overly influence embeddings
- Metaphorical or misleading titles may rank highly

---

### Genre Noise
- Genre labels are inconsistent and overly granular
- No weighting or taxonomy enforcement yet

Deferred to later hybrid ranking and reranking stages.

---

### Edition-Level Duplication
- Multiple editions of the same book may appear as separate rows
- Acceptable for Phase 1 and revisitable later

---

## Phase 1 Status

**Phase 1 is complete.**

It establishes:
- Clean, normalized data
- Reliable semantic embeddings
- Fast vector-based retrieval
- Clear, documented failure modes

This provides a stable and well-understood foundation for Phase 2: Collaborative Filtering and Hybrid Recommendation Logic.
