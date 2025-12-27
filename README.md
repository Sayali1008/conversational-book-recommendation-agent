# Semantic Book Recommendation System
Building a full-stack book recommendation system with a sophisticated swipe + conversational interface.

This project builds a semantic search and recommendation foundation for books. Instead of relying on keyword matching, it represents each book as a dense vector embedding so that similarity is based on meaning, themes, and style.

The system is designed in phases. Phase 1 focuses on data preparation and semantic retrieval, which later phases will build on to support recommendation logic, ranking, and conversational interaction.

## Project Phases
### Phase 1: Semantic Indexing and Retrieval
- Ingest raw book metadata
- Clean and normalize text fields
- Generate semantic embeddings for each book
- Store embeddings in a vector index
- Enable fast semantic similarity search

**Outcome**: Given a natural-language query or a reference book, the system can retrieve meaningfully similar books.

## High-Level Architecture

At a high level, Phase 1 follows this pipeline:
- Load raw book metadata (title, authors, description, genres)
- Clean and normalize text fields
- Combine selected fields into a single embedding input
- Generate sentence embeddings using a pretrained model
- Store vectors in a FAISS index with associated metadata
- Perform semantic search via nearest-neighbor lookup

This phase produces a reusable semantic retrieval layer that later phases can treat as a black box.

## Tech Stack
- Python
- Pandas / NumPy for data processing
- SentenceTransformers for semantic embeddings
- FAISS for vector similarity search
- Jupyter Notebooks for exploration and analysis

## Repository Structure
```
README.md
docs/
  data_sources.md          # Dataset options and sourcing rationale
  data_preprocessing.md    # Cleaning, normalization, and deduplication steps
  embedding_design.md      # Embedding strategy and model selection
  vector_search.md         # FAISS index design and similarity metrics
notebooks/
  exploratory_data_analysis.ipynb
```

## How to Run Phase 1
- Clone the repository
- Install dependencies
- Run the data preprocessing pipeline
- Generate embeddings
- Build the FAISS index
- Execute semantic search queries

### Example Use Cases
- “Find books similar to The Left Hand of Darkness”
- “Books about found family in a sci-fi setting”
- “Literary novels with unreliable narrators”
