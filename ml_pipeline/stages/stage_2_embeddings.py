import ast
import os

import faiss
import numpy as np

from common.constants import PATHS
from common.utils import setup_logging

logger = setup_logging(__name__, PATHS["eval_log_file"])


def generate_embeddings(df, model, batch_size=64):
    if os.path.exists(PATHS["catalog_books_index"]) and os.path.exists(PATHS["catalog_books_embeddings"]):
        embeddings = np.load(PATHS["catalog_books_embeddings"])
        index = faiss.read_index(PATHS["catalog_books_index"])
        return embeddings, index

    # Convert each dataframe row into a single text input for the model
    texts = df.apply(_prepare_text, axis=1).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.asarray(embeddings)

    np.save(PATHS["catalog_books_embeddings"], embeddings)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Write to index
    faiss.write_index(index, PATHS["catalog_books_index"])
    return embeddings, index


def _prepare_text(row):
    title = row["title"]
    genres = (
        ", ".join(ast.literal_eval(row["genres"]) if isinstance(row["genres"], str) else row["genres"])
        if row["genres"]
        else ""
    )
    desc = row["description"]
    return f"Title: {title} | Genre: {genres} | Description: {desc}"


def _run_semantic_search(query, model, index, top_k=5):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]
