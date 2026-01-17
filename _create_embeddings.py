import argparse
import ast
import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import *


# Concatenate title + genres + description for each book
def prepare_text(row):
    title = row["title"]
    genres = ", ".join(ast.literal_eval(row["genres"]) if isinstance(row["genres"], str) else row["genres"]) if row["genres"] else ""
    desc = row["description"]
    return f"Title: {title} | Genre: {genres} | Description: {desc}"

def generate_embeddings(logger, df, model, batch_size=64):
    """Generate embeddings for each row in a dataframe and create a FAISS index."""

    if os.path.exists(OUTPUT_CATALOG_BOOKS_INDEX) and os.path.exists(OUTPUT_CATALOG_BOOKS_EMBEDDINGS):
        embeddings = np.load(OUTPUT_CATALOG_BOOKS_EMBEDDINGS)
        index = faiss.read_index(OUTPUT_CATALOG_BOOKS_INDEX)
        return embeddings, index

    # Convert each dataframe row into a single text input for the model
    texts = df.apply(prepare_text, axis=1).tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.asarray(embeddings)

    np.save(OUTPUT_CATALOG_BOOKS_EMBEDDINGS, embeddings)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Write to index
    faiss.write_index(index, OUTPUT_CATALOG_BOOKS_INDEX)
    return embeddings, index

def run_semantic_search(query, model, index, top_k=5):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]
