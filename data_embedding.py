from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pandas as pd
import constants as const

# Concatenate title + genre_union + description for each book
def prepare_text(row):
    title = row["title"]
    genres = ", ".join(row["genre_union"]) if row["genre_union"] else ""
    desc = row["desc"]
    return f"Title: {title}. Genre: {genres}. Description: {desc}"

def create_vector_store(df, model, batch_size=64, index_name="embeddings/books.index"):
    """
    Check if embeddings already exist in the database and if they do, return the vector store.
    Generate vector embeddings for each row in a dataframe.
    Each row is first converted into a single text string using `prepare_text`.
    Those strings are then encoded with a SentenceTransformer model.
    The final result is returned as a NumPy array for easy downstream use.
    FAISS index is created and embeddings are added to it.
    """

    if os.path.exists(index_name):
        embeddings = [] # embeddings not needed when vector store exists
        index = faiss.read_index(index_name)
        return embeddings, index

    # Convert each dataframe row into a single text input for the model
    texts = df.apply(prepare_text, axis=1).tolist()

    # Encode texts into embeddings
    # Batching helps control memory usage on large datasets
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Ensure a consistent NumPy output
    embeddings = np.asarray(embeddings)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index and add embeddings to index
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Write to index
    faiss.write_index(index, "books.index")
    return embeddings, index

def run_semantic_search(query, model, index, top_k=5):
    # Embed the search query
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    # Search and retrieve results
    scores, indices = index.search(query_vector, top_k)
    return scores[0], indices[0]

def main():
    df = pd.read_csv('data/books_cleaned.csv')
    
    # Load the embedding model once
    model = SentenceTransformer(const.EMBEDDING_MODEL)

    embeddings, index = create_vector_store(df, model)
    if embeddings:
        print("Shape of embeddings array:", embeddings.shape)
    print("Number of vectors in index:", index.ntotal)

    query = "emotionally devastating fiction about loss, grief, love, or tragedy"
    scores, indices = run_semantic_search(query, model, index, top_k=7)
    for score, idx in zip(scores, indices):
        print(f"Score: {score} | Title: {df.iloc[idx]['title']}\n Author: {df.iloc[idx]['author']} | Genre {df.iloc[idx]['genre_union']}")

    # Example output: (num_rows, 384) for all-MiniLM-L6-v2
    # Example output: (num_rows, 768) for all-mpnet-base-v2

if __name__ == "__main__":
    main()
