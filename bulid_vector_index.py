import faiss
import h5py
import numpy as np

# Function to load embeddings from .h5 files
def load_embeddings(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        ids = h5_file['id'][:]  # Load document or query IDs
        embeddings = h5_file['embedding'][:]  # Load embeddings
    return ids, embeddings

# Load document embeddings
doc_ids, doc_embeddings = load_embeddings("F:/python project/RAG_dense_vector_search/dataset/msmarco_passages_embeddings_subset.h5")  # Change the path to your own path
# Ensure document embeddings are in float32 for compatibility with faiss
doc_embeddings = doc_embeddings.astype('float32')

# Step 1: Normalize document embeddings for dot product similarity
# Normalize document embeddings for dot-product similarity search
faiss.normalize_L2(doc_embeddings)

# Step 2: Build HNSW index for dense vector search using faiss
# Create HNSW index using faiss with specified parameters for ANN search
def build_hnsw_index(embeddings, m=8, ef_construction=200, ef_search=200):
    dimension = embeddings.shape[1]  # Assuming 384-dimensional embeddings
    index = faiss.IndexHNSWFlat(dimension, m)  # Initialize HNSW index with 'm' neighbors
    index.hnsw.efConstruction = ef_construction  # Set construction parameter for indexing
    index.add(embeddings)  # Add document embeddings to the index

    # Set ef_search parameter for searching (higher values give more accurate results)
    index.hnsw.efSearch = ef_search
    return index

# Choose top_k value and set ef_search dynamically as at least double of top_k
top_k = 10
ef_search_value = max(2 * top_k, 400)  # Setting ef_search as at least twice of top_k

# Create the HNSW index with the specified parameters
hnsw_index = build_hnsw_index(doc_embeddings, m=8, ef_construction=200, ef_search=ef_search_value)

# Save the index to disk for future use
faiss.write_index(hnsw_index, "hnsw_index.faiss")
print("Index has been built and saved to 'hnsw_index.faiss'")




