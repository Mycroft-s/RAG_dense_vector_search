import faiss
import h5py
import numpy as np

# Define output file paths for indices and distances
indices_file_path = "top10_documents_indices.txt"
distances_file_path = "top10_documents_distances.txt"

# Function to load embeddings from .h5 files
def load_embeddings(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5_file:
        ids = h5_file['id'][:]  # Load query IDs
        embeddings = h5_file['embedding'][:]  # Load query embeddings
    return ids, embeddings
# Load query embeddings
query_ids, query_embeddings = load_embeddings("F:/python project/RAG_dense_vector_search/dataset/msmarco_queries_dev_eval_embeddings.h5")  # Change the path to your own path
# Ensure query embeddings are in float32 for compatibility with faiss
query_embeddings = query_embeddings.astype('float32')

# Limit to the first 10 queries for testing
query_ids = query_ids[:10]
query_embeddings = query_embeddings[:10]

# Normalize the query embeddings for dot-product similarity search
faiss.normalize_L2(query_embeddings)

# Load the index from disk
hnsw_index = faiss.read_index("hnsw_index.faiss")
print("Index has been loaded from 'hnsw_index.faiss'")

# Perform the search for each query to get top-K similar documents
# Function to perform the search and retrieve the top-K nearest neighbors
def search_index(index, query_embeddings, top_k=10):
    distances, indices = index.search(query_embeddings, top_k)  # Retrieve top_k nearest documents
    return distances, indices

# Perform search for top-10 documents for each query
top_k = 10
distances, indices = search_index(hnsw_index, query_embeddings, top_k=top_k)


# Print the number of queries
num_queries = query_embeddings.shape[0]
print("Number of queries:", num_queries)


# Output the indices and distances to files
# Save top-10 document indices for each query to a text file
with open(indices_file_path, "w") as indices_file:
    for i, doc_indices in enumerate(indices):
        # Use the actual query ID from query_ids, decode if it's a byte string
        query_id = query_ids[i].decode('utf-8') if isinstance(query_ids[i], bytes) else str(query_ids[i])
        doc_indices_str = ",".join(map(str, doc_indices))
        indices_file.write(f"{query_id}, {doc_indices_str}\n")

# Save top-10 distances for each query to a separate text file
with open(distances_file_path, "w") as distances_file:
    for i, doc_distances in enumerate(distances):
        # Use the actual query ID from query_ids, decode if it's a byte string
        query_id = query_ids[i].decode('utf-8') if isinstance(query_ids[i], bytes) else str(query_ids[i])
        doc_distances_str = ",".join(map(str, doc_distances))
        distances_file.write(f"{query_id}, {doc_distances_str}\n")


print(f"Top 10 document indices saved to {indices_file_path}")
print(f"Top 10 document distances saved to {distances_file_path}")