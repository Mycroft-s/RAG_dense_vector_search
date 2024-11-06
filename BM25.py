from rank_bm25 import BM25Okapi
import numpy as np
import json

# Sample documents
documents = [
    "This is the first document.",
    "This is the second document.",
    "Another document about similar topics.",
    "This document is about machine learning and AI.",
    "Document related to search engines and information retrieval."
]

# Tokenize the documents for BM25
tokenized_docs = [doc.split(" ") for doc in documents]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Define the query
query = "information retrieval and search engines"
tokenized_query = query.split(" ")

# Get BM25 scores for the query
bm25_scores = bm25.get_scores(tokenized_query)

# Get top-k documents based on BM25 scores
top_k = 3  # You can adjust this value
top_k_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
top_k_bm25_scores = bm25_scores[top_k_bm25_indices]
top_k_documents = [documents[i] for i in top_k_bm25_indices]

# Save top-k documents and scores to a JSON file
bm25_results = {
    "query": query,
    "top_k_documents": top_k_documents,
    "top_k_bm25_scores": top_k_bm25_scores.tolist()
}

with open("bm25_results.json", "w") as f:
    json.dump(bm25_results, f)

print("BM25 ranking completed and saved to 'bm25_results.json'")
