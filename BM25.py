from rank_bm25 import BM25Okapi
import numpy as np

# Example documents
documents = ["This is the first document.", "This is the second document.", "Another document about similar topics."]
tokenized_docs = [doc.split(" ") for doc in documents]

# Initialize BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Query example
query = "first document"
tokenized_query = query.split(" ")

# BM25 scoring
bm25_scores = bm25.get_scores(tokenized_query)

# Get top 10 documents based on BM25 scores
top_k_bm25_indices = np.argsort(bm25_scores)[::-1][:10]  # Change 10 to another number if you want more candidates
top_k_bm25_scores = bm25_scores[top_k_bm25_indices]
top_k_documents = [documents[i] for i in top_k_bm25_indices]
