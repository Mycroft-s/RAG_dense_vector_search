from transformers import BertTokenizer, BertModel
import torch
import json
import numpy as np

# Load BM25 results from JSON file
with open("bm25_results.json", "r") as f:
    bm25_data = json.load(f)

query = bm25_data["query"]
top_k_documents = bm25_data["top_k_documents"]
top_k_bm25_scores = np.array(bm25_data["top_k_bm25_scores"])

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to encode text with BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).detach()  # CLS token, shape (768,)

# Encode the query with BERT
query_embedding = encode_text(query).unsqueeze(0)  # Shape: (1, 768)
print("Query embedding shape:", query_embedding.shape)

# Encode each of the top-k BM25 documents
doc_embeddings = torch.stack([encode_text(doc) for doc in top_k_documents])  # Shape: (3, 768)
print("Document embeddings shape:", doc_embeddings.shape)

# Expand query_embedding to match doc_embeddings' shape for cosine similarity
query_embedding = query_embedding.expand(doc_embeddings.size(0), -1)  # Shape: (3, 768)
print("Expanded query embedding shape:", query_embedding.shape)

# Compute cosine similarity between the query and each candidate document
cos = torch.nn.CosineSimilarity(dim=1)
bert_scores = cos(query_embedding, doc_embeddings).numpy()  # Now bert_scores should have shape (3,)
print("BERT scores shape:", bert_scores.shape)

# Combine BM25 scores and BERT scores (simple weighted combination)
combined_scores = top_k_bm25_scores + 0.5 * bert_scores  # Adjust weight as needed
print("Combined scores shape:", combined_scores.shape)

# Sort candidates by combined scores
reranked_indices = np.argsort(combined_scores)[::-1]
reranked_documents = [top_k_documents[i] for i in reranked_indices]
reranked_scores = combined_scores[reranked_indices]

# Print re-ranked results
print("Re-ranked documents:")
for doc, score in zip(reranked_documents, reranked_scores):
    print(f"Document: {doc}, Combined Score: {score}")
