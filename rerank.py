from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to encode text with BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()  # CLS token

# Encode the query with BERT
query_embedding = encode_text(query)

# Encode each of the top-k BM25 documents
doc_embeddings = [encode_text(doc) for doc in top_k_documents]
doc_embeddings = torch.stack(doc_embeddings)

# Compute cosine similarity between the query and each candidate document
cos = torch.nn.CosineSimilarity(dim=1)
bert_scores = cos(query_embedding, doc_embeddings).numpy()

# Combine BM25 scores and BERT scores (simple weighted combination)
combined_scores = top_k_bm25_scores + 0.5 * bert_scores  # Adjust weight as needed

# Sort candidates by combined scores
reranked_indices = np.argsort(combined_scores)[::-1]
reranked_documents = [top_k_documents[i] for i in reranked_indices]
reranked_scores = combined_scores[reranked_indices]

# Print re-ranked results
print("Re-ranked documents:")
for doc, score in zip(reranked_documents, reranked_scores):
    print(f"Document: {doc}, Combined Score: {score}")
