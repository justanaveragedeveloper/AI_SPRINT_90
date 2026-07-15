import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents to search through
documents = [
    "The sky appears blue because of Rayleigh scattering of sunlight.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Photosynthesis is the process plants use to convert sunlight into energy.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are inspired by the human brain's structure.",
    "Jupiter is the largest planet in our solar system."
]

# Encode all documents to vectors
document_embeddings = model.encode(documents, convert_to_tensor=True)

def semantic_search(query, documents, document_embeddings, top_k=3):
    """
    Perform semantic search using cosine similarity.
    
    Args:
        query: Search query string
        documents: List of document strings
        document_embeddings: Pre-computed embeddings for documents
        top_k: Number of top results to return
    
    Returns:
        List of tuples (document, similarity_score)
    """
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarity between query and all documents
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    
    # Get top_k results using torch (now properly imported)
    top_results = torch.topk(cosine_scores, k=min(top_k, len(documents)))
    
    # Return documents with their similarity scores
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append((documents[idx], score.item()))
    
    return results

# Example usage
query = "How do plants make energy?"

print(f"Query: '{query}'\n")
print("Top 3 most semantically similar documents:")
print("-" * 60)

results = semantic_search(query, documents, document_embeddings, top_k=3)

for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f}")
    print(f"   Document: {doc}\n")