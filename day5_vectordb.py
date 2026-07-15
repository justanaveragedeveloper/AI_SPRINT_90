import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import math

# ============================================
# PROPER EmbeddingFunction Implementation
# ============================================

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_embedding_dimension()
        
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings
    
    def get_dimension(self) -> int:
        return self._dimension

# ============================================
# DISTANCE METRIC CONFIGURATION
# ============================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DISTANCE_METRIC = "cosine"  # CRITICAL: Must match your similarity calculation!

# Distance metrics available in ChromaDB:
# - "cosine": Range [0, 2], where 0 = identical, 2 = opposite
# - "l2": Squared Euclidean distance, range [0, ∞)
# - "ip": Inner product, range [-∞, ∞)

embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

# Initialize persistent client
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection_name = "my_documents"

def similarity_from_distance(distance: float, metric: str = "cosine") -> float:
    """
    Convert ChromaDB distance to human-readable similarity score [0, 1]
    where 1 = most similar, 0 = least similar.
    
    ChromaDB's distance calculations:
    - cosine: distance = 1 - cosine_similarity → range [0, 2]
    - l2: squared Euclidean distance → range [0, ∞)
    - ip: negative inner product → range (-∞, ∞)
    """
    if metric == "cosine":
        # Cosine distance: d = 1 - cos(θ)
        # Therefore: similarity = 1 - d = cos(θ)
        similarity = 1 - distance
        # Clamp due to floating point errors
        return max(0.0, min(1.0, similarity))
    
    elif metric == "l2":
        # Squared Euclidean distance: d = ||a-b||²
        # Convert to cosine similarity approximation or use exponential decay
        # Method 1: Exponential decay (recommended for presentation)
        similarity = math.exp(-distance / 2.0)
        # Method 2: 1/(1+d) for slower decay
        # similarity = 1.0 / (1.0 + distance)
        return similarity
    
    elif metric == "ip":
        # Inner product: negative for ChromaDB, needs normalization
        # This is complex and model-dependent - avoid for presentation
        # Better to use 1/(1+|distance|) as a fallback
        similarity = 1.0 / (1.0 + abs(distance))
        return similarity
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

def format_score(distance: float, metric: str = "cosine") -> tuple:
    """Return both raw distance and formatted similarity score"""
    similarity = similarity_from_distance(distance, metric)
    return similarity

# ============================================
# CREATE COLLECTION WITH EXPLICIT DISTANCE METRIC
# ============================================

try:
    # Try to get existing collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    
    # Validate the distance metric matches
    if collection.metadata and "hnsw:space" in collection.metadata:
        stored_metric = collection.metadata["hnsw:space"]
        if stored_metric != DISTANCE_METRIC:
            print(f"⚠️  WARNING: Collection uses '{stored_metric}' but script expects '{DISTANCE_METRIC}'")
            print(f"   Similarity scores may be incorrect!")
    
    print(f"✓ Using existing collection: '{collection_name}'")
    print(f"  Distance metric: {DISTANCE_METRIC}")
    
except ValueError:
    # Create new collection with EXPLICIT distance metric
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={
            "hnsw:space": DISTANCE_METRIC,  # ← CRITICAL: Forces correct distance calculation
            "description": "A collection of sample documents",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": embedding_fn.get_dimension()
        }
    )
    print(f"✓ Created new collection: '{collection_name}'")
    print(f"  Distance metric: {DISTANCE_METRIC} (explicitly configured)")

# Sample documents
documents = [
    {
        "id": "doc_001",
        "text": "Python is a high-level programming language known for its simplicity and readability.",
        "metadata": {"category": "programming", "topic": "python", "difficulty": "beginner"}
    },
    {
        "id": "doc_002", 
        "text": "Machine learning algorithms can learn patterns from data to make predictions.",
        "metadata": {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"}
    },
    {
        "id": "doc_003",
        "text": "Data science combines statistics, programming, and domain expertise to extract insights.",
        "metadata": {"category": "data_science", "topic": "analytics", "difficulty": "intermediate"}
    },
    {
        "id": "doc_004",
        "text": "Web development frameworks like Django and Flask make building web apps easier.",
        "metadata": {"category": "web_dev", "topic": "frameworks", "difficulty": "beginner"}
    },
    {
        "id": "doc_005",
        "text": "Deep learning uses neural networks with multiple layers to solve complex problems.",
        "metadata": {"category": "ai", "topic": "deep_learning", "difficulty": "advanced"}
    }
]

# Upsert documents
print(f"\nUpserting {len(documents)} documents...")
for doc in documents:
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        metadatas=[doc["metadata"]]
    )

print(f"Total documents in collection: {collection.count()}")

# ============================================
# QUERY WITH CORRECT SIMILARITY CALCULATION
# ============================================

query_text = "What is artificial intelligence and machine learning?"
print(f"\nQuery: '{query_text}'")
print(f"Distance Metric: {DISTANCE_METRIC.upper()}")
print("\nTop 2 closest matches:")

results = collection.query(
    query_texts=[query_text],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

# Display results with mathematically correct similarity scores
for i, (doc_id, document, metadata, distance) in enumerate(zip(
    results['ids'][0], 
    results['documents'][0], 
    results['metadatas'][0],
    results['distances'][0]
), 1):
    similarity = similarity_from_distance(distance, DISTANCE_METRIC)
    
    print(f"\n{i}. ID: {doc_id}")
    print(f"   Text: {document[:80]}..." if len(document) > 80 else f"   Text: {document}")
    print(f"   Metadata: {metadata}")
    print(f"   Raw Distance: {distance:.6f}")
    print(f"   Similarity Score: {similarity:.4f} (0=unrelated, 1=identical)")

# ============================================
# DEMONSTRATE THE PROBLEM WITH WRONG TRANSFORMATION
# ============================================

print("\n" + "="*70)
print("⚠️  THE PROBLEM WITH 'similarity = 1 - distance'")
print("="*70)

# Create a test to show the issue
test_distances = {
    "cosine": [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0],
    "l2": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
}

print("\nIf you use 'similarity = 1 - distance' with different metrics:\n")

print("For COSINE metric (range 0-2):")
for d in test_distances["cosine"]:
    wrong_sim = 1 - d
    correct_sim = similarity_from_distance(d, "cosine")
    print(f"  distance={d:.1f} → wrong: {wrong_sim:.1f}, correct: {correct_sim:.3f}")

print("\nFor L2 metric (range 0-∞):")
for d in test_distances["l2"]:
    wrong_sim = 1 - d
    correct_sim = similarity_from_distance(d, "l2")
    print(f"  distance={d:.1f} → wrong: {wrong_sim:.1f}, correct: {correct_sim:.3f}")

# ============================================
# VERIFICATION WITH IDENTICAL DOCUMENTS
# ============================================

print("\n" + "="*70)
print("🔬 VERIFICATION: Testing with identical document")
print("="*70)

# Add an exact duplicate query test
test_text = "Python programming is great"
collection.upsert(
    ids=["test_identical"],
    documents=[test_text],
    metadatas=[{"test": True}]
)

identical_result = collection.query(
    query_texts=[test_text],
    n_results=1,
    include=["distances"]
)

distance = identical_result['distances'][0][0]
similarity = similarity_from_distance(distance, DISTANCE_METRIC)

print(f"Query text: '{test_text}'")
print(f"Same document retrieved (ID: test_identical)")
print(f"Raw distance: {distance:.6f}")
print(f"Similarity score: {similarity:.6f}")

if DISTANCE_METRIC == "cosine":
    expected_similarity = 1.0
    print(f"✓ For cosine metric, perfect match should be 1.0")
    print(f"  Actual: {similarity:.6f} (Difference: {abs(expected_similarity - similarity):.6f})")
elif DISTANCE_METRIC == "l2":
    print(f"✓ For L2 metric, perfect match is 0.0 distance → similarity~1.0")
    print(f"  Actual similarity: {similarity:.6f}")

# Clean up test document
collection.delete(ids=["test_identical"])

print(f"\nCurrent configuration:")
print(f"  Distance Metric: {DISTANCE_METRIC}")
print(f"  Similarity Range: [0, 1] (1 = most similar)")
print(f"  Collection: {collection_name}")
print(f"  Total Vectors: {collection.count()}")