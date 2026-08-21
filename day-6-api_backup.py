import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. PYDANTIC DATA VALIDATION SCHEMAS
# ==========================================

class UpsertRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for the document (e.g., user_id or job_id)")
    text: str = Field(..., min_length=3, description="The text content to be vectorized and stored")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata key-value pairs")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The semantic search query string")
    n_results: int = Field(default=3, ge=1, le=20, description="Number of top matches to return")

# ==========================================
# 2. APPLICATION INITIALIZATION & CORS
# ==========================================

app = FastAPI(
    title="Talent-IQ AI Core Gateway",
    description="Production Network Bridge for Semantic Matching and Embedding Management",
    version="1.0.0"
)

# Enable CORS for your MERN Stack (Adjust origins in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all network headers
)

# ==========================================
# 3. DATABASE & EMBEDDING LIFECYCLE
# ==========================================

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

try:
    # Initialize Persistent Client and Embedding Function
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Using the same sentence-transformers model utilized in previous steps
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create the collection
    collection = chroma_client.get_or_create_collection(
        name="talent_pool",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Forces mathematically normalized cosine distance
    )
    print(f"✅ Successfully mounted ChromaDB from: {DB_PATH}")
except Exception as e:
    print(f"❌ Database initialization failure: {str(e)}")
    raise RuntimeError("Could not connect to the persistent ChromaDB cluster.")

# ==========================================
# 4. API ROUTING ENDPOINTS
# ==========================================

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Verifies server health and database connection."""
    return {"status": "healthy", "database_connected": True}


@app.post("/upsert", status_code=status.HTTP_201_CREATED)
async def upsert_document(payload: UpsertRequest):
    """
    Validates, vectorizes, and stores a document chunk with optional metadata.
    """
    try:
        # ChromaDB accepts lists for bulk operations; wrapping our single payload item
        collection.upsert(
            documents=[payload.text],
            ids=[payload.id],
            metadatas=[payload.metadata] if payload.metadata else None
        )
        return {
            "status": "success",
            "message": f"Document ID '{payload.id}' successfully indexed into vector space."
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upsert payload into ChromaDB: {str(e)}"
        )


@app.post("/search", status_code=status.HTTP_200_OK)
async def search_vectors(payload: SearchRequest):
    """
    Performs a semantic mathematical vector search against stored indexes.
    Returns clean results with similarity scores mapped from [0, 1].
    """
    try:
        results = collection.query(
            query_texts=[payload.query],
            n_results=payload.n_results
        )
        
        # Parse the nested lists returned by ChromaDB into a clean, readable API structure
        formatted_results = []
        
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                # ChromaDB cosine returns distance. Similarity score = 1 - distance
                raw_distance = results["distances"][0][i]
                similarity_score = round(1.0 - raw_distance, 4)
                
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity_score": max(0.0, min(1.0, similarity_score)) # Clamp between 0 and 1
                })
                
        return {
            "query": payload.query,
            "total_matches": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search engine encountered an error: {str(e)}"
        )

# ==========================================
# 5. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Host on localhost, port 8000
    uvicorn.run("day6_api.py:app", host="127.0.0.1", port=8000, reload=True)