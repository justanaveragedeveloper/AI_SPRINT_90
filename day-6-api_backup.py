import importlib
import os

try:
    chromadb = importlib.import_module("chromadb")
except ModuleNotFoundError as exc:
    raise RuntimeError("Install ChromaDB to run this application (pip install chromadb).") from exc
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========================================
# 1. PYDANTIC DATA VALIDATION SCHEMAS
# ==========================================

class UpsertRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for the document (e.g., user_id or job_id)")
    text: str = Field(..., min_length=3, description="The text content to be vectorized and stored")
    class Any:
        ...

    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata key-value pairs")

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
    
    # ChromaDB's default embedding function uses all-MiniLM-L6-v2.
    collection = chroma_client.get_or_create_collection(
        name="talent_pool",
        metadata={"hnsw:space": "cosine"} # Forces mathematically normalized cosine distance
    )
    print(f"✅ Successfully mounted ChromaDB from: {DB_PATH}")
except (chromadb.errors.ChromaError, OSError, ValueError, RuntimeError, ImportError) as e:
    print(f"❌ Database initialization failure: {e!s}")
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
    except chromadb.errors.ChromaError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upsert payload into ChromaDB: {e!s}"
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
        
    except (chromadb.errors.ChromaError, ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search engine encountered an error: {e!s}"
        )

# ==========================================
# 5. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    import importlib

    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install uvicorn to run this application (pip install uvicorn).") from exc

    # Host on localhost, port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)