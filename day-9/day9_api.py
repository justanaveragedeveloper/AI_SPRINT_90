import os
import time
import re
import uuid
import hashlib
import logging
import secrets
import tiktoken
from collections import deque, OrderedDict
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
import chromadb
from chromadb.utils import embedding_functions
import anyio
from functools import partial
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
)

logger = logging.getLogger(__name__)

ENCODING = tiktoken.get_encoding("cl100k_base")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ==========================================
# 1. PYDANTIC DATA VALIDATION SCHEMAS
# ==========================================


class UpsertRequest(BaseModel):
    id: str = Field(..., description="Unique identifier for the parent document")
    text: str = Field(
        ..., min_length=3, description="The text content to be vectorized and stored"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text cannot be empty")
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", stripped)
        dangerous_patterns = [r"\x00", r"\%[0-9a-fA-F]{2}", r"\\u[0-9a-fA-F]{4}"]
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, "", sanitized)
        if len(sanitized) < 2:
            raise ValueError("Text too short after sanitization")
        return sanitized

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("ID cannot be empty")
        safe_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "", stripped)
        if not safe_id or len(safe_id) < 2:
            raise ValueError("ID contains invalid characters")
        return safe_id

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        MAX_FLATTENED_KEYS = 200
        MAX_STRING_LENGTH = 1000

        def flatten_metadata(
            obj: Any, parent_key: str = "", sep: str = "."
        ) -> Dict[str, Any]:
            flattened = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    clean_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(k))[:50]
                    new_key = (
                        f"{parent_key}{sep}{clean_key}" if parent_key else clean_key
                    )
                    if isinstance(v, (dict, list)):
                        flattened.update(flatten_metadata(v, new_key, sep))
                    else:
                        if isinstance(v, str):
                            flattened[new_key] = v[:MAX_STRING_LENGTH]
                        elif isinstance(v, (int, float, bool)):
                            flattened[new_key] = v
                        else:
                            flattened[new_key] = str(v)[:MAX_STRING_LENGTH]
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
                    if isinstance(item, (dict, list)):
                        flattened.update(flatten_metadata(item, new_key, sep))
                    else:
                        if isinstance(item, str):
                            flattened[new_key] = item[:MAX_STRING_LENGTH]
                        elif isinstance(item, (int, float, bool)):
                            flattened[new_key] = item
                        else:
                            flattened[new_key] = str(item)[:MAX_STRING_LENGTH]
            return flattened

        _check_metadata_depth(v)
        flattened_metadata = flatten_metadata(v)
        if len(flattened_metadata) > MAX_FLATTENED_KEYS:
            raise ValueError(f"Metadata exceeded {MAX_FLATTENED_KEYS} flattened keys")
        return flattened_metadata


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=2, description="The semantic search query string"
    )
    n_results: int = Field(default=3, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty")
        clean_query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", stripped)
        if len(clean_query) < 2:
            raise ValueError("Query too short")
        return clean_query


# Configuration Constants
CHUNK_THRESHOLD = 1500
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOCUMENT = 50
MAX_TEXT_LENGTH = 50000
MAX_METADATA_DEPTH = 10

# Safe Secure Fallbacks for Local Verification Handshakes
load_dotenv()
API_KEY = os.getenv("API_KEY", "local-dev-key")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "local-admin-key")

if not API_KEY:
    raise RuntimeError("API_KEY missing")
if not ADMIN_API_KEY:
    raise RuntimeError("ADMIN_API_KEY missing")


RATE_LIMIT = 60
RATE_WINDOW = 60
MAX_TRACKED_IPS = 10000

# FIX 1: Use a bounded OrderedDict ring buffer to handle rate-limiting.
# This prevents synchronous table scans and avoids memory exhaustion vectors.
rate_store = OrderedDict()


def _check_metadata_depth(obj, depth=0):
    if depth > MAX_METADATA_DEPTH:
        raise ValueError("Metadata nesting too deep")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_metadata_depth(v, depth + 1)
    elif isinstance(obj, list):
        for i in obj:
            _check_metadata_depth(i, depth + 1)


# ==========================================
# 2. TEXT SPLITTER ENGINE
# ==========================================


ENCODING = tiktoken.get_encoding("cl100k_base")


def split_text_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    parent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Split text into token-aware chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Token overlap between chunks.
        parent_id: Parent document identifier.

    Returns:
        List of chunk dictionaries.

    Raises:
        ValueError: Invalid configuration.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = ENCODING.encode(text)

    chunks: List[Dict[str, Any]] = []

    start = 0
    chunk_index = 1

    while start < len(tokens):

        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]

        chunk_text = ENCODING.decode(chunk_tokens)

        chunks.append(
            {
                "id": (
                    f"{parent_id}_chunk_{chunk_index}"
                    if parent_id
                    else str(uuid.uuid4())
                ),
                "text": chunk_text,
                "metadata": {
                    "parent_id": parent_id,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                    "total_chunks": None,
                },
            }
        )

        chunk_index += 1

        start += chunk_size - chunk_overlap

    total_chunks = len(chunks)

    if total_chunks > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(f"Document exceeds {MAX_CHUNKS_PER_DOCUMENT} chunks")

    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = total_chunks

    return chunks


# ==========================================
# 3. ATOMIC DATABASE OPERATIONS
# ==========================================


class AtomicDBOperations:
    def __init__(self, collection):
        self.collection = collection

    def _generate_version_hash(self, parent_id: str, timestamp: int = None) -> str:
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        hash_input = f"{parent_id}_{timestamp}"
        # FIX 2: Do not truncate cryptographic hashes used for system indexing keys.
        # Retain the full string to guarantee uniqueness across vector indexes.
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def atomic_upsert_single(
        self, doc_id: str, document: str, metadata: Optional[Dict] = None
    ) -> None:
        try:
            self.collection.delete(ids=[doc_id])
        except Exception as e:
            logger.exception("Delete failed")
            raise

        if metadata is None:
            metadata = {}
        metadata["_version_hash"] = self._generate_version_hash(doc_id)
        metadata["parent_id"] = doc_id
        metadata["_updated_at"] = int(time.time())

        self.collection.upsert(documents=[document], ids=[doc_id], metadatas=[metadata])

    def atomic_upsert_bulk(
        self, ids: List[str], documents: List[str], metadatas: List[Dict]
    ) -> None:
        if not ids:
            return

        parent_id = None
        for doc_id in ids:
            if "_chunk_" in doc_id:
                parent_id = doc_id.split("_chunk_")[0]
                break

        if parent_id:
            try:
                self.collection.delete(where={"parent_id": parent_id})
            except Exception as e:
                logger.exception(
                    "Failed deleting parent document", extra={"parent_id": parent_id}
                )
                raise

        timestamp = int(time.time())
        version_hash = self._generate_version_hash(
            (parent_id or str(ids[0])) + "_" + str(uuid.uuid4()), timestamp
        )

        for metadata in metadatas:
            metadata["_version_hash"] = version_hash
            metadata["_updated_at"] = timestamp
            metadata["_batch_size"] = len(ids)

        new_ids = [f"{i}_{version_hash}" for i in ids]
        self.collection.upsert(documents=documents, ids=new_ids, metadatas=metadatas)


# ==========================================
# 4. APPLICATION INITIALIZATION
# ==========================================

app = FastAPI(
    title="Talent-IQ AI Core Gateway",
    description="Production Network Bridge with Atomic Operations & Schema Flattening",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.company.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

try:
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="talent_pool",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )
    atomic_db = AtomicDBOperations(collection)
    logger.info(
        "ChromaDB initialized successfully",
        extra={"db_path": DB_PATH},
    )
except Exception as e:
    logger.exception("Failed to initialize ChromaDB")
    raise RuntimeError("Could not connect to the persistent ChromaDB cluster.")


def get_document_count() -> int:
    try:
        return collection.count()
    except Exception:
        return 0


def _sync_upsert_operation(payload: UpsertRequest) -> Dict[str, Any]:
    text_length = len(payload.text)

    logger.info(
        "Upsert started",
        extra={
            "document_id": payload.id,
            "text_length": text_length,
        },
    )

    if text_length > MAX_TEXT_LENGTH:
        raise ValueError("Document exceeds maximum allowed size")

    if text_length <= CHUNK_THRESHOLD:
        atomic_db.atomic_upsert_single(
            doc_id=payload.id, document=payload.text, metadata=payload.metadata
        )

        logger.info(
            "Single document indexed",
            extra={
                "document_id": payload.id,
                "chunked": False,
                "chunks": 1,
            },
        )

        return {
            "status": "success",
            "parent_id": payload.id,
            "total_chunks_indexed": 1,
            "was_chunked": False,
            "original_length": text_length,
            "chunks": [{"id": payload.id, "chars": text_length}],
        }

    chunks = split_text_into_chunks(
        text=payload.text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        parent_id=payload.id,
    )
    chunk_ids = [chunk["id"] for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_metadatas = []

    for chunk in chunks:
        combined_metadata = {
            **(payload.metadata or {}),
            **chunk["metadata"],
            "is_chunk": True,
        }
        chunk_metadatas.append(combined_metadata)

    atomic_db.atomic_upsert_bulk(chunk_ids, chunk_texts, chunk_metadatas)

    logger.info(
        "Chunked document indexed",
        extra={
            "document_id": payload.id,
            "chunked": True,
            "chunks": len(chunks),
        },
    )

    chunks_summary = [
        {
            "id": chunk["id"],
            "chars": len(chunk["text"]),
            "index": chunk["metadata"]["chunk_index"],
            "no_of_tokens": chunk["metadata"]["token_count"],
        }
        for chunk in chunks
    ]

    return {
        "status": "success",
        "parent_id": payload.id,
        "total_chunks_indexed": len(chunks),
        "was_chunked": True,
        "original_length": text_length,
        "chunks": chunks_summary,
    }


def authenticate_request(request: Request) -> str:
    key = request.headers.get("X-API-Key")
    if not key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not (
        secrets.compare_digest(key, API_KEY)
        or secrets.compare_digest(key, ADMIN_API_KEY)
    ):
        logger.warning(
            "Authentication failed",
            extra={"ip": getattr(request.client, "host", "unknown")},
        )

        raise HTTPException(status_code=401, detail="Unauthorized")
    return key


def is_admin_key(api_key: str) -> bool:
    return secrets.compare_digest(api_key, ADMIN_API_KEY)


def check_rate_limit(request: Request) -> None:
    ip = getattr(request.client, "host", "unknown")
    now = time.time()
    if ip not in rate_store:
        if len(rate_store) >= MAX_TRACKED_IPS:
            rate_store.popitem(last=False)
        rate_store[ip] = deque()
    else:
        rate_store.move_to_end(ip)
    q = rate_store[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        logger.warning("Rate limit exceeded", extra={"ip": ip, "requests": len(q)})
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    q.append(now)


# ==========================================
# 5. API ROUTING ENDPOINTS
# ==========================================


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@app.post("/upsert", status_code=status.HTTP_201_CREATED)
async def upsert_document(payload: UpsertRequest, request: Request):
    request_id = str(uuid.uuid4())
    authenticate_request(request)
    check_rate_limit(request)
    logger.info(
        "Upsert request started",
        extra={"request_id": request_id, "document_id": payload.id},
    )
    try:
        sync_func = partial(_sync_upsert_operation, payload)
        result = await anyio.to_thread.run_sync(sync_func)
        logger.info(
            "Upsert request completed",
            extra={"request_id": request_id, "document_id": payload.id},
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}",
        )
    except Exception:
        logger.exception("Upsert failed")
        raise HTTPException(status_code=500, detail="Upsert operation failed")


@app.post("/search", status_code=status.HTTP_200_OK)
async def search_vectors(payload: SearchRequest, request: Request):
    request_id = str(uuid.uuid4())
    api_key = authenticate_request(request)
    check_rate_limit(request)
    is_admin = is_admin_key(api_key)
    logger.info("Search started", extra={"request_id": request_id})
    start_time = time.perf_counter()
    try:
        results = collection.query(
            query_texts=[payload.query], n_results=payload.n_results
        )
        formatted_results = []

        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                similarity_score = 0.0
                try:
                    raw_distance = results["distances"][0][i]
                    if raw_distance is not None and isinstance(
                        raw_distance, (int, float)
                    ):
                        similarity_score = round(1.0 - raw_distance, 4)
                        similarity_score = max(0.0, min(1.0, similarity_score))
                except Exception:
                    similarity_score = 0.0

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                clean_metadata = {
                    k: v for k, v in metadata.items() if not k.startswith("_")
                }

                if not is_admin:
                    sensitive_patterns = re.compile(
                        r"(email|phone|owner_id|tenant_id)", re.I
                    )
                    clean_metadata = {
                        k: v
                        for k, v in clean_metadata.items()
                        if not sensitive_patterns.search(k)
                    }

                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "document_snippet": results["documents"][0][i][:300],
                        "metadata": clean_metadata,
                        "similarity_score": similarity_score,
                    }
                )

        end_time = time.perf_counter()
        query_time_ms = round((end_time - start_time) * 1000, 2)
        logger.info(
            "Search completed",
            extra={
                "request_id": request_id,
                "query": payload.query[:50],
                "results": len(formatted_results),
                "latency_ms": query_time_ms,
            },
        )
        return {
            "query": payload.query,
            "total_matches": len(formatted_results),
            "query_time_ms": query_time_ms,
            "results": formatted_results,
        }
    except Exception as e:

        logger.exception("Search operation failed")

        raise HTTPException(status_code=500, detail="Search operation failed")


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def get_detailed_metrics(request: Request):
    authenticate_request(request)
    check_rate_limit(request)
    doc_count = get_document_count()
    return {
        "collection_name": "talent_pool",
        "total_documents": doc_count,
        "embedding_model": "all-MiniLM-L6-v2",
        "resource_limits": {
            "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
            "max_flattened_metadata_keys": 200,
            "max_metadata_string_length": 1000,
            "chunk_threshold": CHUNK_THRESHOLD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "metadata_schema": "flattened_dot_notation",
        "atomic_operations": "direct_where_clause",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
