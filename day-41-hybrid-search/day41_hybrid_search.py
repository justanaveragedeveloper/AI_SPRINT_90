"""
Day 41 – Hybrid Search & RAG Retrieval Pipeline

First‑principles implementation of:
- BM25 sparse lexical retrieval
- Dense vector similarity (cosine)
- Reciprocal Rank Fusion (RRF) for merging ranked lists

All operations include defensive guardrails, logging, and full type hints.
"""

import logging
import math
from collections import Counter, defaultdict

import numpy as np

# Configure module‑level logger
logger = logging.getLogger(__name__)


class BM25Okapi:
    """
    First‑principles BM25 implementation for lexical document scoring.

    The BM25 score for a document D and a query Q is:
        BM25(D,Q) = Σ_{q in Q} IDF(q) * [f(q,D)*(k1+1)] / [f(q,D) + k1*(1-b+b*|D|/avgdl)]

    This implementation follows the standard Okapi formulation.

    Attributes:
        k1 (float): Term frequency saturation parameter (must be > 0).
        b (float): Document length normalisation parameter (0 ≤ b ≤ 1).
        corpus_size (int): Number of documents in the index.
        doc_lengths (list[int]): Length (in tokens) of each document.
        avg_doc_length (float): Average document length over the corpus.
        doc_term_freqs (list[Counter]): Term frequency counters per document.
        doc_freq (dict[str, int]): Number of documents containing each term.
        idf (dict[str, float]): Inverse document frequency for each term.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        """
        Initialise BM25 from a tokenised corpus.

        Args:
            corpus: List of tokenised documents (each document is a list of strings).
            k1: Term frequency saturation parameter (must be > 0 and finite).
            b: Length normalisation parameter (must be between 0 and 1, finite).

        Raises:
            ValueError: If k1 <= 0, b outside [0,1], or either is not finite.
        """
        # Validate parameters
        if not math.isfinite(k1) or not math.isfinite(b):
            raise ValueError(f"k1 and b must be finite numbers, got k1={k1}, b={b}")
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not (0 <= b <= 1):
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]

        # Handle empty corpus gracefully
        if self.corpus_size == 0:
            logger.warning("BM25 initialised with empty corpus.")
            self.avg_doc_length = 0.0
            self.doc_term_freqs = []
            self.doc_freq = {}
            self.idf = {}
            return

        # Compute average document length
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size

        # Build per‑document term frequency counters and global document frequencies
        self.doc_term_freqs = []          # list of Counter objects
        self.doc_freq = defaultdict(int)  # term -> number of documents containing it

        for doc in corpus:
            term_counts = Counter(doc)
            self.doc_term_freqs.append(term_counts)
            # Each term contributes exactly 1 to the document frequency per document
            for term in term_counts:
                self.doc_freq[term] += 1

        # Compute IDF for all terms using the standard BM25 formula
        self.idf = {}
        for term, df in self.doc_freq.items():
            # IDF(q) = log( (N - df + 0.5) / (df + 0.5) + 1 )
            idf_val = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            # Clamp to non‑negative (some variants allow negative, but we follow the spec)
            self.idf[term] = max(idf_val, 0.0)

        logger.info(
            "BM25 initialised with %d documents, %d unique terms, avgdl=%.2f",
            self.corpus_size,
            len(self.idf),
            self.avg_doc_length,
        )

    def get_scores(self, query: list[str]) -> list[float]:
        """
        Compute BM25 scores for all documents given a tokenised query.

        Args:
            query: List of query terms (strings).

        Returns:
            list[float]: Scores for each document in corpus order.
                         Empty list if corpus is empty.
        """
        if self.corpus_size == 0:
            return []

        # Keep only query terms that actually appear in the index
        query_terms = [term for term in query if term in self.idf]
        if not query_terms:
            return [0.0] * self.corpus_size

        # Pre‑compute the document‑length factor for each document:
        # factor = 1 - b + b * (doc_length / avg_doc_length)
        doc_len_factors = []
        for doc_len in self.doc_lengths:
            if self.avg_doc_length == 0:
                factor = 1.0
            else:
                factor = 1 - self.b + self.b * (doc_len / self.avg_doc_length)
            doc_len_factors.append(factor)

        scores = [0.0] * self.corpus_size

        # For each query term, add its contribution to every document
        for term in query_terms:
            idf = self.idf[term]
            for doc_idx, term_counts in enumerate(self.doc_term_freqs):
                if term not in term_counts:
                    continue
                tf = term_counts[term]  # term frequency in this document
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * doc_len_factors[doc_idx]
                scores[doc_idx] += idf * (numerator / denominator)

        return scores


class HybridSearchEngine:
    """
    Combines BM25 and dense cosine similarity via Reciprocal Rank Fusion (RRF).

    The engine tokenises documents with simple whitespace/lowercase splitting.
    Dense embeddings are provided externally (e.g., from a transformer model).
    """

    def __init__(self, documents: list[str], embeddings: list[list[float]]):
        """
        Initialise the hybrid search engine.

        Args:
            documents: List of raw document strings.
            embeddings: List of dense vector representations (same order as documents).

        Raises:
            ValueError: If document/embedding count mismatch, empty inputs,
                        inconsistent embedding dimensions, or non‑finite values.
        """
        # Basic input validation
        if not documents:
            raise ValueError("Documents list cannot be empty.")
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty.")
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) does not match "
                f"number of embeddings ({len(embeddings)})."
            )

        # Validate embedding dimensions and finite values
        emb_dim = len(embeddings[0])
        if emb_dim == 0:
            raise ValueError("Embedding dimension must be > 0.")
        for i, emb in enumerate(embeddings):
            if len(emb) != emb_dim:
                raise ValueError(
                    f"Embedding at index {i} has dimension {len(emb)}, "
                    f"expected {emb_dim}."
                )
            if not np.all(np.isfinite(emb)):
                raise ValueError(f"Embedding at index {i} contains non‑finite values (inf/nan).")

        # Store documents and embeddings (as a NumPy array for efficient computation)
        self.documents = documents
        self.embeddings = np.array(embeddings, dtype=np.float64)
        self.embedding_dim = emb_dim
        self.num_docs = len(documents)

        # Tokenise documents for BM25 (simple lowercase + whitespace split)
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            "HybridSearchEngine initialised with %d documents and embedding dimension %d",
            self.num_docs,
            self.embedding_dim,
        )

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns 0.0 if either vector has zero norm.

        Args:
            v1, v2: Lists of floats (same length).

        Returns:
            float: Cosine similarity.

        Raises:
            ValueError: If vectors have different lengths or contain non‑finite values.
        """
        if len(v1) != len(v2):
            raise ValueError(
                f"Vector dimension mismatch: {len(v1)} vs {len(v2)}"
            )
        if not np.all(np.isfinite(v1)) or not np.all(np.isfinite(v2)):
            raise ValueError("Vectors must contain only finite numbers.")

        a = np.array(v1, dtype=np.float64)
        b = np.array(v2, dtype=np.float64)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _validate_query_vector(self, query_vector: list[float]) -> None:
        """
        Validate a query vector for dense search.

        Raises ValueError if dimension mismatch or non‑finite values.
        """
        if len(query_vector) != self.embedding_dim:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"embedding dimension {self.embedding_dim}."
            )
        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query vector contains non‑finite values.")

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Perform BM25 lexical search.

        Args:
            query: Raw query string.
            top_k: Number of top results to return (if > corpus size, returns all).

        Returns:
            list of (document_index, bm25_score) sorted descending by score,
            ties broken by lower index first.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Sort by score descending, then by index ascending (deterministic tie‑break)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], x[0])
        )
        return ranked[:min(top_k, self.num_docs)]

    def search_dense(self, query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """
        Perform dense vector search using cosine similarity.

        Args:
            query_vector: Dense query embedding.
            top_k: Number of top results to return (if > corpus size, returns all).

        Returns:
            list of (document_index, cosine_similarity) sorted descending,
            ties broken by lower index first.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self._validate_query_vector(query_vector)

        q = np.array(query_vector, dtype=np.float64)
        norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(q)

        if q_norm == 0.0:
            # Zero query vector => all similarities are 0
            scores = np.zeros(self.num_docs, dtype=np.float64)
        else:
            # Compute dot products and divide by norms
            dots = np.dot(self.embeddings, q)
            with np.errstate(divide='ignore', invalid='ignore'):
                scores = np.where(norms > 0, dots / (norms * q_norm), 0.0)
            # If any score is non‑finite, something went wrong
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("Computed dense scores contain non‑finite values.")
            # Defensive cleanup (though we already checked)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: (-x[1], x[0])
        )
        return ranked[:min(top_k, self.num_docs)]

    def search_hybrid_rrf(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        rrf_k: int = 60,
        candidate_k: int | None = None,
    ) -> list[tuple[int, float, str]]:
        """
        Perform hybrid retrieval combining BM25 and dense cosine similarity via RRF.

        The method retrieves candidate lists from each system and fuses them using
        Reciprocal Rank Fusion. If `candidate_k` is provided, each retriever returns
        only the top‑candidate_k results before fusion – this is closer to production
        practice. If `candidate_k` is None (default), the method retrieves all
        documents from both systems (educational full‑ranking behaviour).

        Args:
            query_text: Raw query string for BM25.
            query_vector: Dense query embedding.
            top_k: Number of final results to return.
            rrf_k: Smoothing constant for RRF (default 60, must be > 0 and finite).
            candidate_k: Number of candidates to retrieve from each system before fusion.
                         If None, uses all documents.

        Returns:
            list of (document_index, fused_rrf_score, document_text) sorted by
            descending RRF score, ties broken by lower index.

        Raises:
            ValueError: If rrf_k <= 0 or not finite, top_k invalid, or candidate_k invalid.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not math.isfinite(rrf_k):
            raise ValueError(f"rrf_k must be a finite number, got {rrf_k}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
        if candidate_k is not None and (not isinstance(candidate_k, int) or candidate_k < 1):
            raise ValueError(f"candidate_k must be a positive integer or None, got {candidate_k}")

        # Determine how many candidates to fetch per system
        fetch_k = self.num_docs if candidate_k is None else min(candidate_k, self.num_docs)

        # Get ranked lists from both retrieval systems
        bm25_results = self.search_bm25(query_text, top_k=fetch_k)
        dense_results = self.search_dense(query_vector, top_k=fetch_k)

        # Accumulate RRF scores
        rrf_scores: dict[int, float] = defaultdict(float)

        # BM25 contributions: rank 1 → 1/(k+1), rank 2 → 1/(k+2), ...
        for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank)

        # Dense contributions
        for rank, (doc_idx, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank)

        # Sort by RRF score descending, then by index ascending
        sorted_rrf = sorted(
            rrf_scores.items(),
            key=lambda item: (-item[1], item[0])
        )

        # Build final payload
        results = []
        for doc_idx, score in sorted_rrf[:top_k]:
            results.append((doc_idx, score, self.documents[doc_idx]))

        logger.info(
            "Hybrid search returned %d results (top_k=%d, candidate_k=%s)",
            len(results),
            top_k,
            candidate_k if candidate_k is not None else "full",
        )
        return results