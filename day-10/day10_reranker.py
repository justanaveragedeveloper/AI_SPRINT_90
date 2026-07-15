"""
day10_reranker.py
Production-style semantic reranking engine (single-file version).
"""

from __future__ import annotations

import logging
import re
import time
from functools import partial
from typing import Any, Dict, List, Tuple

import anyio
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class EngineConfig:
    MIN_VECTOR_DIMENSION = 2
    MAX_VECTOR_DIMENSION = 4096
    MAX_CANDIDATES = 1000
    MAX_TEXT_LENGTH = 10000
    MAX_TOP_N = 100
    FLOAT_DTYPE = np.float32


class RerankerError(Exception):
    pass


class RankingError(RerankerError):
    pass


class DocumentCandidate(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=EngineConfig.MAX_TEXT_LENGTH)
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]):
        if not (
            EngineConfig.MIN_VECTOR_DIMENSION
            <= len(v)
            <= EngineConfig.MAX_VECTOR_DIMENSION
        ):
            raise ValueError("Invalid embedding dimension")

        if not all(np.isfinite(x) for x in v):
            raise ValueError("Embedding contains NaN or Inf")

        return v


class RerankRequestPayload(BaseModel):
    query_vector: List[float]
    candidates: List[DocumentCandidate]
    top_n: int = Field(default=3, ge=1, le=EngineConfig.MAX_TOP_N)

    @field_validator("query_vector")
    @classmethod
    def validate_query_vector(cls, v: List[float]):
        if not (
            EngineConfig.MIN_VECTOR_DIMENSION
            <= len(v)
            <= EngineConfig.MAX_VECTOR_DIMENSION
        ):
            raise ValueError("Invalid query dimension")

        if not all(np.isfinite(x) for x in v):
            raise ValueError("Query contains NaN or Inf")

        return v

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, v):
        if not v:
            raise ValueError("Candidates cannot be empty")

        if len(v) > EngineConfig.MAX_CANDIDATES:
            raise ValueError("Too many candidates")

        return v

    @model_validator(mode="after")
    def validate_dimensions(self):
        query_dim = len(self.query_vector)

        for candidate in self.candidates:
            if len(candidate.embedding) != query_dim:
                raise ValueError(
                    f"Dimension mismatch. Query={query_dim}, Candidate={len(candidate.embedding)}"
                )


class RerankResultItem(BaseModel):
    id: str
    text: str
    raw_similarity: float
    refined_score: float
    rank: int
    metadata: Dict[str, Any]


class VectorMathCore:
    @staticmethod
    def cosine_similarity_matrix(
        query: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:

        epsilon = 1e-12

        query_norm = np.linalg.norm(query)

        if query_norm < epsilon:
            return np.zeros(matrix.shape[0], dtype=EngineConfig.FLOAT_DTYPE)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        matrix_norms = np.where(matrix_norms < epsilon, epsilon, matrix_norms)

        dot_products = matrix @ query

        similarities = dot_products / (query_norm * matrix_norms)

        return np.clip(similarities, -1.0, 1.0)


class HeuristicScorer:
    @staticmethod
    def score(text: str) -> float:
        caps = len(re.findall(r"[A-Z]{2,}", text))
        return min(caps * 0.02, 0.20)


class ProductionReRankingEngine:
    def __init__(self, cross_encoder_weight: float = 0.30):
        self.weight = float(max(0.0, min(1.0, cross_encoder_weight)))
        self.math_core = VectorMathCore()

    def _rerank_sync(
        self,
        query: List[float],
        candidates: List[DocumentCandidate],
        top_n: int,
    ) -> List[RerankResultItem]:

        start = time.perf_counter()

        np_query = np.asarray(query, dtype=EngineConfig.FLOAT_DTYPE)
        np_matrix = np.asarray(
            [c.embedding for c in candidates],
            dtype=EngineConfig.FLOAT_DTYPE,
        )

        similarities = self.math_core.cosine_similarity_matrix(
            np_query,
            np_matrix,
        )

        scored: List[Tuple[float, float, DocumentCandidate]] = []

        for idx, sim in enumerate(similarities):
            candidate = candidates[idx]

            bonus = HeuristicScorer.score(candidate.text)

            refined = (float(sim) * (1.0 - self.weight)) + (bonus * self.weight)

            refined = max(-1.0, min(1.0, refined))

            scored.append((float(sim), refined, candidate))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []

        for rank, (raw, refined, candidate) in enumerate(
            scored[:top_n],
            start=1,
        ):
            results.append(
                RerankResultItem(
                    id=candidate.id,
                    text=candidate.text,
                    raw_similarity=round(raw, 4),
                    refined_score=round(refined, 4),
                    rank=rank,
                    metadata=candidate.metadata,
                )
            )

        logger.info(
            "Reranked %s candidates in %.2fms",
            len(candidates),
            (time.perf_counter() - start) * 1000,
        )

        return results

    async def rerank_candidates(
        self,
        payload: RerankRequestPayload,
    ) -> List[RerankResultItem]:
        try:
            sync_call = partial(
                self._rerank_sync,
                payload.query_vector,
                payload.candidates,
                payload.top_n,
            )

            return await anyio.to_thread.run_sync(sync_call)

        except Exception as exc:
            logger.exception("Reranking failure")
            raise RankingError("Internal reranking failure") from exc
