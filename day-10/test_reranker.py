import pytest
import numpy as np

from day10_reranker import (
    DocumentCandidate,
    ProductionReRankingEngine,
    RerankRequestPayload,
    RankingError,
)


@pytest.mark.anyio
async def test_valid_rerank():
    engine = ProductionReRankingEngine()

    payload = RerankRequestPayload(
        query_vector=[1, 0, 0],
        candidates=[
            DocumentCandidate(
                id="a",
                text="SYSTEM TEXT",
                embedding=[1, 0, 0],
            ),
            DocumentCandidate(
                id="b",
                text="other",
                embedding=[0, 1, 0],
            ),
        ],
        top_n=1,
    )

    result = await engine.rerank_candidates(payload)

    assert len(result) == 1
    assert result[0].id == "a"


def test_empty_candidates():
    with pytest.raises(ValueError):
        RerankRequestPayload(
            query_vector=[1, 0],
            candidates=[],
        )


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        RerankRequestPayload(
            query_vector=[1, 0],
            candidates=[
                DocumentCandidate(
                    id="x",
                    text="x",
                    embedding=[1, 0, 0],
                )
            ],
        )


def test_nan_query():
    with pytest.raises(ValueError):
        RerankRequestPayload(
            query_vector=[1.0, np.nan],
            candidates=[
                DocumentCandidate(
                    id="x",
                    text="x",
                    embedding=[1, 0],
                )
            ],
        )


def test_inf_embedding():
    with pytest.raises(ValueError):
        DocumentCandidate(
            id="x",
            text="x",
            embedding=[1.0, np.inf],
        )


@pytest.mark.anyio
async def test_zero_norm_query():
    engine = ProductionReRankingEngine()

    payload = RerankRequestPayload(
        query_vector=[0.0, 0.0],
        candidates=[
            DocumentCandidate(
                id="x",
                text="x",
                embedding=[1.0, 0.0],
            )
        ],
    )

    result = await engine.rerank_candidates(payload)

    assert len(result) == 1
    assert result[0].raw_similarity == 0.0


def test_top_n_validation():
    with pytest.raises(ValueError):
        RerankRequestPayload(
            query_vector=[1, 0],
            candidates=[
                DocumentCandidate(
                    id="x",
                    text="x",
                    embedding=[1, 0],
                )
            ],
            top_n=1000,
        )


def test_text_limit():
    with pytest.raises(ValueError):
        DocumentCandidate(
            id="x",
            text="a" * 20000,
            embedding=[1, 0],
        )


@pytest.mark.anyio
async def test_similarity_bounds():
    engine = ProductionReRankingEngine()

    payload = RerankRequestPayload(
        query_vector=[1, 0],
        candidates=[
            DocumentCandidate(
                id="x",
                text="ABC",
                embedding=[-1, 0],
            )
        ],
    )

    result = await engine.rerank_candidates(payload)

    assert -1.0 <= result[0].raw_similarity <= 1.0


def test_invalid_id():
    with pytest.raises(ValueError):
        DocumentCandidate(
            id="",
            text="abc",
            embedding=[1, 0],
        )
