"""
Comprehensive pytest suite for Day 41 Hybrid Search.
Covers BM25, dense similarity, RRF, integration, and defensive failures.
"""

import math

import pytest
from day41_hybrid_search import BM25Okapi, HybridSearchEngine

# ----------------------------------------------------------------------
# BM25 Tests
# ----------------------------------------------------------------------

def test_bm25_positive_scores():
    """BM25 should give positive scores for documents containing query terms."""
    corpus = [
        ["deep", "learning", "autograd"],
        ["machine", "learning", "supervised"],
        ["deep", "neural", "networks"]
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["deep", "learning"])
    assert scores[0] > 0.0
    assert scores[1] > 0.0
    assert scores[2] > 0.0  # has "deep"


def test_bm25_zero_for_no_match():
    """Documents with no query terms should get zero score."""
    corpus = [["a", "b"], ["c", "d"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["x", "y"])
    assert all(s == 0.0 for s in scores)


def test_bm25_hand_computable_idf():
    """Check IDF values against manual calculation."""
    corpus = [["cat", "dog"], ["dog"]]
    bm25 = BM25Okapi(corpus)
    idf_cat = bm25.idf["cat"]
    idf_dog = bm25.idf["dog"]
    # N=2, df(cat)=1, df(dog)=2
    # IDF(cat) = log((2-1+0.5)/(1+0.5)+1) = log(2)
    # IDF(dog) = log((2-2+0.5)/(2+0.5)+1) = log(1.2)
    assert abs(idf_cat - math.log(2)) < 1e-6
    assert abs(idf_dog - math.log(1.2)) < 1e-6


def test_bm25_term_frequency_and_doc_len():
    """TF and document length should affect scores as expected."""
    corpus = [["term"] * 3, ["term"] * 1, ["other"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["term"])
    # Document with higher TF should score higher
    assert scores[0] > scores[1]
    # Document without the term gets zero
    assert scores[2] == 0.0


def test_bm25_empty_corpus():
    """Empty corpus should be handled gracefully."""
    bm25 = BM25Okapi([])
    assert bm25.get_scores(["query"]) == []
    with pytest.raises(ValueError, match="k1 must be > 0"):
        BM25Okapi([["a"]], k1=0)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Okapi([["a"]], b=1.5)


def test_bm25_invalid_params():
    """Invalid k1 or b should raise ValueError."""
    with pytest.raises(ValueError, match="k1 must be > 0"):
        BM25Okapi([["a"]], k1=-1.0)
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Okapi([["a"]], b=2.0)


def test_bm25_duplicate_query_terms():
    """Duplicate query terms should be summed (per our implementation)."""
    corpus = [["term", "term", "other"], ["term", "other"]]
    bm25 = BM25Okapi(corpus)
    scores_once = bm25.get_scores(["term"])
    scores_twice = bm25.get_scores(["term", "term"])
    assert scores_twice[0] == 2 * scores_once[0]
    assert scores_twice[1] == 2 * scores_once[1]


def test_bm25_finite_validation():
    """BM25 parameters must be finite numbers."""
    with pytest.raises(ValueError, match="must be finite"):
        BM25Okapi([["a"]], k1=float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        BM25Okapi([["a"]], b=float("inf"))


# ----------------------------------------------------------------------
# Dense Similarity Tests
# ----------------------------------------------------------------------

def test_cosine_known_values():
    """Check cosine similarity for simple known cases."""
    engine = HybridSearchEngine(["dummy"], [[1.0, 0.0]])
    assert engine.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert engine.cosine_similarity([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert engine.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    # Exact value for known vectors
    v1 = [1.0, 1.0]
    v2 = [0.0, 1.0]
    expected = 1.0 / math.sqrt(2)  # 0.70710678
    assert engine.cosine_similarity(v1, v2) == pytest.approx(expected)


def test_dense_ranking():
    """Dense search should rank documents correctly."""
    docs = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ]
    engine = HybridSearchEngine(docs, embeddings)
    query = [1.0, 0.0]
    results = engine.search_dense(query, top_k=3)
    # Expected order: doc0 (cos=1), doc2 (cos=0.707), doc1 (cos=0)
    assert [idx for idx, _ in results] == [0, 2, 1]
    # Check exact cosine similarity for doc2
    expected = 0.5 / math.sqrt(0.5)  # = 0.70710678
    assert results[1][1] == pytest.approx(expected)


def test_dense_zero_vectors():
    """Zero‑norm vectors should yield cosine similarity 0.0."""
    engine = HybridSearchEngine(["zero doc"], [[0.0, 0.0]])
    assert engine.cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert engine.cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0


def test_dense_dimension_mismatch():
    """Query vector with wrong dimension should raise ValueError."""
    engine = HybridSearchEngine(["doc"], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="does not match embedding dimension"):
        engine.search_dense([1.0, 2.0, 3.0], top_k=1)


def test_dense_non_finite():
    """Non‑finite embeddings or query vectors should be rejected."""
    with pytest.raises(ValueError, match="non‑finite"):
        HybridSearchEngine(["bad"], [[float('nan'), 1.0]])
    engine = HybridSearchEngine(["doc"], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="non‑finite"):
        engine.search_dense([float('inf'), 0.0], top_k=1)


def test_dense_zero_query_ranking():
    """Zero query vector gives all scores zero; tie‑break by index."""
    docs = ["a", "b"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    engine = HybridSearchEngine(docs, embeddings)
    results = engine.search_dense([0.0, 0.0], top_k=2)
    assert [idx for idx, _ in results] == [0, 1]
    assert results[0][1] == 0.0
    assert results[1][1] == 0.0


# ----------------------------------------------------------------------
# RRF Tests
# ----------------------------------------------------------------------

def test_rrf_hand_computable():
    """Test the RRF formula with manually computed ranks and k=1."""
    k = 1
    ranks1 = {0: 1, 1: 2}
    ranks2 = {0: 2, 1: 1}
    rrf_scores = {}
    for doc in (0, 1):
        rrf_scores[doc] = 1.0/(k + ranks1[doc]) + 1.0/(k + ranks2[doc])
    # Both docs have the same RRF score: 5/6
    assert abs(rrf_scores[0] - 5.0/6.0) < 1e-9
    assert abs(rrf_scores[1] - 5.0/6.0) < 1e-9


def test_rrf_exact_fusion():
    """End‑to‑end RRF fusion through the engine, with exact expected scores."""
    docs = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.70710678, 0.70710678]  # cos similarity with [0,1] = 0.707
    ]
    engine = HybridSearchEngine(docs, embeddings)
    # Use rrf_k=1 for easy math
    results = engine.search_hybrid_rrf("a", [0.0, 1.0], top_k=3, rrf_k=1)
    # BM25 ranks: doc0 rank1, doc1 rank2, doc2 rank3 (doc2 has no match)
    # Dense ranks: doc1 (cos=1), doc2 (cos=0.707), doc0 (cos=0)
    # RRF scores:
    # doc0: 1/2 + 1/4 = 0.75
    # doc1: 1/3 + 1/2 = 0.8333
    # doc2: 1/4 + 1/3 = 0.5833
    expected_order = [1, 0, 2]
    actual_order = [idx for idx, _, _ in results]
    assert actual_order == expected_order
    expected_scores = {
        0: 1.0/2.0 + 1.0/4.0,
        1: 1.0/3.0 + 1.0/2.0,
        2: 1.0/4.0 + 1.0/3.0,
    }
    for idx, score in expected_scores.items():
        for res_idx, res_score, _ in results:
            if res_idx == idx:
                assert res_score == pytest.approx(score)
                break


def test_rrf_with_candidate_k():
    """Test that candidate_k limits the candidates fused from each system."""
    docs = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ]
    engine = HybridSearchEngine(docs, embeddings)
    results = engine.search_hybrid_rrf("a", [0.0, 1.0], top_k=3, rrf_k=60, candidate_k=2)
    # BM25 top2: doc0 (rank1), doc1 (rank2)
    # Dense top2: doc1 (rank1), doc2 (rank2)
    # doc0 only appears in BM25 -> 1/61
    # doc1 appears in both -> 1/62 + 1/61
    # doc2 only appears in dense -> 1/62
    expected_order = [1, 0, 2]  # doc1 highest, then doc0, then doc2
    assert [idx for idx, _, _ in results] == expected_order
    expected = {
        0: 1.0/61.0,
        1: 1.0/62.0 + 1.0/61.0,
        2: 1.0/62.0,
    }
    for idx, score in expected.items():
        for r_idx, r_score, _ in results:
            if r_idx == idx:
                assert r_score == pytest.approx(score)


def test_rrf_doc_in_one_list_only_full_ranking():
    """With full ranking, every doc appears in both lists; this test documents that."""
    docs = ["doc0", "doc1", "doc2"]
    engine = HybridSearchEngine(
        docs,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    query_text = "apple banana"  # no match -> all BM25 scores zero
    query_vec = [0.0, 0.0, 1.0]  # closest to doc2

    results = engine.search_hybrid_rrf(query_text, query_vec, top_k=3, rrf_k=60)
    # BM25 ranks: doc0, doc1, doc2 (all score 0, order by index)
    # Dense ranks: doc2 (1), doc0 (2), doc1 (3)
    # Order: doc0 (1/61+1/62), doc2 (1/63+1/61), doc1 (1/62+1/63)
    expected_order = [0, 2, 1]
    assert [idx for idx, _, _ in results] == expected_order


def test_rrf_doc_in_one_list_only_with_candidate_k():
    """Using candidate_k, we can truly have a doc appear in only one list."""
    docs = ["doc0", "doc1", "doc2"]
    engine = HybridSearchEngine(
        docs,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    query_text = "apple banana"  # no match -> BM25 scores zero
    query_vec = [0.0, 0.0, 1.0]  # closest to doc2

    # candidate_k=2: BM25 returns doc0, doc1 (both score 0); dense returns doc2, doc0
    results = engine.search_hybrid_rrf(query_text, query_vec, top_k=3, rrf_k=60, candidate_k=2)
    # doc0: BM25 rank1 + dense rank2 -> 1/61 + 1/62
    # doc1: BM25 rank2 only -> 1/62
    # doc2: dense rank1 only -> 1/61
    # Order: doc0, doc2, doc1
    expected_order = [0, 2, 1]
    assert [idx for idx, _, _ in results] == expected_order
    # Check that doc1 and doc2 have only one contribution each
    doc1_score = None
    doc2_score = None
    for idx, score, _ in results:
        if idx == 1:
            doc1_score = score
        elif idx == 2:
            doc2_score = score
    assert doc1_score == pytest.approx(1.0/62.0)
    assert doc2_score == pytest.approx(1.0/61.0)


def test_rrf_top_k():
    """top_k should limit the number of returned results."""
    docs = ["a", "b", "c"]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    engine = HybridSearchEngine(docs, embeddings)
    results = engine.search_hybrid_rrf("a", [1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 0


def test_rrf_tie_break():
    """Ties in RRF score should be broken by lower document index."""
    docs = ["apple", "apple orange"]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]  # identical dense
    engine = HybridSearchEngine(docs, embeddings)
    query_vec = [1.0, 0.0]
    results = engine.search_hybrid_rrf("apple", query_vec, top_k=2)
    # Both docs have same BM25 and dense ranks -> RRF scores equal.
    # Tie‑breaking by index -> doc0 first, doc1 second.
    assert results[0][0] == 0
    assert results[1][0] == 1


def test_rrf_no_score_normalization():
    """RRF should ignore raw score magnitudes; only ranks matter."""
    docs = ["a", "b"]
    embeddings1 = [[1.0, 0.0], [0.0, 1.0]]
    engine1 = HybridSearchEngine(docs, embeddings1)
    embeddings2 = [[10.0, 0.0], [0.0, 10.0]]  # scaled
    engine2 = HybridSearchEngine(docs, embeddings2)
    q_vec = [1.0, 0.0]
    res1 = engine1.search_hybrid_rrf("a", q_vec, top_k=2)
    res2 = engine2.search_hybrid_rrf("a", q_vec, top_k=2)
    # Ranking should be identical because cosine similarity unchanged
    assert [idx for idx, _, _ in res1] == [idx for idx, _, _ in res2]


def test_rrf_custom_k():
    """RRF with a custom k value should compute scores correctly."""
    docs = ["a", "b"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    engine = HybridSearchEngine(docs, embeddings)
    results = engine.search_hybrid_rrf("a", [0.0, 1.0], top_k=2, rrf_k=10)
    expected_doc0 = 1.0/11.0 + 1.0/12.0  # BM25 rank1 + dense rank2
    expected_doc1 = 1.0/12.0 + 1.0/11.0  # BM25 rank2 + dense rank1
    assert results[0][0] == 0
    assert results[1][0] == 1
    assert abs(results[0][1] - expected_doc0) < 1e-6
    assert abs(results[1][1] - expected_doc1) < 1e-6


def test_rrf_invalid_k():
    """Invalid rrf_k values should raise ValueError."""
    engine = HybridSearchEngine(["doc"], [[1.0]])
    with pytest.raises(ValueError, match="rrf_k must be > 0"):
        engine.search_hybrid_rrf("query", [1.0], rrf_k=0)
    with pytest.raises(ValueError, match="rrf_k must be > 0"):
        engine.search_hybrid_rrf("query", [1.0], rrf_k=-5)
    with pytest.raises(ValueError, match="rrf_k must be a finite number"):
        engine.search_hybrid_rrf("query", [1.0], rrf_k=float("nan"))
    with pytest.raises(ValueError, match="rrf_k must be a finite number"):
        engine.search_hybrid_rrf("query", [1.0], rrf_k=float("inf"))


def test_rrf_invalid_candidate_k():
    """Invalid candidate_k values should raise ValueError."""
    engine = HybridSearchEngine(["doc"], [[1.0]])
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        engine.search_hybrid_rrf("query", [1.0], candidate_k=0)
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        engine.search_hybrid_rrf("query", [1.0], candidate_k=-1)


# ----------------------------------------------------------------------
# Integration & End‑to‑End
# ----------------------------------------------------------------------

def test_hybrid_search_full_payload():
    """End‑to‑end test: verify the final payload contains correct data."""
    docs = [
        "Python autograd engine with backpropagation",
        "MERN stack web application with React and Node",
        "Deep learning transformer models with self attention"
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.0, 0.6]
    ]
    engine = HybridSearchEngine(docs, embeddings)
    query_text = "autograd backpropagation"
    query_vec = [0.9, 0.0, 0.1]

    results = engine.search_hybrid_rrf(query_text, query_vec, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 0
    assert "autograd" in results[0][2]
    assert isinstance(results[0][1], float)
    assert results[0][1] > 0.0


def test_hybrid_search_with_empty_query():
    """Empty query string should be handled (BM25 will get zero scores)."""
    docs = ["test"]
    engine = HybridSearchEngine(docs, [[1.0]])
    results = engine.search_hybrid_rrf("", [1.0], top_k=1)
    assert len(results) == 1
    assert results[0][0] == 0


def test_invalid_top_k():
    """Invalid top_k values should raise ValueError."""
    engine = HybridSearchEngine(["doc"], [[1.0]])
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        engine.search_bm25("query", top_k=0)
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        engine.search_dense([1.0], top_k=-1)
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        engine.search_hybrid_rrf("query", [1.0], top_k=0)


# ----------------------------------------------------------------------
# Additional Validation Tests
# ----------------------------------------------------------------------

def test_document_embedding_count_mismatch():
    """Number of documents must match number of embeddings."""
    with pytest.raises(ValueError, match="does not match"):
        HybridSearchEngine(["doc1", "doc2"], [[1.0]])


def test_embedding_dimension_mismatch():
    """All embeddings must have the same dimension."""
    with pytest.raises(ValueError, match="expected"):
        HybridSearchEngine(["doc1", "doc2"], [[1.0, 2.0], [1.0]])


def test_zero_dimensional_embedding():
    """Embedding dimension must be > 0."""
    with pytest.raises(ValueError, match="dimension must be > 0"):
        HybridSearchEngine(["doc"], [[]])


def test_bm25_nan_params():
    """BM25 parameters must be finite."""
    with pytest.raises(ValueError, match="finite"):
        BM25Okapi([["a"]], k1=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        BM25Okapi([["a"]], b=float("-inf"))


# ----------------------------------------------------------------------
# Run with: pytest test_hybrid_search.py -v
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])