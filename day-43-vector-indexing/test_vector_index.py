"""Tests for Day 43: IVFIndex and HNSWGraph.

Test layout:
    1. Helpers            -- exact top-k and layer-0 reachability utilities.
    2. IVF: correctness   -- clustering, assignments, distances, tie-breaks.
    3. IVF: retrieval     -- nprobe behaviour, recall vs brute force.
    4. IVF: robustness    -- invalid inputs, edge cases.
    5. HNSW: correctness  -- levels, membership, adjacency, degrees.
    6. HNSW: retrieval    -- exact query, top-k ordering, recall, efSearch.
    7. HNSW: robustness   -- invalid inputs, empty graph.
    8. Reproducibility    -- seeded builds yield identical structures.
    9. Validation         -- random_state guard for both classes.
"""

import numpy as np
import pytest

from day43_vector_index import HNSWGraph, IVFIndex


# ===========================================================================
# 1. Helpers
# ===========================================================================


def brute_force_knn(vectors: np.ndarray, query: np.ndarray, top_k: int) -> list[int]:
    """Exact top-k ids by Euclidean distance. Stable tie-break by id."""
    d = np.linalg.norm(vectors - query, axis=1)
    order = np.argsort(d, kind="stable")[:top_k]
    return [int(i) for i in order]


def layer0_reachable(graph: HNSWGraph) -> set[int]:
    """Nodes reachable from the entry point through layer-0 edges."""
    if graph.entry_point == -1:
        return set()
    visited = {graph.entry_point}
    stack = [graph.entry_point]
    while stack:
        node = stack.pop()
        for neighbor in graph.graphs[0].get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


# ===========================================================================
# 2. IVF: correctness
# ===========================================================================


def test_ivf_build_is_deterministic_for_same_seed() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)

    a = IVFIndex(n_centroids=4, nprobe=2, random_state=123)
    b = IVFIndex(n_centroids=4, nprobe=2, random_state=123)
    a.build(X, max_iters=10)
    b.build(X, max_iters=10)

    np.testing.assert_allclose(a.centroids, b.centroids)
    assert a.inverted_lists == b.inverted_lists


def test_ivf_centroid_count_and_shape() -> None:
    X = np.random.default_rng(1).normal(size=(30, 4)).astype(np.float32)
    ivf = IVFIndex(n_centroids=6, nprobe=2, random_state=0)
    ivf.build(X)

    assert ivf.centroids.shape == (6, 4)
    assert len(ivf.inverted_lists) == 6


def test_ivf_separates_two_well_separated_groups() -> None:
    """Two far-apart clusters should be captured by two centroids."""
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [100.0, 100.0],
            [101.0, 100.0],
            [100.0, 101.0],
        ],
        dtype=np.float32,
    )
    ivf = IVFIndex(n_centroids=2, nprobe=2, random_state=42)
    ivf.build(X, max_iters=20)

    owner = next(cid for cid, ids in ivf.inverted_lists.items() if 0 in ids)
    assert {0, 1, 2}.issubset(set(ivf.inverted_lists[owner]))
    assert {3, 4, 5}.issubset(set(ivf.inverted_lists[1 - owner]))


def test_ivf_every_vector_assigned_exactly_once() -> None:
    X = np.random.default_rng(2).normal(size=(50, 3)).astype(np.float32)
    ivf = IVFIndex(n_centroids=5, nprobe=2, random_state=0)
    ivf.build(X)

    all_ids = [i for ids in ivf.inverted_lists.values() for i in ids]
    assert sorted(all_ids) == list(range(len(X)))


def test_ivf_inverted_list_matches_voronoi_partition() -> None:
    """Every vector in list `c` must have centroid `c` as its nearest centroid."""
    X = np.random.default_rng(3).normal(size=(40, 4)).astype(np.float32)
    ivf = IVFIndex(n_centroids=4, nprobe=2, random_state=0)
    ivf.build(X)

    for cid, ids in ivf.inverted_lists.items():
        assert 0 <= cid < len(ivf.centroids)
        for idx in ids:
            d = np.linalg.norm(X[idx] - ivf.centroids, axis=1)
            assert int(np.argmin(d)) == cid


def test_ivf_exact_query_returns_itself_at_distance_zero() -> None:
    X = np.random.default_rng(4).normal(size=(30, 6)).astype(np.float32)
    ivf = IVFIndex(n_centroids=4, nprobe=4, random_state=0)
    ivf.build(X)

    results = ivf.search(X[0], top_k=3)
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(0.0, abs=1e-5)


def test_ivf_hand_computable_distances() -> None:
    """Distances from origin to (3,4) and (6,8) are exactly 5 and 10."""
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    ivf = IVFIndex(n_centroids=3, nprobe=3, random_state=0)
    ivf.build(X)

    results = ivf.search(np.array([0.0, 0.0], dtype=np.float32), top_k=3)
    d_by_id = {i: d for i, d in results}
    assert d_by_id[0] == pytest.approx(0.0, abs=1e-5)
    assert d_by_id[1] == pytest.approx(5.0, abs=1e-5)
    assert d_by_id[2] == pytest.approx(10.0, abs=1e-5)


def test_ivf_tie_breaking_keeps_lower_id_first() -> None:
    """Two equidistant points: stable ordering returns id 0, then id 1."""
    X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    ivf = IVFIndex(n_centroids=1, nprobe=1, random_state=0)
    ivf.build(X)

    results = ivf.search(np.array([0.0, 0.0], dtype=np.float32), top_k=2)
    assert [i for i, _ in results] == [0, 1]


def test_ivf_handles_empty_clusters_gracefully() -> None:
    """Duplicated points can leave some centroids with zero members."""
    X = np.array([[1.0, 1.0]] * 10 + [[5.0, 5.0]] * 2, dtype=np.float32)
    ivf = IVFIndex(n_centroids=8, nprobe=8, random_state=0)
    ivf.build(X, max_iters=5)

    all_ids = [i for ids in ivf.inverted_lists.values() for i in ids]
    assert sorted(all_ids) == list(range(len(X)))

    results = ivf.search(X[0], top_k=3)
    assert len(results) == 3
    assert results[0][0] == 0


def test_ivf_top_k_is_sorted_ascending() -> None:
    X = np.random.default_rng(5).normal(size=(30, 4)).astype(np.float32)
    ivf = IVFIndex(n_centroids=4, nprobe=4, random_state=0)
    ivf.build(X)

    results = ivf.search(X[1], top_k=5)
    assert len(results) <= 5
    assert [d for _, d in results] == sorted(d for _, d in results)


# ===========================================================================
# 3. IVF: retrieval quality
# ===========================================================================


def test_ivf_full_nprobe_recovers_bruteforce_recall() -> None:
    """When nprobe == C, every list is scanned, so recall must be exactly 1.0."""
    rng = np.random.default_rng(300)
    X = rng.normal(size=(200, 8)).astype(np.float32)

    ivf = IVFIndex(n_centroids=8, nprobe=8, random_state=42)
    ivf.build(X)

    queries = rng.normal(size=(20, 8)).astype(np.float32)
    recalls: list[float] = []
    for q in queries:
        exact_ids = set(brute_force_knn(X, q, top_k=5))
        approx_ids = {i for i, _ in ivf.search(q, top_k=5)}
        recalls.append(len(exact_ids & approx_ids) / 5)

    mean_recall = float(np.mean(recalls))
    assert mean_recall == pytest.approx(1.0, abs=1e-12)


def test_ivf_recall_does_not_decrease_as_nprobe_grows() -> None:
    """Small nprobe is approximate; full nprobe is exact."""
    rng = np.random.default_rng(301)
    X = rng.normal(size=(300, 8)).astype(np.float32)
    queries = rng.normal(size=(25, 8)).astype(np.float32)

    def mean_recall(nprobe: int) -> float:
        ivf = IVFIndex(n_centroids=16, nprobe=nprobe, random_state=42)
        ivf.build(X)
        vals = []
        for q in queries:
            exact = set(brute_force_knn(X, q, top_k=5))
            approx = {i for i, _ in ivf.search(q, top_k=5)}
            vals.append(len(exact & approx) / 5)
        return float(np.mean(vals))

    r_small = mean_recall(1)
    r_full = mean_recall(16)

    assert r_full == pytest.approx(1.0, abs=1e-12)
    assert r_small <= r_full + 1e-12


def test_ivf_nprobe_1_returns_subset_of_nprobe_all() -> None:
    """Larger nprobe touches at least as many candidates as smaller nprobe."""
    X = np.array(
        [[0.0, 0.0], [10.0, 10.0], [0.1, 0.1], [10.1, 10.1]],
        dtype=np.float32,
    )
    ivf1 = IVFIndex(n_centroids=2, nprobe=1, random_state=0)
    ivf2 = IVFIndex(n_centroids=2, nprobe=2, random_state=0)
    ivf1.build(X)
    ivf2.build(X)

    q = np.array([0.2, 0.2], dtype=np.float32)
    r1 = ivf1.search(q, top_k=4)
    r2 = ivf2.search(q, top_k=4)

    assert len(r1) <= len(r2) <= len(X)
    assert all(0 <= i < len(X) for i, _ in r1)
    assert all(0 <= i < len(X) for i, _ in r2)


# ===========================================================================
# 4. IVF: robustness
# ===========================================================================


def test_ivf_search_before_build_raises() -> None:
    ivf = IVFIndex()
    with pytest.raises(RuntimeError):
        ivf.search(np.zeros(2, dtype=np.float32))


def test_ivf_rejects_complex_input() -> None:
    ivf = IVFIndex(n_centroids=2)
    X = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
    with pytest.raises(TypeError):
        ivf.build(X)

    ivf.build(np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32))
    with pytest.raises(TypeError):
        ivf.search(np.array([1 + 0j, 0 + 0j], dtype=np.complex64))


def test_ivf_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError):
        IVFIndex(n_centroids=0)
    with pytest.raises(TypeError):
        IVFIndex(n_centroids=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IVFIndex(n_centroids=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        IVFIndex(n_centroids=2, nprobe=0)

    ivf = IVFIndex(n_centroids=2)

    with pytest.raises(TypeError):
        ivf.build([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ivf.build(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError):
        ivf.build(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        ivf.build(np.array([[1.0, np.nan]], dtype=np.float32))

    X = np.random.default_rng(6).normal(size=(10, 3)).astype(np.float32)
    ivf.build(X)

    with pytest.raises(ValueError):
        ivf.search(np.zeros(2, dtype=np.float32))
    with pytest.raises(TypeError):
        ivf.search([0.0, 0.0, 0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ivf.search(np.array([0.0, np.inf, 0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        ivf.search(np.zeros(3, dtype=np.float32), top_k=0)


def test_ivf_clamps_n_centroids_when_greater_than_n() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ivf = IVFIndex(n_centroids=10, nprobe=3, random_state=0)
    ivf.build(X)

    assert len(ivf.centroids) == 3
    all_ids = [i for ids in ivf.inverted_lists.values() for i in ids]
    assert sorted(all_ids) == [0, 1, 2]

    results = ivf.search(X[1], top_k=1)
    assert results[0][0] == 1
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


def test_ivf_single_vector_dataset() -> None:
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    ivf = IVFIndex(n_centroids=4, nprobe=4, random_state=0)
    ivf.build(X)

    assert ivf.centroids.shape == (1, 3)
    results = ivf.search(X[0], top_k=1)
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# 5. HNSW: correctness
# ===========================================================================


def test_hnsw_distance_matches_3_4_5_triangle() -> None:
    g = HNSWGraph(dim=2)
    assert g._distance(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ) == pytest.approx(5.0)


def test_hnsw_levels_are_deterministic_for_same_seed() -> None:
    X = np.random.default_rng(7).normal(size=(20, 3)).astype(np.float32)

    g1 = HNSWGraph(dim=3, random_state=7)
    g2 = HNSWGraph(dim=3, random_state=7)
    for v in X:
        g1.insert(v)
        g2.insert(v)

    assert g1.node_levels == g2.node_levels


def test_hnsw_two_node_graph_search_is_exact() -> None:
    """Hand-computable distances: query at (0.1, 0)."""
    g = HNSWGraph(dim=2, m=4, ef_construction=4, random_state=0)
    g.insert(np.array([0.0, 0.0], dtype=np.float32))
    g.insert(np.array([1.0, 0.0], dtype=np.float32))

    assert 0 in g.graphs[0] and 1 in g.graphs[0]

    results = g.search(np.array([0.1, 0.0], dtype=np.float32), top_k=2, ef_search=4)
    assert results[0][0] == 0
    assert results[0][1] == pytest.approx(0.1, abs=1e-5)
    assert results[1][0] == 1
    assert results[1][1] == pytest.approx(0.9, abs=1e-5)


def test_hnsw_first_insert_becomes_entry_point() -> None:
    g = HNSWGraph(dim=2, random_state=0)
    node_id = g.insert(np.array([1.0, 2.0], dtype=np.float32))

    assert node_id == 0
    assert g.entry_point == 0
    assert g.max_level == g.node_levels[0]
    assert 0 in g.graphs[0]


def test_hnsw_multiple_inserts_grow_the_graph() -> None:
    X = np.random.default_rng(8).normal(size=(25, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=4, ef_construction=8, random_state=1)
    for v in X:
        g.insert(v)

    assert len(g.vectors) == 25
    assert len(g.node_levels) == 25
    assert g.entry_point != -1
    assert g.max_level >= 0


def test_hnsw_layer0_contains_every_node() -> None:
    X = np.random.default_rng(9).normal(size=(30, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=6, ef_construction=12, random_state=2)
    for v in X:
        g.insert(v)

    assert all(i in g.graphs[0] for i in range(len(X)))


def test_hnsw_layer_membership_respects_node_level() -> None:
    """A node with level L lives in layers 0..L, and nowhere above."""
    X = np.random.default_rng(10).normal(size=(30, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=6, ef_construction=12, random_state=3)
    for v in X:
        g.insert(v)

    for node_id, level in enumerate(g.node_levels):
        for layer in range(level + 1):
            assert node_id in g.graphs[layer]
        for layer in range(level + 1, len(g.graphs)):
            assert node_id not in g.graphs[layer]


def test_hnsw_adjacency_has_no_self_loops_or_duplicates() -> None:
    X = np.random.default_rng(11).normal(size=(30, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=6, ef_construction=12, random_state=4)
    for v in X:
        g.insert(v)

    for layer in g.graphs:
        for node, neighbors in layer.items():
            assert 0 <= node < len(g.vectors)
            assert len(neighbors) == len(set(neighbors))
            for n in neighbors:
                assert 0 <= n < len(g.vectors)
                assert n != node


def test_hnsw_edges_are_reciprocal() -> None:
    X = np.random.default_rng(12).normal(size=(40, 3)).astype(np.float32)
    g = HNSWGraph(dim=3, m=4, ef_construction=8, random_state=5)
    for v in X:
        g.insert(v)

    for layer in g.graphs:
        for node, neighbors in layer.items():
            for n in neighbors:
                assert node in layer[n]


def test_hnsw_degree_never_exceeds_m() -> None:
    X = np.random.default_rng(13).normal(size=(40, 5)).astype(np.float32)
    g = HNSWGraph(dim=5, m=5, ef_construction=10, random_state=6)
    for v in X:
        g.insert(v)

    for layer in g.graphs:
        for neighbors in layer.values():
            assert len(neighbors) <= g.m


def test_hnsw_layer0_is_connected() -> None:
    """Every node is reachable from the entry point at layer 0."""
    X = np.random.default_rng(123).normal(size=(100, 8)).astype(np.float32)
    g = HNSWGraph(dim=8, m=8, ef_construction=32, random_state=42)
    for v in X:
        g.insert(v)

    assert layer0_reachable(g) == set(range(len(X)))


def test_hnsw_layer0_is_connected_with_heuristic() -> None:
    """Connectivity must also hold when using diversity-aware selection."""
    X = np.random.default_rng(124).normal(size=(80, 6)).astype(np.float32)
    g = HNSWGraph(dim=6, m=6, ef_construction=24, random_state=42, use_heuristic=True)
    for v in X:
        g.insert(v)

    assert layer0_reachable(g) == set(range(len(X)))


# ===========================================================================
# 6. HNSW: retrieval quality
# ===========================================================================


def test_hnsw_exact_query_returns_itself() -> None:
    X = np.random.default_rng(14).normal(size=(20, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=20, ef_construction=32, random_state=123)
    for v in X:
        g.insert(v)

    results = g.search(X[5], top_k=3, ef_search=32)
    assert results[0][0] == 5
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


def test_hnsw_top_k_is_sorted_ascending() -> None:
    X = np.random.default_rng(15).normal(size=(20, 4)).astype(np.float32)
    g = HNSWGraph(dim=4, m=20, ef_construction=32, random_state=123)
    for v in X:
        g.insert(v)

    results = g.search(X[2], top_k=4, ef_search=32)
    assert len(results) <= 4
    assert [d for _, d in results] == sorted(d for _, d in results)


def test_hnsw_mean_recall_at_5_beats_threshold() -> None:
    """Compare HNSW's top-5 to exact top-5 across many queries."""
    rng = np.random.default_rng(200)
    X = rng.normal(size=(200, 8)).astype(np.float32)

    g = HNSWGraph(dim=8, m=16, ef_construction=32, random_state=42)
    for v in X:
        g.insert(v)

    queries = rng.normal(size=(20, 8)).astype(np.float32)
    recalls: list[float] = []
    for q in queries:
        exact_ids = set(brute_force_knn(X, q, top_k=5))
        approx_ids = {i for i, _ in g.search(q, top_k=5, ef_search=32)}
        recalls.append(len(exact_ids & approx_ids) / 5)

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.7, f"mean Recall@5 too low: {mean_recall:.3f}"


def test_hnsw_larger_ef_search_does_not_hurt_mean_recall() -> None:
    """Higher ef_search explores more, so mean recall should not drop."""
    rng = np.random.default_rng(201)
    X = rng.normal(size=(150, 8)).astype(np.float32)
    g = HNSWGraph(dim=8, m=8, ef_construction=32, random_state=42)
    for v in X:
        g.insert(v)

    queries = rng.normal(size=(15, 8)).astype(np.float32)

    def mean_recall(ef: int) -> float:
        vals = []
        for q in queries:
            exact = set(brute_force_knn(X, q, top_k=5))
            approx = {i for i, _ in g.search(q, top_k=5, ef_search=ef)}
            vals.append(len(exact & approx) / 5)
        return float(np.mean(vals))

    r_small = mean_recall(4)
    r_large = mean_recall(32)
    assert r_large >= r_small - 0.05


# ===========================================================================
# 7. HNSW: robustness
# ===========================================================================


def test_hnsw_empty_graph_search_returns_empty_list() -> None:
    g = HNSWGraph(dim=2)
    assert g.search(np.zeros(2, dtype=np.float32), top_k=3) == []


def test_hnsw_rejects_complex_input() -> None:
    g = HNSWGraph(dim=2)
    with pytest.raises(TypeError):
        g.insert(np.array([1 + 0j, 0 + 0j], dtype=np.complex64))

    g.insert(np.array([0.0, 0.0], dtype=np.float32))
    with pytest.raises(TypeError):
        g.search(np.array([1 + 0j, 0 + 0j], dtype=np.complex64))


def test_hnsw_rejects_invalid_parameters_and_dimensions() -> None:
    with pytest.raises(ValueError):
        HNSWGraph(dim=0)
    with pytest.raises(ValueError):
        HNSWGraph(dim=2, m=0)
    with pytest.raises(ValueError):
        HNSWGraph(dim=2, ef_construction=0)
    with pytest.raises(ValueError):
        HNSWGraph(dim=2, ml=0.0)
    with pytest.raises(TypeError):
        HNSWGraph(dim=2, use_heuristic=1)  # type: ignore[arg-type]

    g = HNSWGraph(dim=3)
    with pytest.raises(ValueError):
        g.insert(np.zeros(2, dtype=np.float32))
    with pytest.raises(TypeError):
        g.insert([0.0, 0.0, 0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        g.insert(np.array([0.0, np.nan, 0.0], dtype=np.float32))


# ===========================================================================
# 8. Reproducibility
# ===========================================================================


def test_hnsw_build_is_reproducible_with_same_seed() -> None:
    X = np.random.default_rng(16).normal(size=(25, 4)).astype(np.float32)

    g1 = HNSWGraph(dim=4, m=4, ef_construction=8, random_state=42)
    g2 = HNSWGraph(dim=4, m=4, ef_construction=8, random_state=42)
    for v in X:
        g1.insert(v)
        g2.insert(v)

    assert g1.node_levels == g2.node_levels
    assert g1.graphs == g2.graphs
    assert g1.entry_point == g2.entry_point
    assert g1.max_level == g2.max_level


def test_hnsw_heuristic_build_is_reproducible_with_same_seed() -> None:
    X = np.random.default_rng(17).normal(size=(25, 4)).astype(np.float32)

    g1 = HNSWGraph(dim=4, m=4, ef_construction=8, random_state=42, use_heuristic=True)
    g2 = HNSWGraph(dim=4, m=4, ef_construction=8, random_state=42, use_heuristic=True)
    for v in X:
        g1.insert(v)
        g2.insert(v)

    assert g1.graphs == g2.graphs
    assert g1.node_levels == g2.node_levels


# ===========================================================================
# 9. random_state validation
# ===========================================================================


def test_both_classes_reject_invalid_random_state() -> None:
    # Invalid values
    with pytest.raises(TypeError):
        IVFIndex(n_centroids=2, random_state=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IVFIndex(n_centroids=2, random_state=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HNSWGraph(dim=2, random_state=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HNSWGraph(dim=2, random_state="seed")  # type: ignore[arg-type]

    # Valid values: int and None
    IVFIndex(n_centroids=2, random_state=None)
    IVFIndex(n_centroids=2, random_state=0)
    HNSWGraph(dim=2, random_state=None)
    HNSWGraph(dim=2, random_state=7)
