"""Day 43: First-principles IVF and HNSW vector indexes (NumPy only).

This module implements two Approximate Nearest Neighbor (ANN) data structures
from scratch, using nothing but NumPy:

    1. IVFIndex  -- partitions vectors into Voronoi cells with K-Means and
                    stores one "inverted list" of vector ids per centroid.
    2. HNSWGraph -- builds a multi-layer navigable small-world graph and
                    searches it by greedy descent + bounded beam search.

Why bother? Brute-force nearest-neighbor search is O(N * D) per query. At
millions of vectors this is too slow for interactive use. ANN trades a little
recall for a lot of speed. IVF and HNSW are the two most common building
blocks in modern vector databases (FAISS, Milvus, Qdrant, pgvector, ...).

Neither class here is production-equivalent. Real systems additionally handle
persistence, concurrent read/write, updates/deletes, and compressed storage
(PQ/SQ8). We implement the algorithmic core only, on purpose.
"""

from __future__ import annotations

import bisect
import heapq
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Small floor used when clamping probabilities away from 0 before log().
_EPS = 1e-12


# ===========================================================================
# Input validation helpers
# ===========================================================================
# Every public method validates its inputs up front. This keeps the search
# and build loops free of defensive checks and gives callers clear errors.


def _validate_positive_int(value: object, name: str) -> None:
    """Require a strictly positive integer.

    `bool` is rejected even though it subclasses `int`, because passing
    `True` where a count is expected is almost always a bug.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_random_state(value: object) -> None:
    """Require `int | None` for the `random_state` argument."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"random_state must be int | None, got {type(value).__name__}")


def _validate_real_2d_array(arr: object, name: str = "array") -> None:
    """Require a 2D, finite, real-valued NumPy array (the vector corpus)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (N, D), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one feature")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    if np.iscomplexobj(arr):
        # Complex values would be silently truncated to their real part by
        # `astype(np.float32)`, which is data corruption. Reject explicitly.
        raise TypeError(
            f"{name} must contain real values, got complex dtype {arr.dtype}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def _validate_real_vector(vec: object, dim: int, name: str = "vector") -> None:
    """Require a 1D, finite, real-valued NumPy vector of length `dim`."""
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(vec).__name__}")
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D (D,), got shape {vec.shape}")
    if vec.shape[0] != dim:
        raise ValueError(f"{name} dim mismatch: expected {dim}, got {vec.shape[0]}")
    if not np.issubdtype(vec.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {vec.dtype}")
    if np.iscomplexobj(vec):
        raise TypeError(
            f"{name} must contain real values, got complex dtype {vec.dtype}"
        )
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN or infinite values")


# ===========================================================================
# IVFIndex
# ===========================================================================
#
# Big picture
# -----------
#
#               ┌─────────────────────────────┐
#   vectors ───▶│  K-Means → C centroids       │
#               │  assign each vector to its   │
#               │  nearest centroid           │
#               └──────────────┬──────────────┘
#                              │
#                              ▼
#               inverted lists: centroid → [vector ids]
#
#   query ──▶ find nprobe nearest centroids
#             scan only those inverted lists
#             exact Euclidean distance on candidates
#             return top-k
#
# Under balanced lists, search touches roughly (nprobe / C) * N vectors
# instead of all N. That's the speed/recall knob: bigger nprobe → more
# candidates → higher recall but slower queries.
# ===========================================================================


class IVFIndex:
    """Inverted File Index with K-Means Voronoi partitioning.

    Each vector is assigned to exactly one cluster. Search probes the
    `nprobe` nearest centroids and scans only the vectors in those
    inverted lists -- so search is approximate, not a full-corpus scan.

    Args:
        n_centroids: Requested K-Means centroid count. If larger than the
            number of input vectors, it is clamped to that number.
        nprobe: Nearest centroids probed at search time. If larger than the
            effective centroid count, it is clamped to that count.
        random_state: Seed for K-Means initialization (default 42).

    Notes:
        - Ties in argmin resolve to the lowest centroid index.
        - Empty clusters keep their previous centroid (a documented policy;
          we do not reseed them).
        - Distances are computed in float64 internally to reduce overflow
          risk when inputs sit near the float32 limit. Storage stays float32.
    """

    def __init__(
        self,
        n_centroids: int = 4,
        nprobe: int = 2,
        random_state: int | None = 42,
    ) -> None:
        _validate_positive_int(n_centroids, "n_centroids")
        _validate_positive_int(nprobe, "nprobe")
        _validate_random_state(random_state)

        self.requested_n_centroids: int = int(n_centroids)
        self.n_centroids: int = int(n_centroids)  # set to min(requested, N) at build
        self.nprobe: int = int(nprobe)
        self.random_state: int | None = random_state

        self.centroids: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.inverted_lists: dict[int, list[int]] = {}
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.is_built: bool = False
        self.dim: int = 0

    # ---------------------------------------------------------------------
    # Internal math
    # ---------------------------------------------------------------------

    @staticmethod
    def _squared_euclidean(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Return the (N, C) matrix of squared Euclidean distances.

        Uses the identity ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x·c, which is
        a single matrix multiply instead of an (N, C, D) broadcast. For K-Means
        we can compare squared distances directly -- square root is monotonic,
        so argmin is unchanged.
        """
        x_norm = np.einsum("ij,ij->i", X, X)[:, None]  # (N, 1)
        c_norm = np.einsum("ij,ij->i", C, C)[None, :]  # (1, C)
        d2 = x_norm + c_norm - 2.0 * (X @ C.T)
        # Clamp tiny negative values from floating-point cancellation.
        np.maximum(d2, 0.0, out=d2)
        return d2

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Nearest centroid id for each row of X. Ties → lowest index."""
        return np.argmin(self._squared_euclidean(X, centroids), axis=1)

    # ---------------------------------------------------------------------
    # Build
    # ---------------------------------------------------------------------

    def build(self, vectors: np.ndarray, max_iters: int = 20) -> None:
        """Run K-Means and populate the inverted lists.

        Args:
            vectors: Array of shape (N, D), the corpus to index.
            max_iters: Maximum K-Means iterations (convergence often sooner).
        """
        _validate_real_2d_array(vectors, "vectors")
        _validate_positive_int(max_iters, "max_iters")

        X = vectors.astype(np.float32, copy=True)
        N, D = X.shape
        self.dim = D
        self.vectors = X

        # Clamp K when the user asks for more centroids than we have vectors.
        k = min(self.requested_n_centroids, N)
        self.n_centroids = k

        # ---- Initialize centroids ----
        rng = np.random.default_rng(self.random_state)
        if k == N:
            # Every vector becomes its own centroid.
            centroids = X.copy()
        else:
            # Random distinct rows as seeds (no k-means++ here -- see the
            # module docstring for scope limits).
            idx = rng.choice(N, size=k, replace=False)
            centroids = X[idx].copy()

        # ---- Lloyd's iterations ----
        for _ in range(max_iters):
            assignments = self._assign(X, centroids)

            # Accumulate sums per cluster. `np.add.at` handles duplicate
            # cluster ids correctly (a plain indexing assignment would not).
            sums = np.zeros_like(centroids)
            np.add.at(sums, assignments, X)
            counts = np.bincount(assignments, minlength=k).astype(np.int64)

            new_centroids = centroids.copy()
            non_empty = counts > 0
            # Empty cluster policy: keep the previous centroid where it was.
            new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

            if np.allclose(centroids, new_centroids, rtol=0.0, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        # ---- Final assignment against the final centroids ----
        # The assignments used inside the loop referred to the *previous*
        # centroids. Recompute them once so inverted lists reflect the
        # actual Voronoi partition: every vector appears in exactly one list.
        final_assignments = self._assign(X, centroids)
        self.centroids = centroids.astype(np.float32, copy=True)
        self.inverted_lists = {c: [] for c in range(k)}
        for idx, c_id in enumerate(final_assignments):
            self.inverted_lists[int(c_id)].append(int(idx))

        self.is_built = True
        logger.debug("IVF built: N=%d D=%d effective_centroids=%d", N, D, k)

    # ---------------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------------

    def search(self, query: np.ndarray, top_k: int = 3) -> list[tuple[int, float]]:
        """Find the top_k nearest vectors to `query`.

        Only vectors in the `nprobe` nearest inverted lists are examined.

        Returns:
            Sorted list of `(vector_id, euclidean_distance)` (ascending),
            length <= top_k.
        """
        if not self.is_built:
            raise RuntimeError("IVFIndex must be built before search")
        _validate_real_vector(query, self.dim, "query")
        _validate_positive_int(top_k, "top_k")

        N = self.vectors.shape[0]
        if top_k > N:
            raise ValueError(f"top_k={top_k} exceeds indexed vectors N={N}")

        # float64 for distance arithmetic -- guards against overflow when
        # the input magnitudes are close to float32's limit.
        q64 = query.astype(np.float64, copy=False)
        C64 = self.centroids.astype(np.float64, copy=False)

        # ---- Pick nprobe nearest centroids ----
        diff = C64 - q64
        centroid_d2 = np.einsum("ij,ij->i", diff, diff)

        nprobe_eff = min(self.nprobe, C64.shape[0])
        if nprobe_eff == C64.shape[0]:
            nearest = np.arange(C64.shape[0])
        else:
            # argpartition is O(C); a full sort would be O(C log C).
            # We still stable-sort the selected subset so the probe order
            # is deterministic (matters when distances tie).
            nearest = np.argpartition(centroid_d2, kth=nprobe_eff - 1)[:nprobe_eff]
        nearest = nearest[np.argsort(centroid_d2[nearest], kind="stable")]

        # ---- Gather candidate ids from the selected lists ----
        candidate_ids: list[int] = []
        for c in nearest:
            candidate_ids.extend(self.inverted_lists.get(int(c), []))

        if not candidate_ids:
            return []

        # ---- Exact Euclidean distance on the candidate subset ----
        cand_idx = np.asarray(candidate_ids, dtype=np.int64)
        V64 = self.vectors[cand_idx].astype(np.float64, copy=False)
        diff = V64 - q64
        dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))

        order = np.argsort(dists, kind="stable")[:top_k]
        return [(int(cand_idx[i]), float(dists[i])) for i in order]


# ===========================================================================
# HNSWGraph
# ===========================================================================
#
# Big picture
# -----------
#
#   Layer 2:            A
#                      / \
#   Layer 1:      A --- B --- C
#                / \    |    / \
#   Layer 0:  A--B--C--D--E--F--G--H    ← every node lives here
#
# Each node gets a random top level. Every node exists in all layers
# 0..level. Search starts at the entry point (highest-level node), does a
# greedy descent through the upper layers, then a bounded beam search
# (`ef_search`) at layer 0.
#
# Level distribution: floor(-ln(U) * ml) with U in (0, 1). This yields a
# geometrically shrinking number of nodes per layer, so upper layers act as
# cheap "skip list" shortcuts over the dense local graph below.
# ===========================================================================


class HNSWGraph:
    """Educational multi-layer Navigable Small World graph.

    Neighbor selection has two policies:

        * `use_heuristic=False` (default): keep the `m` closest candidates.
          Simple and deterministic, but can produce geometrically redundant
          neighbor sets that hurt navigability.

        * `use_heuristic=True`: apply the HNSW paper's diversity heuristic --
          accept a candidate only if it is closer to the node than to any
          already-selected neighbor. Tends to improve recall at the same `m`,
          at the cost of extra distance computations during pruning.

    Args:
        dim: Vector dimensionality.
        m: Maximum degree per node per layer.
        ef_construction: Candidate list size during insertion.
        ml: Level multiplier (default ``1 / log(16)``).
        random_state: Seed for level generation (default 42).
        use_heuristic: Enable diversity-aware neighbor selection.
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 32,
        ml: float = 1.0 / math.log(16),
        random_state: int | None = 42,
        use_heuristic: bool = False,
    ) -> None:
        _validate_positive_int(dim, "dim")
        _validate_positive_int(m, "m")
        _validate_positive_int(ef_construction, "ef_construction")
        _validate_random_state(random_state)

        if (
            not isinstance(ml, (int, float))
            or not math.isfinite(float(ml))
            or float(ml) <= 0
        ):
            raise ValueError(f"ml must be a positive finite float, got {ml}")
        if not isinstance(use_heuristic, bool):
            raise TypeError(
                f"use_heuristic must be bool, got {type(use_heuristic).__name__}"
            )

        self.dim: int = int(dim)
        self.m: int = int(m)
        self.ef_construction: int = int(ef_construction)
        self.ml: float = float(ml)
        self.random_state: int | None = random_state
        self.use_heuristic: bool = use_heuristic
        self._rng: np.random.Generator = np.random.default_rng(random_state)

        self.vectors: list[np.ndarray] = []
        # graphs[level] is a dict: node_id -> list of neighbor ids.
        self.graphs: list[dict[int, list[int]]] = []
        self.node_levels: list[int] = []
        self.entry_point: int = -1
        self.max_level: int = -1

    # ---------------------------------------------------------------------
    # Primitives
    # ---------------------------------------------------------------------

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Euclidean distance. Internally float64 for numeric headroom."""
        a = v1.astype(np.float64, copy=False)
        b = v2.astype(np.float64, copy=False)
        return float(np.linalg.norm(a - b))

    def _get_random_level(self) -> int:
        """Sample a level from floor(-ln(U) * ml), U in (0, 1).

        The clamp on U keeps log() away from 0 and 1, so the result is always
        a finite non-negative integer.
        """
        u = float(self._rng.random())
        u = min(max(u, _EPS), 1.0 - _EPS)
        return math.floor(-math.log(u) * self.ml)

    # ---------------------------------------------------------------------
    # Neighbor selection and edge maintenance
    # ---------------------------------------------------------------------

    def _select_neighbors(
        self, node_vec: np.ndarray, candidate_ids: list[int]
    ) -> list[int]:
        """Choose up to `m` neighbors from `candidate_ids`.

        Deterministic: ties are broken by ascending node id.
        """
        scored = sorted(
            ((self._distance(node_vec, self.vectors[c]), c) for c in candidate_ids),
            key=lambda x: (x[0], x[1]),
        )
        if not self.use_heuristic:
            return [c for _, c in scored[: self.m]]

        # Diversity heuristic from the HNSW paper: accept a candidate only
        # if it is closer to the current node than to any already-selected
        # neighbor. This avoids filling the neighbor list with a tight cluster
        # of nearly-identical directions.
        selected: list[int] = []
        for d_q, c in scored:
            if len(selected) >= self.m:
                break
            if all(
                self._distance(self.vectors[c], self.vectors[s]) >= d_q
                for s in selected
            ):
                selected.append(c)
        return selected

    def _add_edge(self, a: int, b: int, level: int) -> None:
        """Add a symmetric edge (a, b) at `level`, then enforce degree limits."""
        if a == b:
            return

        self.graphs[level].setdefault(a, [])
        self.graphs[level].setdefault(b, [])

        if b not in self.graphs[level][a]:
            self.graphs[level][a].append(b)
        if a not in self.graphs[level][b]:
            self.graphs[level][b].append(a)

        # Pruning one endpoint may remove the reverse edge; prune both.
        self._maybe_prune(a, level)
        self._maybe_prune(b, level)

    def _maybe_prune(self, node: int, level: int) -> None:
        """If `node` exceeds degree m at this layer, keep only its best m."""
        current = self.graphs[level][node]
        if len(current) <= self.m:
            return

        keep = self._select_neighbors(self.vectors[node], list(current))
        keep_set = set(keep)
        removed = [n for n in current if n not in keep_set]

        self.graphs[level][node] = keep
        # Remove the reverse edge for every pruned neighbor so the graph
        # stays symmetric.
        for r in removed:
            reverse = self.graphs[level].get(r, [])
            if node in reverse:
                reverse.remove(node)

    # ---------------------------------------------------------------------
    # Search primitives
    # ---------------------------------------------------------------------

    def _greedy_search_layer(
        self, query: np.ndarray, entry_node: int, level: int
    ) -> tuple[int, float]:
        """Greedy descent to a local minimum within a single layer.

        Repeatedly move to the closest neighbor until no neighbor improves.
        """
        curr_node = entry_node
        curr_dist = self._distance(query, self.vectors[curr_node])

        improved = True
        while improved:
            improved = False
            for n in self.graphs[level].get(curr_node, []):
                d = self._distance(query, self.vectors[n])
                if d < curr_dist:
                    curr_node, curr_dist = n, d
                    improved = True
        return curr_node, curr_dist

    def _search_layer(
        self,
        query: np.ndarray,
        entry_node: int,
        ef: int,
        level: int,
    ) -> list[tuple[float, int]]:
        """Bounded best-first search in one layer.

        Maintains two structures:
            * `candidates` -- a min-heap of nodes still worth expanding.
            * `results`    -- up to `ef` best-so-far, kept sorted by distance.

        Stops as soon as the closest unexplored candidate is farther than
        the worst result we are currently keeping.

        Returns a sorted list of `(distance, node_id)` of length <= ef.
        """
        if ef <= 0:
            raise ValueError("ef must be positive")
        if level >= len(self.graphs):
            raise ValueError(f"level {level} does not exist")

        visited = {entry_node}
        entry_dist = self._distance(query, self.vectors[entry_node])

        candidates: list[tuple[float, int]] = [(entry_dist, entry_node)]
        results: list[tuple[float, int]] = [(entry_dist, entry_node)]

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)

            # Nearest unexplored candidate is worse than our worst kept
            # result → nothing left to gain. Stop.
            if c_dist > results[-1][0]:
                break

            for n in self.graphs[level].get(c_node, []):
                if n in visited:
                    continue
                visited.add(n)

                d = self._distance(query, self.vectors[n])
                if len(results) < ef or d < results[-1][0]:
                    heapq.heappush(candidates, (d, n))
                    bisect.insort(results, (d, n))
                    if len(results) > ef:
                        results.pop()  # drop the current worst

        return results

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def insert(self, vector: np.ndarray) -> int:
        """Insert one vector into the graph. Returns its node id."""
        _validate_real_vector(vector, self.dim, "vector")
        v = vector.astype(np.float32, copy=True)

        node_id = len(self.vectors)
        self.vectors.append(v)

        # Random top level for this node.
        node_level = self._get_random_level()
        self.node_levels.append(node_level)

        # Ensure the graph has dictionaries for every layer up to node_level.
        while len(self.graphs) <= node_level:
            self.graphs.append({})

        # Invariant: every node lives in layers 0..node_level.
        for level in range(node_level + 1):
            self.graphs[level].setdefault(node_id, [])

        # First node: becomes the entry point.
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = node_level
            return node_id

        curr_node = self.entry_point

        # Phase 1: greedy descent through layers above our node's level.
        # We only need the closest node id from each layer.
        for level in range(self.max_level, node_level, -1):
            curr_node, _ = self._greedy_search_layer(v, curr_node, level)

        # Phase 2: at each layer from min(max_level, node_level) down to 0,
        # find a good local neighborhood and link the new node into it.
        for level in range(min(self.max_level, node_level), -1, -1):
            candidates = self._search_layer(v, curr_node, self.ef_construction, level)
            candidate_ids = [n for _, n in candidates if n != node_id]
            selected = self._select_neighbors(v, candidate_ids)

            for neighbor in selected:
                self._add_edge(node_id, neighbor, level)

            if candidates:
                # Best candidate becomes the entry point for the next layer down.
                best = min(candidates, key=lambda x: (x[0], x[1]))
                curr_node = best[1]

        # If this node is taller than any before it, it becomes the new entry.
        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

        return node_id

    def search(
        self,
        query: np.ndarray,
        top_k: int = 3,
        ef_search: int = 16,
    ) -> list[tuple[int, float]]:
        """Find the top_k nearest nodes to `query`.

        Returns:
            Sorted list of `(node_id, euclidean_distance)` (ascending),
            length <= top_k.
        """
        _validate_real_vector(query, self.dim, "query")
        _validate_positive_int(top_k, "top_k")
        _validate_positive_int(ef_search, "ef_search")

        if self.entry_point == -1:
            return []

        if top_k > len(self.vectors):
            raise ValueError(
                f"top_k={top_k} exceeds indexed vectors N={len(self.vectors)}"
            )

        q = query.astype(np.float32, copy=False)
        curr_node = self.entry_point

        # Top-down greedy descent. Only the node id is carried forward; the
        # distance is recomputed inside `_search_layer` for the layer-0 beam.
        for level in range(self.max_level, 0, -1):
            curr_node, _ = self._greedy_search_layer(q, curr_node, level)

        # Layer-0 bounded beam search. ef must be at least top_k so we can
        # return top_k results.
        ef = max(ef_search, top_k)
        results = self._search_layer(q, curr_node, ef, level=0)
        results.sort(key=lambda x: (x[0], x[1]))
        return [(int(node), float(dist)) for dist, node in results[:top_k]]
