"""Day 60 — FlashDecoding reference implementation (NumPy).

WHAT THIS SOLVES
================
Single-query-token ("decode phase") attention over a KV cache. During
autoregressive generation Q has length 1, so FlashAttention's sequence-
tiling parallelism has nothing to tile. FlashDecoding splits the *key/
value* sequence into C contiguous chunks, computes a local online-softmax
summary per chunk, and merges the summaries via a global log-sum-exp
reduction. The result is mathematically identical to standard attention;
only the reduction structure changes, and that structure is what enables
sequence-parallel execution on real hardware.

COMPLEXITY
==========
Let N = KV length, C = number of chunks, d = Q/K feature dim,
d_v = V feature dim.

  * Total arithmetic: O(N*d + N*d_v). Chunking does NOT reduce total
    work. Every KV row is still dotted with Q once and contributes to
    the weighted value sum once.
  * Idealized per-chunk critical path: O((N/C)*(d + d_v)). This is only
    a parallel-time estimate with actual concurrent execution; a
    sequential NumPy loop runs all chunks one after the other, so its
    wall-clock time is O(N*(d + d_v)).
  * Merge: O(C*d_v + C).

Assumptions for the above: (a) Q @ K.T uses a BLAS-backed matmul so each
score row costs O(d); (b) accumulation is float64 (constant-factor
irrelevant); (c) chunks are balanced up to remainder of 1, which
_chunk_bounds guarantees by construction; (d) the complexity figures are
asymptotic in N, d, d_v, C, not measured memory-movers.

TEMPORARY MEMORY (honest accounting)
====================================
Persistent for the duration of the call:
  * O(C) for local maxima m_chunks.
  * O(C) for local normalizers l_chunks.
  * O(C*d_v) for local normalized outputs o_chunks.

Additional per-call memory that scales with input size:
  * O(d) for a promoted float64 copy of Q (only when Q is not already
    float64).
  * O(N*d) for a promoted float64 copy of K (only when K is not already
    float64).
  * O(N*d_v) for a promoted float64 copy of V (only when V is not
    already float64).
  * O(max(N*d, N*d_v)) peak for the boolean finiteness-validation
    temporaries inside np.isfinite(arr).all() (one tensor at a time).

Transient per-iteration temporaries (peak O(N/C + d_v)):
  * s, e, w of length n_c for the current chunk.

There is no O(N)-sized *index* array; chunk boundaries are (start, end)
tuples and chunk slices are views into K64 / V64. The above bounds are
asymptotic; NumPy/BLAS may allocate additional internal workspace on top.

This NumPy implementation runs chunks sequentially in a Python loop. It
is a correctness simulator; it does NOT demonstrate wall-clock speedup
or GPU occupancy improvements.

WHAT THIS IS NOT
================
  * Not a CUDA/GPU kernel or thread-block scheduler.
  * Not a benchmarked optimization.
  * Not a multi-query / grouped-query / multi-head / batched variant.
    Q is exactly shape (1, d); batched or headed inputs are rejected.
  * Not a persistent KV-cache manager.

DETERMINISM
===========
Same-environment repeatability: no RNG, no unordered iteration, and chunk
bounds are contiguous and produced in ascending order. Repeated calls
with identical inputs on the same BLAS backend yield bit-identical
outputs.

Cross-backend bitwise reproducibility is NOT claimed: different BLAS
implementations can reorder reductions and produce last-ULP differences.
The determinism claim is scoped to a single execution environment.

NUMERICAL CONTRACT
==================
  * Inputs must be float32 or float64, finite, and share a dtype.
  * Internal accumulation is float64; output is cast back to input dtype.
  * Per-chunk softmax subtracts the local max m_c before exp.
  * Per-chunk normalized output is computed as o_c = (w_c @ V_c) where
    w_c = exp(s - m_c) / l_c. Because sum(w_c) = 1 and w_c <= 1 in
    exact arithmetic, w_c @ V_c is mathematically bounded by max|V|.
    This avoids the *specific* unnormalized-numerator overflow
    demonstrated by test_large_values_with_finite_expected_output
    (u_c = 1e308 + 1e308 overflows while o_c = 1e308 is representable).
    General floating-point accumulation behavior for arbitrary
    extreme-magnitude values is NOT exhaustively verified; a pathological
    combination of rounding and accumulation order could still produce
    numerical issues not covered by the tests.
  * Global merge uses weights w_global,c = alpha_c * l_c / L, where
    alpha_c = exp(m_c - M), M = max_c m_c, and L = sum_c alpha_c * l_c.
    These weights are in [0, 1] and sum to 1, so the final output is a
    convex combination of bounded local outputs. The closed-interval form
    (allowing 0) is required because alpha_c underflows to exactly 0 in
    float64 when m_c - M < ~-745; see
    test_cross_chunk_extreme_magnitude_separation.
    Mathematically this equals U / L with U = sum_c alpha_c * u_c,
    u_c = exp(s - m_c) @ V_c.
  * Equivalence with a stable full-sequence attention oracle is
    guaranteed up to floating-point tolerance, not bitwise: reduction
    order differs across chunk partitions.
  * No epsilon clamp is used at the final division because L >= 1 for
    any finite non-empty input (one chunk attains M, its alpha is 1, and
    its l_c >= 1). The code asserts L is finite and positive and fails
    loudly otherwise.

Hard limitations
----------------
  * If Q @ K.T overflows float64 (e.g. |Q| ~ |K| ~ 1e308), the score
    matrix is not representable and the function raises
    FloatingPointError with a stage-specific message. This is a hard
    limitation of float64 accumulation, not a bug; supporting such
    inputs would require higher precision or log-domain score
    computation.
  * The output is representable iff max|V| is representable; the
    per-chunk normalization handles the specific weighted-mean overflow
    case demonstrated by tests, but cannot manufacture precision that
    float64 lacks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["flash_decode_forward"]


_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _chunk_bounds(N: int, num_chunks: int) -> list[tuple[int, int]]:
    """Half-open bounds [start, end) for C non-empty contiguous chunks.

    Matches ``numpy.array_split`` semantics: sizes differ by at most 1, and
    the first ``N % num_chunks`` chunks receive the remainder rows. Every
    index in ``[0, N)`` appears in exactly one chunk. Requires
    ``1 <= num_chunks <= N``.
    """
    base, rem = divmod(N, num_chunks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for c in range(num_chunks):
        size = base + (1 if c < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def flash_decode_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_chunks: int | np.integer = 4,
    sm_scale: float | None = None,
) -> np.ndarray:
    """FlashDecoding forward pass for a single decode query token.

    Parameters
    ----------
    Q : np.ndarray, shape (1, d), float32 or float64
    K : np.ndarray, shape (N, d), same dtype as Q
    V : np.ndarray, shape (N, d_v), same dtype as Q; ``d_v`` may differ
        from ``d``.
    num_chunks : int, default 4
        Python int or numpy integer. Must satisfy ``1 <= num_chunks <= N``.
        Booleans and non-integral floats are rejected.
    sm_scale : float or None, default None
        Softmax scale. ``None`` selects ``1/sqrt(d)``. Explicit values
        must be finite and strictly positive.

    Returns
    -------
    np.ndarray, shape (1, d_v), same dtype as inputs.

    Raises
    ------
    TypeError
        Non-float dtype, mixed dtypes, or non-numeric ``num_chunks`` /
        ``sm_scale``.
    ValueError
        Shape mismatch, non-finite values, ``num_chunks`` out of range,
        non-positive/non-finite ``sm_scale``, or ``sm_scale`` that cannot
        be converted to a finite float.
    FloatingPointError
        Intermediate scores are not representable in float64, or an
        internal invariant was violated (unreachable for valid inputs).
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)

    # ---- rank / shape ----
    if Q.ndim != 2 or Q.shape[0] != 1:
        raise ValueError(f"Q must have shape (1, d); got {Q.shape}")
    if K.ndim != 2:
        raise ValueError(f"K must have shape (N, d); got {K.shape}")
    if V.ndim != 2:
        raise ValueError(f"V must have shape (N, d_v); got {V.shape}")

    d = Q.shape[1]
    N = K.shape[0]
    d_v = V.shape[1]

    if d == 0 or N == 0 or d_v == 0:
        raise ValueError(
            f"Q, K, V must have non-empty dimensions; got d={d}, N={N}, d_v={d_v}"
        )
    if K.shape[1] != d:
        raise ValueError(f"K feature dim {K.shape[1]} does not match Q feature dim {d}")
    if V.shape[0] != N:
        raise ValueError(
            f"V sequence length {V.shape[0]} does not match K sequence length {N}"
        )

    # ---- dtype ----
    for name, arr in (("Q", Q), ("K", K), ("V", V)):
        if arr.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"{name} must be float32 or float64; got {arr.dtype}")
    if not (Q.dtype == K.dtype == V.dtype):
        raise TypeError(
            f"Q, K, V must share a dtype; got {Q.dtype}, {K.dtype}, {V.dtype}"
        )
    out_dtype = Q.dtype

    # ---- finiteness ----
    for name, arr in (("Q", Q), ("K", K), ("V", V)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinity")

    # ---- num_chunks ----
    if isinstance(num_chunks, bool) or not isinstance(num_chunks, (int, np.integer)):
        raise TypeError(
            f"num_chunks must be a positive int; got {type(num_chunks).__name__}"
        )
    num_chunks = int(num_chunks)
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive; got {num_chunks}")
    if num_chunks > N:
        raise ValueError(
            f"num_chunks ({num_chunks}) exceeds sequence length N ({N}); "
            f"policy is to reject rather than create empty chunks"
        )

    # ---- sm_scale ----
    if sm_scale is None:
        sm_scale = 1.0 / float(np.sqrt(d))
    else:
        if isinstance(sm_scale, bool) or not isinstance(
            sm_scale, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"sm_scale must be a real number; got {type(sm_scale).__name__}"
            )
        try:
            sm_scale = float(sm_scale)
        except OverflowError as exc:
            raise ValueError(
                f"sm_scale cannot be converted to a finite float: {exc}"
            ) from None
        if not np.isfinite(sm_scale) or sm_scale <= 0.0:
            raise ValueError(f"sm_scale must be finite and positive; got {sm_scale}")

    # Promote for accumulation; never mutate caller arrays. When input is
    # float32, K64 and V64 are O(N*d) and O(N*d_v) allocations respectively.
    Q64 = Q.astype(np.float64, copy=False)
    K64 = K.astype(np.float64, copy=False)
    V64 = V.astype(np.float64, copy=False)

    bounds = _chunk_bounds(N, num_chunks)
    C = len(bounds)

    m_chunks = np.empty(C, dtype=np.float64)
    l_chunks = np.empty(C, dtype=np.float64)
    o_chunks = np.empty((C, d_v), dtype=np.float64)

    for c, (start, end) in enumerate(bounds):
        Kc = K64[start:end]  # view
        Vc = V64[start:end]  # view
        # Suppress RuntimeWarning from intentional overflow detection: the
        # np.isfinite check below is the authoritative gate.
        with np.errstate(over="ignore"):
            s = (Q64 @ Kc.T)[0] * sm_scale
        if not np.isfinite(s).all():
            raise FloatingPointError(
                f"float64 score overflow in chunk {c}: Q @ K.T * sm_scale "
                f"is not representable. This is a hard float64 limit; "
                f"reduce input magnitude or sm_scale."
            )
        m_c = float(s.max())
        e = np.exp(s - m_c)  # (n_c,), all in (0, 1]
        l_c = float(e.sum())  # >= 1 since one e == 1
        w = e / l_c  # (n_c,), sum == 1
        m_chunks[c] = m_c
        l_chunks[c] = l_c
        o_chunks[c] = w @ Vc  # exact-arithmetic bound: max|V|

    # Global merge in normalized form:
    #   alpha_c = exp(m_c - M),  L = sum_c alpha_c * l_c
    #   w_global,c = alpha_c * l_c / L  (sum == 1)
    #   O = sum_c w_global,c * o_c
    # equals (sum_c alpha_c * u_c) / L with u_c = e @ V_c, but avoids the
    # specific overflow that the unnormalized numerator hits when |V| is
    # near float64 max (see test_large_values_with_finite_expected_output).
    M = float(m_chunks.max())
    alpha = np.exp(m_chunks - M)  # (C,), in [0, 1]; can underflow to 0
    L = float(alpha @ l_chunks)  # scalar, in [1, N]
    if not (np.isfinite(L) and L > 0.0):
        raise FloatingPointError(
            f"internal invariant violated: L={L!r}; "
            f"unreachable for validated finite non-empty inputs"
        )
    global_weights = (alpha * l_chunks) / L  # (C,), sums to 1
    O = global_weights @ o_chunks
    if not np.isfinite(O).all():
        raise FloatingPointError(
            "internal invariant violated: output contains NaN or infinity"
        )

    return O.astype(out_dtype, copy=False).reshape(1, d_v)
