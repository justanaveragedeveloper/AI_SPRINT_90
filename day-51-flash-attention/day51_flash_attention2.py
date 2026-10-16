"""Tiled FlashAttention-style attention in NumPy.

Single-head, unmasked scaled dot-product attention computed with tiled
online softmax. No full ``(Nq, Nk)`` score or probability matrix is
explicitly constructed by the tiled Python paths. Educational CPU
reference inspired by FlashAttention-2; not a CUDA kernel and makes no
hardware-performance claim.

Shapes
------
Q: (Nq, d)    K: (Nk, d)    V: (Nk, dv)
O: (Nq, dv)   L: (Nq, 1)    row-wise log-sum-exp of scaled scores
dO: (Nq, dv)  dQ: (Nq, d)   dK: (Nk, d)   dV: (Nk, dv)

Complexity
----------
Arithmetic: O(Nq * Nk * (d + dv)). Tiling does NOT remove the quadratic
dependence on sequence length; it only avoids the O(Nq * Nk) intermediate.
This is an asymptotic count from the visible NumPy operations, not a
benchmark.

Memory
------
Forward
    Tile scratch:  O(block_r * block_c + block_r * dv + block_c * dv)
    Total live:    input arrays (or their float64 up-cast copies),
                   O (Nq, dv), L (Nq, 1), and the tile scratch.

Backward
    Tile scratch:  O(block_r * block_c + block_r * dv + block_c * dv)
                   (S, P, dP, dS, dQ_blk per tile)
    Additional:    dQ (Nq, d), dK (Nk, d), dV (Nk, dv) — full-size
                   gradient accumulators that persist across the whole
                   call. Plus a per-query-block row reduction D of shape
                   (block_r, 1) and a per-query-block dQ_blk of shape
                   (block_r, d).

    No full Nq x Nk intermediate is explicitly constructed in either
    pass. The two-pass backward recomputes score and probability tiles
    so that no full-key-axis intermediate has to be retained; it is not
    a measured FLOP comparison against any alternative design.

No claim is made about GPU SRAM residency or HBM traffic; those require a
hardware-specific kernel and measurement.

L provenance (backward only)
----------------------------
``flash_attention_backward`` requires L to be the value returned by
``flash_attention_forward`` for the SAME Q, K, and scale. The function
validates L's shape and finiteness only. Passing a finite L computed from
different inputs will not raise; the returned gradients will then not be
the derivative of attention on the supplied Q/K/V. This is a documented
caller contract, not a runtime-enforced invariant.

Numerical-failure policy
------------------------
Finite inputs can still overflow a dot product (e.g. ``1e308 * 1e308``).
The implementation checks each tiled score, each backward probability
tile, and each final result, raising ``FloatingPointError`` rather than
returning NaN/Inf.

Determinism
-----------
No random-number generation is used. Repeated calls with identical inputs
were byte-identical in the environment where the test suite was executed;
this is an observed property of that environment, not a guarantee across
NumPy/BLAS versions, CPU architectures, or thread counts.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["flash_attention_backward", "flash_attention_forward"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _check_array(x: object, name: str) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(x).__name__}")
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name} must be float32 or float64, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {x.shape}")


def _check_block(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive int, got {type(value).__name__}")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _validate_qkv(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> None:
    _check_array(Q, "Q")
    _check_array(K, "K")
    _check_array(V, "V")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(f"Q, K feature dims must match: {Q.shape} vs {K.shape}")
    if K.shape[0] != V.shape[0]:
        raise ValueError(f"K, V sequence lengths must match: {K.shape} vs {V.shape}")
    if Q.shape[0] == 0 or K.shape[0] == 0:
        raise ValueError("Nq and Nk must be positive")
    if Q.shape[1] == 0 or V.shape[1] == 0:
        raise ValueError("d and dv must be positive")
    for name, arr in (("Q", Q), ("K", K), ("V", V)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")


def _resolve_scale(sm_scale: object, d: int) -> float:
    if sm_scale is None:
        return 1.0 / math.sqrt(d)
    if isinstance(sm_scale, bool) or not isinstance(
        sm_scale, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"sm_scale must be a float or None, got {type(sm_scale).__name__}"
        )
    scale = float(sm_scale)
    if not math.isfinite(scale):
        raise ValueError(f"sm_scale must be finite, got {scale}")
    return scale


def _require_finite(x: np.ndarray, name: str) -> None:
    """Raise FloatingPointError if x contains NaN or Inf.

    Finite float inputs can still overflow a dot product. Checking each
    tiled score, each backward probability tile, and each final result
    converts that into a loud failure instead of a silent NaN that later
    poisons downstream aggregates.
    """
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(
            f"{name} contains non-finite values (overflow or invalid arithmetic)"
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    sm_scale: float | None = None,
    block_r: int = 64,
    block_c: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiled online-softmax forward attention.

    Returns
    -------
    O : (Nq, dv) float64
    L : (Nq, 1) float64 — row-wise log-sum-exp ``m + log(l)`` of scaled scores.

    Online update per tile:
        m_new = max(m, m_block)
        alpha = exp(m - m_new)
        l     = alpha * l + sum(exp(S - m_new), axis=-1)
        acc   = alpha * acc + exp(S - m_new) @ V_block
    Final output is ``acc / l``. First-tile init uses ``m = -inf`` and
    ``l = 0``; ``exp(-inf - finite) == 0`` avoids NaN.

    Live memory
    -----------
    Per tile: S, P (block_r, block_c); V_blk (block_c, dv); running
    m, l (block_r, 1); acc (block_r, dv). No full (Nq, Nk) matrix is
    explicitly constructed. Plus O (Nq, dv) and L (Nq, 1) for the results.

    Raises FloatingPointError if any tiled score, the final output, or the
    final normalizer contains non-finite values.
    """
    _validate_qkv(Q, K, V)
    block_r = _check_block(block_r, "block_r")
    block_c = _check_block(block_c, "block_c")
    scale = _resolve_scale(sm_scale, Q.shape[1])

    Nq = Q.shape[0]
    Nk = K.shape[0]
    dv = V.shape[1]
    Qf = Q.astype(np.float64, copy=False)
    Kf = K.astype(np.float64, copy=False)
    Vf = V.astype(np.float64, copy=False)

    O = np.empty((Nq, dv), dtype=np.float64)
    L = np.empty((Nq, 1), dtype=np.float64)

    for r0 in range(0, Nq, block_r):
        r1 = min(r0 + block_r, Nq)
        Q_blk = Qf[r0:r1]
        rows = r1 - r0
        m = np.full((rows, 1), -np.inf, dtype=np.float64)
        l = np.zeros((rows, 1), dtype=np.float64)
        acc = np.zeros((rows, dv), dtype=np.float64)

        for c0 in range(0, Nk, block_c):
            c1 = min(c0 + block_c, Nk)
            S = scale * (Q_blk @ Kf[c0:c1].T)
            _require_finite(S, "forward score tile")
            m_new = np.maximum(m, S.max(axis=1, keepdims=True))
            alpha = np.exp(m - m_new)
            P = np.exp(S - m_new)
            l = alpha * l + P.sum(axis=1, keepdims=True)
            acc = alpha * acc + P @ Vf[c0:c1]
            m = m_new

        O[r0:r1] = acc / l
        L[r0:r1] = m + np.log(l)

    _require_finite(O, "forward output O")
    _require_finite(L, "forward normalizer L")
    return O, L


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
def flash_attention_backward(
    dO: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    sm_scale: float | None = None,
    block_r: int = 64,
    block_c: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiled backward attention.

    Dense reference:
        P    = exp(scale * (Q @ K.T) - L)
        dV   = P.T @ dO
        dP   = dO @ V.T
        dS   = P * (dP - sum(dP * P, axis=-1, keepdims=True))
        dQ   = scale * (dS @ K)
        dK   = scale * (dS.T @ Q)

    Two tiled passes over the key axis: pass 1 builds the per-row
    reduction ``D_i = sum_j P_ij dP_ij``; pass 2 accumulates dQ, dK, dV.

    Live memory
    -----------
    Per tile (recomputed in both passes): S, P (block_r, block_c);
    dP (block_r, dv). Pass 1 also carries D (block_r, 1). Pass 2 also
    carries dQ_blk (block_r, d) and reads K_blk (block_c, d),
    V_blk (block_c, dv).

    Persistent buffers across the whole call:
        dQ (Nq, d), dK (Nk, d), dV (Nk, dv).
    These are the returned gradients, not tile intermediates.

    No full Nq x Nk intermediate is explicitly constructed in either
    pass. This is the intended asymptotic bound from the visible NumPy
    operations; it does not reduce to O(block_r * block_c) alone, and it
    is not a measured peak-memory statement.

    Precondition
    ------------
    L must be the value returned by ``flash_attention_forward`` for the
    same Q, K, and scale. Shape and finiteness are validated; provenance
    is not. See module docstring.

    Raises FloatingPointError if any tiled score, any probability tile,
    any dP tile, or any final gradient contains non-finite values.
    """
    _validate_qkv(Q, K, V)
    _check_array(dO, "dO")
    _check_array(L, "L")
    block_r = _check_block(block_r, "block_r")
    block_c = _check_block(block_c, "block_c")
    scale = _resolve_scale(sm_scale, Q.shape[1])

    Nq, d = Q.shape
    Nk = K.shape[0]
    dv = V.shape[1]

    if L.shape != (Nq, 1):
        raise ValueError(f"L must have shape ({Nq}, 1), got {L.shape}")
    if dO.shape != (Nq, dv):
        raise ValueError(f"dO must have shape ({Nq}, {dv}), got {dO.shape}")
    if not np.all(np.isfinite(L)):
        raise ValueError("L contains NaN or Inf")
    if not np.all(np.isfinite(dO)):
        raise ValueError("dO contains NaN or Inf")

    Qf = Q.astype(np.float64, copy=False)
    Kf = K.astype(np.float64, copy=False)
    Vf = V.astype(np.float64, copy=False)
    Lf = L.astype(np.float64, copy=False)
    dOf = dO.astype(np.float64, copy=False)

    dQ = np.zeros((Nq, d), dtype=np.float64)
    dK = np.zeros((Nk, d), dtype=np.float64)
    dV = np.zeros((Nk, dv), dtype=np.float64)

    for r0 in range(0, Nq, block_r):
        r1 = min(r0 + block_r, Nq)
        Q_blk = Qf[r0:r1]
        dO_blk = dOf[r0:r1]
        L_blk = Lf[r0:r1]

        # Pass 1 — per-row reduction D_i = sum_j P_ij * dP_ij
        D = np.zeros((r1 - r0, 1), dtype=np.float64)
        for c0 in range(0, Nk, block_c):
            c1 = min(c0 + block_c, Nk)
            S = scale * (Q_blk @ Kf[c0:c1].T)
            _require_finite(S, "backward score tile")
            P = np.exp(S - L_blk)
            _require_finite(P, "backward probability tile")
            dP = dO_blk @ Vf[c0:c1].T
            _require_finite(dP, "backward dP tile")
            D += np.sum(P * dP, axis=1, keepdims=True)

        # Pass 2 — accumulate dQ, dK, dV
        dQ_blk = np.zeros_like(Q_blk)
        for c0 in range(0, Nk, block_c):
            c1 = min(c0 + block_c, Nk)
            K_blk = Kf[c0:c1]
            V_blk = Vf[c0:c1]
            S = scale * (Q_blk @ K_blk.T)
            _require_finite(S, "backward score tile")
            P = np.exp(S - L_blk)
            _require_finite(P, "backward probability tile")
            dP = dO_blk @ V_blk.T
            _require_finite(dP, "backward dP tile")
            dS = P * (dP - D)
            dQ_blk += scale * (dS @ K_blk)
            dK[c0:c1] += scale * (dS.T @ Q_blk)
            dV[c0:c1] += P.T @ dO_blk

        dQ[r0:r1] = dQ_blk

    _require_finite(dQ, "dQ")
    _require_finite(dK, "dK")
    _require_finite(dV, "dV")
    return dQ, dK, dV
