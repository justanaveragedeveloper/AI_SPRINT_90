"""Verification suite for day51_flash_attention2.

Categories on every test:
  [ROBUST]   fails against a specific wrong implementation
  [CONTRACT] fails only if the public interface or documented contract changes

Not tested (and why) at the bottom of the file.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest
from day51_flash_attention2 import (
    flash_attention_backward,
    flash_attention_forward,
)

try:
    # Optional dependency: the property tests are skipped when it is absent.
    # Ruff/Pylance cannot know whether it is installed, so the import is
    # marked ignore-not-found and guarded by the try/except below.
    from hypothesis import (  # type: ignore[import-not-found]
        HealthCheck,
        given,
        settings,
    )
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# ---------------------------------------------------------------------------
# Dense oracles (test-only; never used by the tiled code path)
# ---------------------------------------------------------------------------
def dense_forward(Q, K, V, scale):
    S = scale * (Q @ K.T)
    m = S.max(axis=-1, keepdims=True)
    P = np.exp(S - m)
    l = P.sum(axis=-1, keepdims=True)
    return (P / l) @ V, m + np.log(l)


def dense_backward(Q, K, V, L, dO, scale):
    S = scale * (Q @ K.T)
    P = np.exp(S - L)
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))
    return scale * (dS @ K), scale * (dS.T @ Q), dV


def _finite_diff(param, Q, K, V, dO, scale, eps=1e-6, br=2, bc=3):
    """Central-difference gradient of sum(forward(Q,K,V)[0] * dO).

    eps=1e-6 gives ~1e-12 truncation and ~1e-10 round-off for O(1) values;
    the 1e-4 rtol in callers is deliberate headroom.
    """
    shapes = {"Q": Q.shape, "K": K.shape, "V": V.shape}
    grad = np.zeros(shapes[param], dtype=np.float64)
    for idx in np.ndindex(*shapes[param]):
        vals = []
        for sign in (+1.0, -1.0):
            Qp, Kp, Vp = Q.copy(), K.copy(), V.copy()
            {"Q": Qp, "K": Kp, "V": Vp}[param][idx] += sign * eps
            Op, _ = flash_attention_forward(
                Qp, Kp, Vp, sm_scale=scale, block_r=br, block_c=bc
            )
            vals.append(float(np.sum(Op * dO)))
        grad[idx] = (vals[0] - vals[1]) / (2.0 * eps)
    return grad


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------
def test_forward_matches_dense_oracle_unmasked():
    """[ROBUST] Fails if any online-softmax update is wrong (max, alpha,
    normalizer, or value accumulator)."""
    rng = _rng(0)
    Nq, Nk, d, dv = 7, 11, 5, 4
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    O_ref, L_ref = dense_forward(Q, K, V, scale)
    O, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=3, block_c=4)
    np.testing.assert_allclose(O, O_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(L, L_ref, rtol=1e-10, atol=1e-10)


def test_forward_returns_logsumexp_not_raw_normalizer():
    """[ROBUST] Fails if the returned L is the raw running sum ``l``
    rather than ``m + log(l)``."""
    rng = _rng(1)
    Nq, Nk, d, dv = 6, 9, 4, 4
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    S = scale * (Q @ K.T)
    m = S.max(axis=-1, keepdims=True)
    expected = m + np.log(np.sum(np.exp(S - m), axis=-1, keepdims=True))
    np.testing.assert_allclose(L, expected, rtol=1e-10, atol=1e-10)


def test_forward_block_size_invariance_numerical():
    """[ROBUST] Fails if any boundary bug in online rescaling makes output
    depend on (block_r, block_c). Compares values, not shapes. Shared-mode
    bugs (present in all tile configurations) are not detected by this
    test alone; the dense-comparison tests provide that coverage."""
    rng = _rng(2)
    Nq, Nk, d, dv = 10, 13, 6, 5
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    combos = [(2, 2), (4, 5), (10, 13)]
    outs = [
        flash_attention_forward(Q, K, V, sm_scale=scale, block_r=br, block_c=bc)
        for br, bc in combos
    ]
    for O, L in outs[1:]:
        np.testing.assert_allclose(O, outs[0][0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(L, outs[0][1], rtol=1e-10, atol=1e-10)


def test_forward_handles_non_divisible_blocks():
    """[ROBUST] Fails if the final partial query or key tile is truncated
    or skipped. Nq=7, Nk=11 with block_r=4, block_c=6 exercises both."""
    rng = _rng(3)
    Nq, Nk, d, dv = 7, 11, 5, 5
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    O_ref, _ = dense_forward(Q, K, V, scale)
    O, _ = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=4, block_c=6)
    np.testing.assert_allclose(O, O_ref, rtol=1e-10, atol=1e-10)


def test_forward_handles_nq_neq_nk():
    """[ROBUST] Fails if softmax reduces over the query axis (assuming
    Nq == Nk) instead of the key axis."""
    rng = _rng(4)
    Nq, Nk, d, dv = 3, 9, 4, 4
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    O_ref, _ = dense_forward(Q, K, V, scale)
    O, _ = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=4)
    np.testing.assert_allclose(O, O_ref, rtol=1e-10, atol=1e-10)


def test_forward_handles_dv_neq_d():
    """[ROBUST] Fails if the value accumulator reuses d as its width."""
    rng = _rng(5)
    Nq, Nk, d, dv = 5, 6, 3, 7
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    O, _ = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=2)
    assert O.shape == (Nq, dv)
    O_ref, _ = dense_forward(Q, K, V, scale)
    np.testing.assert_allclose(O, O_ref, rtol=1e-10, atol=1e-10)


def test_forward_single_row_degenerate():
    """[ROBUST] Fails if first-tile ``m = -inf`` produces NaN through
    ``-inf - -inf`` or a ``0/0`` division."""
    rng = _rng(6)
    Q = rng.standard_normal((1, 4))
    K = rng.standard_normal((1, 4))
    V = rng.standard_normal((1, 4))
    O, L = flash_attention_forward(Q, K, V, block_r=1, block_c=1)
    assert np.all(np.isfinite(O)) and np.all(np.isfinite(L))
    Q = rng.standard_normal((1, 4))
    K = rng.standard_normal((6, 4))
    V = rng.standard_normal((6, 4))
    O, L = flash_attention_forward(Q, K, V, block_r=1, block_c=2)
    O_ref, L_ref = dense_forward(Q, K, V, 1.0 / math.sqrt(4))
    np.testing.assert_allclose(O, O_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(L, L_ref, rtol=1e-10, atol=1e-10)


def test_forward_large_magnitude_no_naive_overflow():
    """[ROBUST] Fails if softmax is computed as ``exp(S)/sum(exp(S))``:
    scores reach |S| > 100 here and un-stabilized exp overflows float64."""
    rng = _rng(7)
    Nq, Nk, d, dv = 4, 5, 8, 3
    Q = rng.standard_normal((Nq, d)) * 30.0
    K = rng.standard_normal((Nk, d)) * 30.0
    V = rng.standard_normal((Nk, dv))
    scale = 1.0 / math.sqrt(d)
    assert np.max(np.abs(scale * (Q @ K.T))) > 100.0
    O, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    assert np.all(np.isfinite(O)) and np.all(np.isfinite(L))
    O_ref, _ = dense_forward(Q, K, V, scale)
    np.testing.assert_allclose(O, O_ref, rtol=1e-9, atol=1e-9)


def test_forward_softmax_normalizes_over_key_axis():
    """[ROBUST] Fails if normalization is applied over the query axis."""
    rng = _rng(8)
    Nq, Nk, d, dv = 5, 11, 3, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    scale = 1.0 / math.sqrt(d)
    O, _ = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=3, block_c=4)
    S = scale * (Q @ K.T)
    m = S.max(axis=-1, keepdims=True)
    Pn = np.exp(S - m)
    Pn = Pn / Pn.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(O, Pn @ V, rtol=1e-10, atol=1e-10)


def test_forward_finite_inputs_overflowing_matmul_raise():
    """[ROBUST] Fails if the forward pass does not detect non-finite
    intermediate scores. Finite ``1e308`` inputs overflow their dot
    product to ``inf``; the subsequent ``-inf - -inf`` becomes NaN. The
    contract requires FloatingPointError instead of silent NaN."""
    Q = np.array([[1e308]], dtype=np.float64)
    K = np.array([[1e308]], dtype=np.float64)
    V = np.array([[1.0]], dtype=np.float64)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError),
    ):
        flash_attention_forward(Q, K, V, sm_scale=1.0, block_r=1, block_c=1)


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------
def test_backward_matches_dense_reference_gradients():
    """[ROBUST] Fails if any of dQ, dK, dV is wrong (missing scale, wrong
    transpose, missing softmax-derivative term, wrong accumulation).
    Limitations: the dense backward oracle shares the same analytical
    equations as the tiled path, so this test alone cannot catch a formula
    error common to both; the finite-difference tests provide that
    independent evidence."""
    rng = _rng(10)
    Nq, Nk, d, dv = 6, 8, 4, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=3, block_c=4)
    dQ, dK, dV = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=3, block_c=4
    )
    dQ_ref, dK_ref, dV_ref = dense_backward(Q, K, V, L, dO, scale)
    np.testing.assert_allclose(dQ, dQ_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dK, dK_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dV, dV_ref, rtol=1e-10, atol=1e-10)


def test_backward_finite_difference_q():
    """[ROBUST] Fails if dQ is analytically inconsistent with the tiled
    forward, e.g. scale applied twice or the wrong contraction."""
    rng = _rng(11)
    Nq, Nk, d, dv = 3, 4, 3, 2
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    dQ, _, _ = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    np.testing.assert_allclose(
        dQ, _finite_diff("Q", Q, K, V, dO, scale), rtol=1e-4, atol=1e-6
    )


def test_backward_finite_difference_k():
    """[ROBUST] Fails if dK is analytically wrong (wrong scale, wrong
    transpose, wrong dS)."""
    rng = _rng(12)
    Nq, Nk, d, dv = 3, 4, 3, 2
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    _, dK, _ = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    np.testing.assert_allclose(
        dK, _finite_diff("K", Q, K, V, dO, scale), rtol=1e-4, atol=1e-6
    )


def test_backward_finite_difference_v():
    """[ROBUST] Fails if dV is not ``P.T @ dO`` (unnormalized P, wrong
    transpose, wrong tile rescale)."""
    rng = _rng(13)
    Nq, Nk, d, dv = 3, 4, 3, 2
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    _, _, dV = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    np.testing.assert_allclose(
        dV, _finite_diff("V", Q, K, V, dO, scale), rtol=1e-4, atol=1e-6
    )


def test_backward_scale_applied_exactly_once():
    """[ROBUST] Fails if scale is applied twice or dropped in dQ/dK. Uses
    scale=0.5 != 1/sqrt(d) so the three variants are numerically distinct."""
    rng = _rng(14)
    Nq, Nk, d, dv = 5, 6, 4, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 0.5
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    dQ, dK, _ = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    dQ_ref, dK_ref, _ = dense_backward(Q, K, V, L, dO, scale)
    np.testing.assert_allclose(dQ, dQ_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dK, dK_ref, rtol=1e-10, atol=1e-10)
    assert not np.allclose(dQ, dQ_ref / scale, rtol=1e-10, atol=1e-10)
    assert not np.allclose(dQ, dQ_ref * scale, rtol=1e-10, atol=1e-10)


def test_backward_softmax_derivative_includes_dp_p_reduction():
    """[ROBUST] Fails if the softmax Jacobian ``-sum_j P_ij dP_ij`` term
    is omitted (dS = P*dP instead of P*(dP - D))."""
    rng = _rng(15)
    Nq, Nk, d, dv = 5, 7, 4, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    dQ, _, _ = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    S = scale * (Q @ K.T)
    P = np.exp(S - L)
    dP = dO @ V.T
    dQ_mut = scale * ((P * dP) @ K)
    assert not np.allclose(dQ, dQ_mut, rtol=1e-8, atol=1e-8)


def test_backward_block_size_invariance_numerical():
    """[ROBUST] Fails if a tile boundary drops or duplicates a
    contribution to dQ, dK, or dV. Shared-mode accumulation bugs
    (present in both tile configurations) are not detected by this test
    alone."""
    rng = _rng(16)
    Nq, Nk, d, dv = 7, 9, 4, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L1 = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=2)
    _, L2 = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=3, block_c=4)
    np.testing.assert_allclose(L1, L2, rtol=1e-10, atol=1e-10)
    grads1 = flash_attention_backward(
        dO, Q, K, V, L1, sm_scale=scale, block_r=2, block_c=2
    )
    grads2 = flash_attention_backward(
        dO, Q, K, V, L2, sm_scale=scale, block_r=3, block_c=4
    )
    for a, b in zip(grads1, grads2):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_backward_reconstructs_probabilities_from_L():
    """[ROBUST] Fails if the backward pass uses a normalizer convention
    different from the forward (e.g. the raw running sum instead of
    log-sum-exp)."""
    rng = _rng(17)
    Nq, Nk, d, dv = 4, 6, 3, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)
    _, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    P = np.exp(scale * (Q @ K.T) - L)
    np.testing.assert_allclose(P.sum(axis=-1), np.ones(Nq), rtol=1e-10, atol=1e-10)
    _, _, dV = flash_attention_backward(
        dO, Q, K, V, L, sm_scale=scale, block_r=2, block_c=3
    )
    np.testing.assert_allclose(dV, P.T @ dO, rtol=1e-10, atol=1e-10)


def test_backward_finite_inputs_overflowing_matmul_raise():
    """[ROBUST] Fails if backward does not detect non-finite score tiles
    arising from finite-but-huge inputs. Requires FloatingPointError."""
    Q = np.array([[1e308, 1e308]], dtype=np.float64)
    K = np.array([[1e308, 1e308]], dtype=np.float64)
    V = np.array([[1.0, 1.0]], dtype=np.float64)
    L = np.array([[0.0]], dtype=np.float64)  # plausible-looking, finite
    dO = np.array([[1.0, 1.0]], dtype=np.float64)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError),
    ):
        flash_attention_backward(dO, Q, K, V, L, sm_scale=1.0, block_r=1, block_c=1)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_public_shapes_and_dtypes():
    """[CONTRACT] O is (Nq, dv) float64, L is (Nq, 1) float64."""
    rng = _rng(20)
    Nq, Nk, d, dv = 5, 6, 4, 3
    Q, K, V = (
        rng.standard_normal(s).astype(np.float32) for s in [(Nq, d), (Nk, d), (Nk, dv)]
    )
    O, L = flash_attention_forward(Q, K, V, block_r=2, block_c=2)
    assert O.shape == (Nq, dv) and O.dtype == np.float64
    assert L.shape == (Nq, 1) and L.dtype == np.float64


def test_inputs_not_mutated_and_deterministic():
    """[CONTRACT] Forward and backward must not modify caller arrays and
    must be byte-identical across repeated calls with identical inputs in
    this environment."""
    rng = _rng(21)
    Nq, Nk, d, dv = 4, 5, 3, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    Q0, K0, V0 = Q.copy(), K.copy(), V.copy()
    O1, L1 = flash_attention_forward(Q, K, V, block_r=2, block_c=2)
    L0 = L1.copy()
    dO = rng.standard_normal((Nq, dv))
    dO0 = dO.copy()
    flash_attention_backward(dO, Q, K, V, L1, block_r=2, block_c=2)
    np.testing.assert_array_equal(Q, Q0)
    np.testing.assert_array_equal(K, K0)
    np.testing.assert_array_equal(V, V0)
    np.testing.assert_array_equal(L1, L0)
    np.testing.assert_array_equal(dO, dO0)
    O2, L2 = flash_attention_forward(Q, K, V, block_r=2, block_c=2)
    np.testing.assert_array_equal(O1, O2)
    np.testing.assert_array_equal(L1, L2)


def test_backward_documents_L_provenance_precondition():
    """[CONTRACT] Documents that L must come from the matching forward
    call. Passing an L computed from different Q/K with the same shape is
    accepted (only shape and finiteness are validated); the returned
    gradients differ from the correct ones. This test locks the current
    behavior as an explicit contract, not a silent assumption."""
    rng = _rng(51)
    Nq, Nk, d, dv = 5, 6, 4, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    Q2 = rng.standard_normal((Nq, d))
    K2 = rng.standard_normal((Nk, d))
    dO = rng.standard_normal((Nq, dv))
    scale = 1.0 / math.sqrt(d)

    _, L_mismatched = flash_attention_forward(
        Q2, K2, V, sm_scale=scale, block_r=2, block_c=3
    )
    dQ_bad, _, _ = flash_attention_backward(
        dO, Q, K, V, L_mismatched, sm_scale=scale, block_r=2, block_c=3
    )

    _, L_ok = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=2, block_c=3)
    dQ_ok, _, _ = flash_attention_backward(
        dO, Q, K, V, L_ok, sm_scale=scale, block_r=2, block_c=3
    )

    # The mismatched L is NOT rejected: shape and finiteness pass.
    assert np.all(np.isfinite(dQ_bad))
    # And the resulting gradients are silently wrong relative to the
    # correct ones for (Q, K, V). This is the documented precondition.
    assert not np.allclose(dQ_bad, dQ_ok, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Guards (classified ROBUST: each catches a specific missing validation)
# ---------------------------------------------------------------------------
def test_rejects_bad_qkv_and_dtypes():
    """[ROBUST] Fails if any of the documented input-shape, dtype, or
    finiteness validations is missing."""
    rng = _rng(30)
    Q = rng.standard_normal((4, 3))
    K = rng.standard_normal((4, 3))
    V = rng.standard_normal((4, 3))
    with pytest.raises(TypeError):
        flash_attention_forward([[1.0]], K, V)
    with pytest.raises(ValueError):
        flash_attention_forward(Q.ravel(), K, V)
    with pytest.raises(ValueError):
        flash_attention_forward(Q, rng.standard_normal((4, 5)), V)
    with pytest.raises(ValueError):
        flash_attention_forward(Q, K, rng.standard_normal((5, 3)))
    with pytest.raises(ValueError):
        flash_attention_forward(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
    with pytest.raises(ValueError):
        Qn = Q.copy()
        Qn[0, 0] = np.nan
        flash_attention_forward(Qn, K, V)
    with pytest.raises(TypeError):
        flash_attention_forward(
            Q.astype(np.int64), K.astype(np.int64), V.astype(np.int64)
        )


def test_rejects_invalid_blocks_and_bools():
    """[ROBUST] Fails if zero, negative, non-int, or boolean block sizes
    are accepted (bools are ints in Python)."""
    rng = _rng(31)
    Q = rng.standard_normal((4, 3))
    K = rng.standard_normal((4, 3))
    V = rng.standard_normal((4, 3))
    for bad in (0, -1):
        with pytest.raises(ValueError):
            flash_attention_forward(Q, K, V, block_r=bad)
        with pytest.raises(ValueError):
            flash_attention_forward(Q, K, V, block_c=bad)
    for bad in (True, False):
        with pytest.raises(TypeError):
            flash_attention_forward(Q, K, V, block_r=bad)
        with pytest.raises(TypeError):
            flash_attention_forward(Q, K, V, block_c=bad)
    with pytest.raises(TypeError):
        flash_attention_forward(Q, K, V, block_r=2.0)


def test_rejects_invalid_scale():
    """[ROBUST] Fails if Inf, NaN, or boolean scale is accepted."""
    rng = _rng(32)
    Q = rng.standard_normal((4, 3))
    K = rng.standard_normal((4, 3))
    V = rng.standard_normal((4, 3))
    with pytest.raises(ValueError):
        flash_attention_forward(Q, K, V, sm_scale=float("inf"))
    with pytest.raises(ValueError):
        flash_attention_forward(Q, K, V, sm_scale=float("nan"))
    with pytest.raises(TypeError):
        flash_attention_forward(Q, K, V, sm_scale=True)


def test_rejects_invalid_backward_arguments():
    """[ROBUST] Fails if missing L, wrong L/dO shape, or non-finite L/dO
    is accepted."""
    rng = _rng(33)
    Nq, Nk, d, dv = 4, 5, 3, 3
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    _, L = flash_attention_forward(Q, K, V, block_r=2, block_c=2)
    dO = rng.standard_normal((Nq, dv))
    with pytest.raises(TypeError):
        flash_attention_backward(dO, Q, K, V, None)
    with pytest.raises(ValueError):
        flash_attention_backward(dO, Q, K, V, L.ravel())
    with pytest.raises(ValueError):
        flash_attention_backward(rng.standard_normal((Nq, dv + 1)), Q, K, V, L)
    with pytest.raises(ValueError):
        bad = dO.copy()
        bad[0, 0] = np.inf
        flash_attention_backward(bad, Q, K, V, L)
    with pytest.raises(ValueError):
        badL = L.copy()
        badL[0, 0] = np.nan
        flash_attention_backward(dO, Q, K, V, badL)


# ---------------------------------------------------------------------------
# Tiled-memory contract
# ---------------------------------------------------------------------------
def test_forward_does_not_allocate_full_score_matrix():
    """[ROBUST] Fails if the tiled forward pass materializes a full
    (Nq, Nk) score or probability matrix. Uses tracemalloc to bound the
    peak Python-visible allocation during a single forward call. A spy
    on numpy.exp/numpy.max is not sufficient because an implementation
    could allocate the full matrix and only slice it for ufuncs. This is
    a heuristic bound on one configuration, not a proof of a global
    allocation bound."""
    rng = _rng(40)
    Nq, Nk, d, dv = 128, 128, 8, 8
    br, bc = 16, 16
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])

    full_score_bytes = Nq * Nk * np.dtype(np.float64).itemsize

    # Sanity: tracemalloc must observe a known-size numpy allocation,
    # otherwise this test cannot validate the memory contract.
    tracemalloc.start()
    _ = np.zeros((Nq, Nk), dtype=np.float64)
    _, sanity_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sanity_peak >= full_score_bytes, (
        "tracemalloc is not tracking numpy allocations; cannot validate "
        "the memory contract in this environment"
    )

    # Warm up, then measure only the tiled forward call.
    flash_attention_forward(Q, K, V, block_r=br, block_c=bc)
    tracemalloc.start()
    flash_attention_forward(Q, K, V, block_r=br, block_c=bc)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    threshold = full_score_bytes // 4
    assert peak < threshold, (
        f"peak traced allocation {peak} bytes >= threshold {threshold} bytes; "
        f"full (Nq, Nk) score matrix is {full_score_bytes} bytes — the tiled "
        f"path appears to materialize it"
    )


def test_backward_does_not_allocate_full_score_matrix():
    """[ROBUST] Fails if the tiled backward pass materializes a full
    (Nq, Nk) score, probability, dP, or dS matrix. Mirrors the forward
    memory test with a looser threshold.

    The backward pass legitimately allocates full-size gradient
    accumulators dQ (Nq, d), dK (Nk, d), dV (Nk, dv) and, on the tested
    configuration, additional per-tile and per-query-block scratch. The
    measured peak on a 128x128 example with 16x16 tiles was observed to
    exceed one quarter of the full score matrix, so the threshold is set
    to one half of the full score matrix. It still fails if the tiled
    backward path materializes a full Nq x Nk intermediate. This is a
    heuristic bound on one configuration, not a proof of a global
    allocation bound, and it depends on tracemalloc observing the
    relevant allocations."""
    rng = _rng(41)
    Nq, Nk, d, dv = 128, 128, 8, 8
    br, bc = 16, 16
    Q, K, V = (rng.standard_normal(s) for s in [(Nq, d), (Nk, d), (Nk, dv)])
    dO = rng.standard_normal((Nq, dv))
    _, L = flash_attention_forward(Q, K, V, block_r=br, block_c=bc)

    full_score_bytes = Nq * Nk * np.dtype(np.float64).itemsize

    # Sanity: confirm tracemalloc sees the reference allocation.
    tracemalloc.start()
    _ = np.zeros((Nq, Nk), dtype=np.float64)
    _, sanity_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sanity_peak >= full_score_bytes, (
        "tracemalloc is not tracking numpy allocations; cannot validate "
        "the memory contract in this environment"
    )

    # Warm up, then measure only the tiled backward call.
    flash_attention_backward(dO, Q, K, V, L, block_r=br, block_c=bc)
    tracemalloc.start()
    flash_attention_backward(dO, Q, K, V, L, block_r=br, block_c=bc)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    threshold = full_score_bytes // 2
    assert peak < threshold, (
        f"peak traced allocation {peak} bytes >= threshold {threshold} bytes; "
        f"full (Nq, Nk) score matrix is {full_score_bytes} bytes — the tiled "
        f"backward path appears to materialize a full attention intermediate"
    )


# ---------------------------------------------------------------------------
# Oracle validation (mutation 21)
# ---------------------------------------------------------------------------
def test_dense_oracle_stable_at_large_magnitude():
    """[ROBUST] Validates the test oracle itself. A naive
    exp(S)/sum(exp(S)) oracle would overflow at |S| > 100 and either
    return NaN or disagree with the stable reference; this test would
    fail in that case. Without it, an unstable oracle could hide a
    correct implementation or mask an incorrect one."""
    rng = _rng(50)
    Nq, Nk, d, dv = 3, 4, 8, 2
    Q = rng.standard_normal((Nq, d)) * 30.0
    K = rng.standard_normal((Nk, d)) * 30.0
    V = rng.standard_normal((Nk, dv))
    scale = 1.0 / math.sqrt(d)
    assert np.max(np.abs(scale * (Q @ K.T))) > 100.0
    O, L = dense_forward(Q, K, V, scale)
    assert np.all(np.isfinite(O)) and np.all(np.isfinite(L))
    P = np.exp(scale * (Q @ K.T) - L)
    np.testing.assert_allclose(P.sum(axis=-1), np.ones(Nq), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
# PROPERTY (Hypothesis): for any Nq, Nk in [1, 8], d, dv in [1, 6],
# block_r in [1, Nq], block_c in [1, Nk], finite entries in [-3, 3]:
#   (1) tiled forward matches the dense oracle.
#   (2) tiled backward gradients match the dense reference.
#   (3) inputs are unchanged after both calls.
# Not collected in environments without Hypothesis.
if HAS_HYPOTHESIS:

    @st.composite
    def _case(draw):
        Nq = draw(st.integers(1, 8))
        Nk = draw(st.integers(1, 8))
        d = draw(st.integers(1, 6))
        dv = draw(st.integers(1, 6))
        vals = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
        to = lambda n, m: np.array(
            draw(st.lists(vals, min_size=n * m, max_size=n * m)), dtype=np.float64
        ).reshape(n, m)
        return (
            to(Nq, d),
            to(Nk, d),
            to(Nk, dv),
            to(Nq, dv),
            draw(st.integers(1, Nq)),
            draw(st.integers(1, Nk)),
        )

    @settings(
        max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    @given(_case())
    def test_property_tiled_matches_dense(case):
        """[ROBUST] Property test: tiled forward and backward match dense
        oracles on bounded random inputs."""
        Q, K, V, dO, br, bc = case
        scale = 1.0 / math.sqrt(Q.shape[1])
        O, L = flash_attention_forward(Q, K, V, sm_scale=scale, block_r=br, block_c=bc)
        O_ref, L_ref = dense_forward(Q, K, V, scale)
        np.testing.assert_allclose(O, O_ref, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(L, L_ref, rtol=1e-9, atol=1e-9)
        dQ, dK, dV = flash_attention_backward(
            dO, Q, K, V, L, sm_scale=scale, block_r=br, block_c=bc
        )
        dQ_ref, dK_ref, dV_ref = dense_backward(Q, K, V, L, dO, scale)
        np.testing.assert_allclose(dQ, dQ_ref, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(dK, dK_ref, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(dV, dV_ref, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Not tested (and why):
# - GPU SRAM/HBM placement, CUDA execution, kernel fusion, real FA-2
#   speedup: this is a CPU NumPy reference; those claims are out of scope.
# - Multi-head, batched, causal, dropout, padding masks: outside the
#   Day-51 contract and not accepted by the public API.
# - fp16 / bf16 numeric paths: rejected by dtype policy.
# - Very long sequences (N > 10^4): tiling correctness is instead
#   covered by block-size invariance and non-divisible-block tests.
# - Torch autograd integration: out of scope for a NumPy reference.
# - L-provenance enforcement: documented as a caller contract (see
#   test_backward_documents_L_provenance_precondition); not enforced at
#   runtime, so no test asserts a rejection of mismatched L.
# - Native (non-Python-visible) allocations: tracemalloc observes
#   allocations made through Python-visible buffers. Neither memory test
#   proves anything about allocator internal bookkeeping or GPU memory.
#   They only bound the peak size of arrays the interpreter can see.
# - Shared-mode oracle errors: the dense backward oracle shares the same
#   analytical equations as the tiled backward path. Finite-difference
#   tests on small float64 inputs provide independent evidence; they do
#   not prove the equations for all inputs.
# - Automated mutation testing: the mutation table in the review is a
#   reasoned mapping, not an executed mutation harness.
# - Overflow warnings emitted by numpy during the deliberate extreme-value
#   tests: numpy still performs the overflowing multiply and prints a
#   RuntimeWarning; the implementation raises FloatingPointError after
#   detecting the non-finite result. This is expected behavior.
# ---------------------------------------------------------------------------
