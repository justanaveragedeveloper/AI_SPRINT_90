"""Tests for day53_moe_router.

Category counts (per master review prompt v2; [GUARD] tests are counted
as [ROBUST] because they fail against a specific mutated implementation):

    ROBUST:   39 test functions
                 - 21 behavioural / numerical
                 - 17 [GUARD] input-validation
                 -  1 Hypothesis property test (skipped without `hypothesis`)
    CONTRACT:  2 test functions
    THEATER:   0 test functions
    Total:    41 test functions

Collected pytest cases: 45
    - 39 ROBUST functions; 2 are parametrized (x3 each)
      -> 39 - 2 + 6 = 43 cases, 1 skipped if `hypothesis` is absent
    - 2 CONTRACT functions, not parametrized -> 2 cases
    - With `hypothesis` installed:   45 passed
    - Without `hypothesis`:          44 passed, 1 skipped

Actual execution (against the prior 41-function revision, before the
two test bodies were extended per the adversarial review):
    44 passed, 0 failed, 1 skipped in 0.09s, Hypothesis unavailable.

This revision extends two existing test bodies (W_g corruption cases;
property-test independent softmax + selected-weight assertions). No test
functions added or deleted. This revision has NOT been executed; the
pass counts above are code-trace predictions.

Every test names the mutation it catches in its docstring.

Run from the day-53-moe-router directory:
    python -m pytest test_moe_router.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from day53_moe_router import (
    RoutingResult,
    Top2MoERouter,
    compute_load_balancing_loss,
)

# --------------------------------------------------------------------------- #
# Optional Hypothesis. If absent, given/settings become skip-decorators.
# --------------------------------------------------------------------------- #
try:
    from hypothesis import given, settings, strategies as st  # type: ignore  # noqa: I001

    HYPOTHESIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    HYPOTHESIS_AVAILABLE = False

    def given(*a, **k):
        def deco(f):
            return pytest.mark.skip(reason="hypothesis not installed")(f)

        return deco

    def settings(*a, **k):
        def deco(f):
            return f

        return deco

    class _StubStrategies:
        @staticmethod
        def integers(*a, **k):
            return None

    st = _StubStrategies()


# --------------------------------------------------------------------------- #
# Independent oracles (never call the implementation's internals)
# --------------------------------------------------------------------------- #


def _naive_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Explicit triple-loop matmul — independent of numpy's X @ W."""
    N, d = X.shape
    d2, E = W.shape
    assert d == d2
    out = np.zeros((N, E), dtype=np.float64)
    for i in range(N):
        for j in range(E):
            s = 0.0
            for k in range(d):
                s += X[i, k] * W[k, j]
            out[i, j] = s
    return out


def _reference_softmax(logits: np.ndarray) -> np.ndarray:
    """Textbook row-wise softmax with max-subtraction."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _make_router_with_weights(W_g: np.ndarray, seed: int = 0) -> Top2MoERouter:
    d, E = W_g.shape
    r = Top2MoERouter(d_model=d, num_experts=E, seed=seed)
    r.W_g = W_g.astype(np.float64, copy=True)
    return r


# --------------------------------------------------------------------------- #
# [ROBUST] tests
# --------------------------------------------------------------------------- #


def test_router_logits_equal_independent_matmul():
    """[ROBUST] Fails on W_g @ X, X @ W_g.T, X.T @ W_g. Non-symmetric
    W_g; oracle is triple-loop matmul, not numpy @."""
    d, E = 3, 3  # noqa: F841
    W_g = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    r = _make_router_with_weights(W_g)
    X = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.5]])
    result = r.route(X)
    expected_logits = _naive_matmul(X, W_g)
    expected_probs = _reference_softmax(expected_logits)
    np.testing.assert_allclose(result.probs, expected_probs, rtol=1e-12, atol=1e-12)


def test_softmax_handles_extreme_finite_logits():
    """[ROBUST] Fails if softmax omits max-subtraction: logits +1000 and
    +999 -> direct exp overflows to inf -> NaN. Also fails if the sum is
    taken along the wrong axis."""
    W_g = np.eye(2, 2)
    r = _make_router_with_weights(W_g)
    X = np.array([[1000.0, 999.0], [-1000.0, 500.0]])
    result = r.route(X)
    assert np.all(np.isfinite(result.probs))
    e = np.exp(-1.0)
    expected_row0 = np.array([1.0 / (1.0 + e), e / (1.0 + e)])
    np.testing.assert_allclose(result.probs[0], expected_row0, rtol=1e-12)
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, rtol=1e-12)


def test_softmax_handles_opposite_sign_extreme_logits():
    """[ROBUST] Fails if (a) max-subtraction overflows to -inf for a
    logit near -MAX_FLOAT when row_max is near +MAX_FLOAT and the
    resulting -inf is treated as invalid rather than as the correct
    underflow-to-zero, or (b) any RuntimeWarning is emitted for the
    expected range event. Runs under warnings.simplefilter("error")."""
    W_g = np.eye(2, 2)
    r = _make_router_with_weights(W_g)
    X = np.array([[1e308, -1e308]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = r.route(X)
    # logits: [1e308, -1e308]; row_max: 1e308
    # shifted: [0, -2e308] = [0, -inf] (overflow suppressed)
    # exp: [1, 0]; probs: [1.0, 0.0]
    assert np.all(np.isfinite(result.probs))
    np.testing.assert_allclose(result.probs[0], np.array([1.0, 0.0]), atol=1e-300)
    assert np.all(np.isfinite(result.top2_weights))


def test_route_raises_runtime_error_on_logit_overflow():
    """[ROBUST] Fails if route() silently returns NaN probs when finite X
    and finite W_g overflow the matmul, OR if the overflow path emits an
    uncontrolled RuntimeWarning (numpy default `over='warn'`). Runs under
    warnings.simplefilter("error") so an un-suppressed warning raises
    before the expected RuntimeError, and the pytest.raises match fails.
    Verifies the guard is a raise, not an assert (survives python -O)."""
    r = _make_router_with_weights(np.array([[1e308, 1e308]]))
    X = np.array([[1e308]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="non-finite"):
            r.route(X)


def test_top2_indices_correct_for_controlled_weights():
    """[ROBUST] Fails on wrong k, probs[:, :2] substitute, double-negation.
    X chosen so logits are exact small integers."""
    W_g = np.array([[1.0, 2.0, 3.0, 4.0]])
    r = _make_router_with_weights(W_g)
    X = np.array([[0.0], [1.0], [10.0], [-5.0]])
    result = r.route(X)
    expected = np.array([[0, 1], [3, 2], [3, 2], [0, 1]])
    np.testing.assert_array_equal(result.top2_indices, expected)


def test_top2_tie_breaks_to_lower_index():
    """[ROBUST] Fails on non-stable argsort (arbitrary tie order) or
    ties-to-higher-index. All-equal logits must yield exactly [0, 1]."""
    r = _make_router_with_weights(np.zeros((1, 5)))
    X = np.ones((4, 1))
    result = r.route(X)
    np.testing.assert_array_equal(
        result.top2_indices, np.tile(np.array([0, 1]), (4, 1))
    )


def test_top2_weights_are_selected_probs_renormalized():
    """[ROBUST] Fails on probs[:, :2] instead of probs[arange, top2_indices].
    Deliberately makes the top-2 be columns 3 and 2."""
    r = _make_router_with_weights(np.array([[0.0, 1.0, 2.0, 3.0]]))
    X = np.array([[5.0]])
    result = r.route(X)
    np.testing.assert_array_equal(result.top2_indices, np.array([[3, 2]]))
    logits = np.array([0.0, 5.0, 10.0, 15.0])
    probs = _reference_softmax(logits[None, :])
    selected = np.array([probs[0, 3], probs[0, 2]])
    np.testing.assert_allclose(
        result.top2_weights[0], selected / selected.sum(), rtol=1e-12
    )


def test_top2_weights_finite_nonneg_sum_to_one():
    """[ROBUST] Fails on wrong-axis renormalization, missing
    renormalization, or sign bugs."""
    rng = np.random.default_rng(123)
    N, d, E = 16, 5, 6
    X = rng.standard_normal((N, d))
    r = Top2MoERouter(d_model=d, num_experts=E, seed=7)
    result = r.route(X)
    w = result.top2_weights
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12)
    rows = np.arange(N)[:, None]
    raw = result.probs[rows, result.top2_indices]
    expected = raw / raw.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(w, expected, rtol=1e-12, atol=1e-12)


def test_top2_indices_are_distinct_per_row():
    """[ROBUST] Fails on sampling with replacement."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 4))
    r = Top2MoERouter(4, 7, seed=0)
    result = r.route(X)
    assert np.all(result.top2_indices[:, 0] != result.top2_indices[:, 1])


def test_loss_matches_hand_oracle_nonuniform():
    """[ROBUST] Fails on count/E (should be count/N), sum-instead-of-mean,
    missing E, or wrong-axis reduction. NON-uniform probs."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
        ]
    )
    top2_indices = np.array([[0, 1], [1, 2]])
    loss = compute_load_balancing_loss(probs, top2_indices, num_experts=3, alpha=0.01)
    # counts=[1,2,1]; f=[0.5,1.0,0.5]; P=[0.30,0.45,0.25]
    # sum(f*P)=0.15+0.45+0.125=0.725; L=0.01*3*0.725=0.02175
    assert loss == pytest.approx(0.02175, rel=1e-10)


def test_loss_invariants_and_decomposition():
    """[ROBUST] Fails if sum(f) != 2, sum(P) != 1, or the loss deviates
    from alpha*E*sum(f*P). NOTE: this recomputes f and P with the same
    numpy primitives (bincount, mean) the implementation uses; it is NOT
    a fully independent oracle. It catches the 2N-denominator and
    wrong-axis mutations because the *formula* in the test uses the
    correct denominators, not because the primitives differ."""
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.4, 0.2],
            [0.1, 0.1, 0.8],
        ]
    )
    top2_indices = np.array([[0, 1], [0, 2], [2, 0]])
    N, E = probs.shape
    counts = np.bincount(top2_indices.ravel(), minlength=E)
    f = counts / N  # sum(f) == 2
    P = probs.mean(axis=0)  # sum(P) == 1
    assert f.sum() == pytest.approx(2.0, rel=1e-12)
    assert P.sum() == pytest.approx(1.0, rel=1e-12)
    loss = compute_load_balancing_loss(probs, top2_indices, E, alpha=0.01)
    expected = 0.01 * E * float(np.sum(f * P))
    assert loss == pytest.approx(expected, rel=1e-12)


def test_loss_uses_both_top2_indices():
    """[ROBUST] Fails on top-1-only counting. Oracle includes both columns
    and asserts the top-1-only value differs."""
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    top2 = np.array([[0, 1], [1, 2]])
    loss = compute_load_balancing_loss(probs, top2, 3, alpha=1.0)
    assert loss == pytest.approx(2.1, rel=1e-10)  # f=[0.5,1,0.5], P=[0.4,0.4,0.2]
    assert loss != pytest.approx(1.2, rel=1e-6)  # top-1-only oracle


def test_loss_changes_when_routing_changes():
    """[ROBUST] Fails if loss ignores top2_indices (computes from probs
    alone). Same probs, different valid routings -> different loss."""
    probs = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    balanced = np.array([[0, 1], [1, 2]])
    skewed = np.array([[0, 1], [0, 1]])
    loss_bal = compute_load_balancing_loss(probs, balanced, 3, alpha=0.01)
    loss_skew = compute_load_balancing_loss(probs, skewed, 3, alpha=0.01)
    assert loss_bal != pytest.approx(loss_skew, rel=1e-6)


def test_loss_scaling_includes_num_experts():
    """[ROBUST] Fails if E multiplier omitted or replaced by k=2.
    E=4 hand-computed value 0.024; E-omitted would be 0.006."""
    probs = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.3, 0.2],
        ]
    )
    top2 = np.array([[0, 1], [1, 2]])
    loss = compute_load_balancing_loss(probs, top2, num_experts=4, alpha=0.01)
    assert loss == pytest.approx(0.024, rel=1e-10)
    assert loss != pytest.approx(0.006, rel=1e-6)


def test_seeded_init_reproducible():
    """[ROBUST] Fails if router ignores seed and reads a global RNG."""
    r1 = Top2MoERouter(4, 5, seed=42)
    r2 = Top2MoERouter(4, 5, seed=42)
    np.testing.assert_array_equal(r1.W_g, r2.W_g)


def test_different_seeds_change_weights():
    """[ROBUST] Fails if seed is ignored entirely."""
    r1 = Top2MoERouter(4, 5, seed=1)
    r2 = Top2MoERouter(4, 5, seed=2)
    assert not np.array_equal(r1.W_g, r2.W_g)


def test_route_deterministic_given_fixed_weights():
    """[ROBUST] Fails if route uses any randomness (unseeded tie-break).
    Two calls on the same router and input must be bit-identical."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    r = Top2MoERouter(3, 4, seed=5)
    a = r.route(X)
    b = r.route(X)
    np.testing.assert_array_equal(a.top2_indices, b.top2_indices)
    np.testing.assert_array_equal(a.top2_weights, b.top2_weights)
    np.testing.assert_array_equal(a.probs, b.probs)


def test_route_does_not_mutate_input():
    """[ROBUST] Fails on in-place softmax / mean-subtraction on the input.
    float32 input exercises the astype path."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((6, 3)).astype(np.float32)
    X_copy = X.copy()
    r = Top2MoERouter(3, 4, seed=0)
    r.route(X)
    np.testing.assert_array_equal(X, X_copy)


def test_loss_zero_when_alpha_zero():
    """[ROBUST] Fails if alpha=0 rejected or alpha ignored."""
    probs = np.full((4, 3), 1.0 / 3.0)
    idx = np.tile(np.array([0, 1]), (4, 1))
    loss = compute_load_balancing_loss(probs, idx, 3, alpha=0.0)
    assert loss == 0.0


def test_uniform_routing_gives_two_alpha():
    """[ROBUST] Fails if loss returns alpha instead of 2*alpha for Top-2.
    Uniform probs + balanced routing -> sum(f)=2, L=2*alpha."""
    N, E, alpha = 6, 3, 0.01
    probs = np.full((N, E), 1.0 / E)
    idx = np.array([[0, 1], [1, 2], [2, 0], [0, 1], [1, 2], [2, 0]])
    loss = compute_load_balancing_loss(probs, idx, E, alpha=alpha)
    assert loss == pytest.approx(2.0 * alpha, rel=1e-10)


def test_loss_accepts_float32_probs():
    """[ROBUST] Fails if the loss rejects float32 probs whose rows sum to
    1 within float32 precision (~1e-6) but not bit-exactly 1.0."""
    probs = np.full((4, 3), 1.0 / 3.0, dtype=np.float32)
    idx = np.tile(np.array([0, 1]), (4, 1))
    loss = compute_load_balancing_loss(probs, idx, 3, alpha=0.01)
    assert np.isfinite(loss)


# --------------------------------------------------------------------------- #
# [CONTRACT] tests
# --------------------------------------------------------------------------- #


def test_route_returns_full_probs_for_loss():
    """[CONTRACT] Fails if route drops full probs (documented change from
    starter 2-tuple)."""
    r = Top2MoERouter(3, 4, seed=0)
    result = r.route(np.zeros((2, 3)))
    assert isinstance(result, RoutingResult)
    assert result.probs.shape == (2, 4)
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, rtol=1e-12)


def test_top2_indices_dtype_is_integer():
    """[CONTRACT] Fails if top2_indices comes back float (breaks
    np.bincount downstream)."""
    r = Top2MoERouter(3, 4, seed=0)
    result = r.route(np.zeros((2, 3)))
    assert np.issubdtype(result.top2_indices.dtype, np.integer)


# --------------------------------------------------------------------------- #
# [GUARD] tests  (subcategory of [ROBUST] under the review rubric)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, 1, -1])
def test_rejects_num_experts_less_than_2(bad):
    """[GUARD] Fails if __init__ accepts num_experts < 2."""
    with pytest.raises(ValueError, match="num_experts"):
        Top2MoERouter(d_model=4, num_experts=bad, seed=0)


def test_rejects_bool_d_model():
    """[GUARD] Fails if __init__ accepts d_model=True (bool subclasses
    int; would silently become d_model=1)."""
    with pytest.raises(TypeError, match="bool"):
        Top2MoERouter(d_model=True, num_experts=4, seed=0)


def test_rejects_bool_num_experts():
    """[GUARD] Fails if __init__ accepts num_experts=True (bool
    subclasses int; would silently become num_experts=1, which also
    fails the >=2 check, but with a confusing message)."""
    with pytest.raises(TypeError, match="bool"):
        Top2MoERouter(d_model=4, num_experts=True, seed=0)


def test_route_rejects_corrupted_w_g_shape():
    """[GUARD] Fails if route() emits an opaque numpy error when the
    publicly mutable W_g has been replaced with something that is not a
    floating ndarray of the right shape. The router should fail loudly
    at the boundary with a clear message for each of the three
    corruption classes: wrong shape, non-ndarray, non-floating dtype."""
    r = Top2MoERouter(3, 4, seed=0)
    r.W_g = np.zeros((5, 4))  # wrong shape; expected (3, 4)
    with pytest.raises(RuntimeError, match="W_g shape"):
        r.route(np.zeros((2, 3)))

    r.W_g = [[0.0] * 4 for _ in range(3)]  # Python list, correct "shape"
    with pytest.raises(TypeError, match="numpy.ndarray"):
        r.route(np.zeros((2, 3)))

    r.W_g = np.zeros((3, 4), dtype=object)  # object dtype, correct shape
    with pytest.raises(RuntimeError, match="floating dtype"):
        r.route(np.zeros((2, 3)))


def test_rejects_wrong_input_ndim():
    """[GUARD] Fails if route accepts 1-D or 3-D X."""
    r = Top2MoERouter(d_model=3, num_experts=4, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        r.route(np.zeros(3))
    with pytest.raises(ValueError, match="2-D"):
        r.route(np.zeros((2, 3, 1)))


def test_rejects_wrong_feature_count():
    """[GUARD] Fails if route accepts X with d != d_model."""
    r = Top2MoERouter(d_model=3, num_experts=4, seed=0)
    with pytest.raises(ValueError, match="feature count"):
        r.route(np.zeros((2, 5)))


def test_rejects_nan_inf_input():
    """[GUARD] Fails if route accepts non-finite X."""
    r = Top2MoERouter(d_model=3, num_experts=4, seed=0)
    X = np.zeros((2, 3))
    X[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        r.route(X)
    X[0, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        r.route(X)


def test_rejects_empty_batch():
    """[GUARD] Fails if route silently accepts N=0."""
    r = Top2MoERouter(3, 4, seed=0)
    with pytest.raises(ValueError, match="at least one token"):
        r.route(np.zeros((0, 3)))


def test_loss_rejects_num_experts_below_2():
    """[GUARD] Fails if compute_load_balancing_loss accepts
    num_experts < 2. Separate from the __init__ check."""
    probs = np.full((2, 3), 1.0 / 3.0)
    idx = np.array([[0, 1], [1, 2]])
    for bad in (0, 1, -1):
        with pytest.raises(ValueError, match="num_experts"):
            compute_load_balancing_loss(probs, idx, bad)


def test_loss_rejects_num_experts_mismatch_with_probs():
    """[GUARD] Fails if loss accepts probs.shape[1] != num_experts."""
    probs = np.full((2, 3), 1.0 / 3.0)
    idx = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="num_experts"):
        compute_load_balancing_loss(probs, idx, 4)


def test_loss_rejects_negative_alpha():
    """[GUARD] Fails if loss accepts alpha < 0 (inverts the loss sign)."""
    probs = np.full((2, 3), 1.0 / 3.0)
    idx = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="alpha"):
        compute_load_balancing_loss(probs, idx, 3, alpha=-0.01)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_loss_rejects_non_finite_alpha(bad):
    """[GUARD] Fails if the alpha finiteness check is removed."""
    probs = np.full((2, 3), 1.0 / 3.0)
    idx = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="alpha"):
        compute_load_balancing_loss(probs, idx, 3, alpha=bad)


def test_loss_rejects_bool_alpha():
    """[GUARD] Fails if loss accepts alpha=True (bool subclasses int; a
    bool alpha is almost certainly a caller bug)."""
    probs = np.full((2, 3), 1.0 / 3.0)
    idx = np.array([[0, 1], [1, 2]])
    with pytest.raises(TypeError, match="bool"):
        compute_load_balancing_loss(probs, idx, 3, alpha=True)


def test_loss_rejects_duplicate_indices():
    """[GUARD] Fails if loss accepts same expert twice in one row."""
    probs = np.full((2, 3), 1.0 / 3.0)
    bad = np.array([[0, 0], [1, 2]])
    with pytest.raises(ValueError, match="DISTINCT"):
        compute_load_balancing_loss(probs, bad, 3)


def test_loss_rejects_out_of_range_indices():
    """[GUARD] Fails if loss accepts indices >= num_experts or < 0."""
    probs = np.full((2, 3), 1.0 / 3.0)
    with pytest.raises(ValueError, match="out of range"):
        compute_load_balancing_loss(probs, np.array([[0, 5], [1, 2]]), 3)
    with pytest.raises(ValueError, match="out of range"):
        compute_load_balancing_loss(probs, np.array([[0, -1], [1, 2]]), 3)


def test_loss_rejects_wrong_width_indices():
    """[GUARD] Fails if loss accepts top2_indices.shape[1] != 2."""
    probs = np.full((2, 3), 1.0 / 3.0)
    with pytest.raises(ValueError, match="N, 2"):
        compute_load_balancing_loss(probs, np.array([[0, 1, 2], [1, 2, 0]]), 3)
    with pytest.raises(ValueError, match="N, 2"):
        compute_load_balancing_loss(probs, np.array([0, 1]), 3)


def test_loss_rejects_non_unit_row_sums():
    """[GUARD] Fails if loss accepts probs whose rows do not sum to 1."""
    probs = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
    idx = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="sum to 1"):
        compute_load_balancing_loss(probs, idx, 3)


# --------------------------------------------------------------------------- #
# Property test (Hypothesis, optional)
# --------------------------------------------------------------------------- #
#
# PROPERTY (text form, active when hypothesis is absent):
#   For any N>=1, d>=1, E>=2, any finite X in R^{N x d}, any finite W_g in
#   R^{d x E}, route(X) satisfies:
#     (a) top2_indices.shape == (N, 2), values in [0, E), distinct per row;
#     (b) top2_weights.shape == (N, 2), finite, >= 0, rows sum to 1;
#     (c) probs.shape == (N, E), finite, >= 0, rows sum to 1;
#     (d) probs == softmax(X @ W_g) (max-subtraction form);
#     (e) top2_weights == probs[arange, top2_indices] / rowsum(top2 probs);
#     (f) loss(probs, top2_indices, E) is finite and >= 0.
#
# Note on scope: the executed body uses the initialized router weights
# (X @ r.W_g), not arbitrary generated W_g matrices. Assertions (d) and
# (e) below compare against a locally recomputed softmax and gather,
# which is independent of the implementation's softmax and gather code
# paths but uses the same numpy primitives. Extending the generator to
# arbitrary W_g is a future revision.


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
@given(
    N=st.integers(min_value=1, max_value=20),
    d=st.integers(min_value=1, max_value=8),
    E=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_property_route_invariants(N, d, E, seed):
    """[ROBUST] Property test — checks shapes, ranges, finiteness,
    normalization, that probs equals a locally recomputed softmax of
    X @ W_g, and that top2_weights equals a locally recomputed gather
    and renormalization of probs at top2_indices."""
    r = Top2MoERouter(d, E, seed=seed)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d)) * 100.0
    result = r.route(X)

    assert result.top2_indices.shape == (N, 2)
    assert result.top2_weights.shape == (N, 2)
    assert result.probs.shape == (N, E)

    assert np.all(result.top2_indices >= 0)
    assert np.all(result.top2_indices < E)
    assert np.all(result.top2_indices[:, 0] != result.top2_indices[:, 1])

    assert np.all(np.isfinite(result.top2_weights))
    assert np.all(result.top2_weights >= 0)
    np.testing.assert_allclose(result.top2_weights.sum(axis=1), 1.0, rtol=1e-10)

    assert np.all(np.isfinite(result.probs))
    assert np.all(result.probs >= 0)
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, rtol=1e-10)

    # (d) probs == softmax(X @ W_g), recomputed locally with max-subtraction
    logits = X @ r.W_g
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.probs, expected_probs, rtol=1e-12, atol=1e-12)

    # (e) top2_weights == probs[arange, top2_indices] renormalized
    selected = result.probs[np.arange(N)[:, None], result.top2_indices]
    expected_weights = selected / selected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.top2_weights, expected_weights, rtol=1e-12)

    loss = compute_load_balancing_loss(result.probs, result.top2_indices, E)
    assert np.isfinite(loss)
    assert loss >= 0.0


# --------------------------------------------------------------------------- #
# Not tested (and why):
# - Sparse expert MLP forward pass and output combination:
#   Out of Day-53 scope; documented in the module docstring. Would need its
#   own spec and unselected-expert non-contribution tests.
# - Backward pass / gradient of L_aux through softmax:
#   Not implemented. Forward-only. No finite-difference gradient check.
# - Capacity factor, token dropping:
#   Separate scheduling feature; would change the loss contract.
# - Distributed all-to-all dispatch:
#   Not applicable to a single-process NumPy reference.
# - Mixed-precision (float32<->float64) numerical equivalence:
#   We compute in float64; loss validates with atol=1e-5. Bit-equivalence
#   across precisions would be numerically false and is not asserted.
# - Concurrency / thread safety of route():
#   route() is pure w.r.t. self.W_g (read-only) and allocates all outputs
#   locally. No shared mutable state to race on.
# - NaN/inf in W_g if the caller corrupts it:
#   The logit finiteness guard raises RuntimeError; the same branch is
#   exercised by test_route_raises_runtime_error_on_logit_overflow via
#   overflow. A dedicated NaN-in-W_g test would hit the same code path.
# - Non-finite alpha combined with alpha >= 0: covered by the
#   parametrized nan/inf/-inf test. alpha=np.float32(0.5) goes through
#   the float() cast; no test distinguishes the cast from a direct use.
# - Array aliasing between result.probs and result.top2_weights:
#   The previous test_route_returns_independent_arrays was deleted
#   (adversarial self-verification Step 2 found it caught no mutation not
#   already caught by test_top2_weights_are_selected_probs_renormalized;
#   take_along_axis and / always allocate by construction).
# - python -O survival of the assert->raise conversion:
#   Asserted in the docstring but not verified by a test that runs the
#   module under -O. The prior revision's asserts are now raises; the
#   test suite does not confirm the raise still fires under -O.
# - Benchmark of argsort vs argpartition:
#   Not measured. The O(N*E*log E) claim is code-derived, not benchmarked.
# - Property test with arbitrary generated W_g:
#   The current test uses initialized router weights (X @ r.W_g). Its
#   docstring was narrowed to say so. Extending the generator to
#   arbitrary finite W_g is a future revision, not a current claim.
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
