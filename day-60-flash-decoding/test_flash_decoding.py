"""Day 60 — Test suite for FlashDecoding.

Categories:
  [ROBUST]   would fail against a specific wrong implementation
  [CONTRACT] would fail only if the public interface changed

Determinism claim tested here is same-environment repeatability only.
Cross-BLAS bitwise reproducibility is not claimed by the implementation
and is not tested.
"""

import numpy as np
import pytest
from day60_flash_decoding import _chunk_bounds, flash_decode_forward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_full_attention(Q, K, V, sm_scale):
    """Stable full-sequence attention oracle, computed in float64."""
    Q64 = np.asarray(Q, dtype=np.float64)
    K64 = np.asarray(K, dtype=np.float64)
    V64 = np.asarray(V, dtype=np.float64)
    scores = (Q64 @ K64.T) * sm_scale
    M = scores.max(axis=1, keepdims=True)
    P = np.exp(scores - M)
    L = P.sum(axis=1, keepdims=True)
    return (P / L) @ V64


def _rand_arrays(rng, N, d, d_v, dtype=np.float32):
    Q = rng.standard_normal((1, d)).astype(dtype)
    K = rng.standard_normal((N, d)).astype(dtype)
    V = rng.standard_normal((N, d_v)).astype(dtype)
    return Q, K, V


# ---------------------------------------------------------------------------
# [ROBUST] — each names a specific wrong implementation
# ---------------------------------------------------------------------------


def test_single_chunk_matches_full_attention():
    """[ROBUST] Fails if num_chunks=1 does not reduce to standard attention."""
    rng = np.random.default_rng(1)
    Q, K, V = _rand_arrays(rng, N=16, d=8, d_v=8)
    scale = 1.0 / np.sqrt(8)
    out = flash_decode_forward(Q, K, V, num_chunks=1, sm_scale=scale)
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_chunks", [2, 3, 4, 8, 16])
def test_multi_chunk_matches_full_attention(num_chunks):
    """[ROBUST] Fails if chunking perturbs the result for any partition."""
    rng = np.random.default_rng(100 + num_chunks)
    Q, K, V = _rand_arrays(rng, N=16, d=8, d_v=6)
    scale = 1.0 / np.sqrt(8)
    out = flash_decode_forward(Q, K, V, num_chunks=num_chunks, sm_scale=scale)
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("N,num_chunks", [(10, 3), (17, 5), (23, 4)])
def test_uneven_partition_matches_full_attention(N, num_chunks):
    """[ROBUST] Fails if remainder rows are dropped, duplicated, or
    associated with the wrong value rows when N is not divisible by C."""
    rng = np.random.default_rng(N + num_chunks)
    Q, K, V = _rand_arrays(rng, N=N, d=4, d_v=4)
    scale = 1.0 / np.sqrt(4)
    out = flash_decode_forward(Q, K, V, num_chunks=num_chunks, sm_scale=scale)
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_chunk_count_invariance():
    """[ROBUST] Fails if the output depends on num_chunks beyond FP noise."""
    rng = np.random.default_rng(42)
    Q, K, V = _rand_arrays(rng, N=32, d=8, d_v=8)
    scale = 1.0 / np.sqrt(8)
    outs = [
        flash_decode_forward(Q, K, V, num_chunks=c, sm_scale=scale)
        for c in (1, 2, 4, 8, 16, 32)
    ]
    for o in outs[1:]:
        np.testing.assert_allclose(o, outs[0], rtol=1e-5, atol=1e-5)


def test_chunk_partition_covers_every_row_exactly_once():
    """[ROBUST] Fails if the partition drops, duplicates, or reorders rows."""
    for N in (1, 2, 5, 10, 17, 100):
        for C in range(1, min(N, 10) + 1):
            bounds = _chunk_bounds(N, C)
            assert len(bounds) == C, f"N={N}, C={C}"
            covered = []
            for start, end in bounds:
                assert 0 <= start < end <= N, f"bad bounds {(start, end)}"
                covered.extend(range(start, end))
            assert covered == list(range(N)), f"N={N}, C={C}"


def test_chunk_bounds_match_array_split_semantics():
    """[ROBUST] Fails if the partition diverges from the documented
    numpy.array_split remainder policy."""
    for N in (1, 2, 5, 10, 17, 100):
        for C in range(1, min(N, 10) + 1):
            bounds = _chunk_bounds(N, C)
            ref = np.array_split(np.arange(N), C)
            for (start, end), ref_idx in zip(bounds, ref):
                np.testing.assert_array_equal(np.arange(start, end), ref_idx)


def test_hand_computed_merge():
    """[ROBUST] Fails if local outputs are weighted incorrectly during merge.

    d=2, N=4, C=2, scale=1:
      Q = [[1,0]]
      K = [[1,0],[0,1],[1,0],[0,1]]
      V = [[1,0],[0,1],[2,0],[0,2]]
    Per chunk: m_c=1, l_c=1+e^-1.
      chunk1: o_1=[1/(1+e^-1), e^-1/(1+e^-1)]
      chunk2: o_2=[2/(1+e^-1), 2e^-1/(1+e^-1)]
    M=1, alpha=1, L=2(1+e^-1), w_global=[0.5, 0.5]
    Expected O = [3, 3e^-1] / (2(1+e^-1)).

    Caveat: because both chunks have identical l_c, this test CANNOT by
    itself detect a "missing l_c" mutation — dropping l_c would produce
    equal weights here. That specific mutation is caught by
    test_alpha_applied_exactly_once_to_both_terms, whose chunks have
    l_1 = 2 and l_2 = 1 + e^-5. This test is retained because it is an
    independent numerical check of the merge arithmetic at a different
    operating point (equal local maxima).
    """
    Q = np.array([[1.0, 0.0]], dtype=np.float64)
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    V = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    out = flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=1.0)
    e_inv = np.exp(-1.0)
    denom = 2.0 * (1.0 + e_inv)
    expected = np.array([[3.0 / denom, 3.0 * e_inv / denom]])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_alpha_applied_exactly_once_to_both_terms():
    """[ROBUST] Fails if alpha_c = exp(m_c - M) is applied twice, zero
    times, with the wrong sign, or to only one of {l_c, u_c}.

    Case: Q=[[1]], K=[[0],[0],[0],[5]], V=[[1],[1],[1],[2]], C=2, scale=1.
      chunk1: s=[0,0], m_1=0, l_1=2, o_1=1
      chunk2: s=[0,5], m_2=5, l_2=e^-5+1, o_2=(e^-5+2)/(e^-5+1)
      M=5, alpha=[e^-5, 1]
      L=3e^-5+1, global_weights=[2e^-5/L, (e^-5+1)/L]
      O = (3e^-5+2)/(3e^-5+1)
    The two chunks have DIFFERENT l_c, so dropping l_c from the merge
    weight changes the answer by ~2x (this is the mutation detected here
    that test_hand_computed_merge cannot detect).
    """
    Q = np.array([[1.0]], dtype=np.float64)
    K = np.array([[0.0], [0.0], [0.0], [5.0]], dtype=np.float64)
    V = np.array([[1.0], [1.0], [1.0], [2.0]], dtype=np.float64)
    out = flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=1.0)
    e5 = np.exp(-5.0)
    expected = (3.0 * e5 + 2.0) / (3.0 * e5 + 1.0)
    np.testing.assert_allclose(out, [[expected]], rtol=1e-12, atol=1e-12)


def test_cross_chunk_extreme_magnitude_separation():
    """[ROBUST] Fails if the global rescaling reference M is chosen such
    that alpha_c = exp(m_c - M) overflows float64 for some chunk c. This
    happens when M < max_c(m_c) - ~709 roughly.

    Q=[[1]], K=[[1000],[-1000]], V=[[1],[2]], C=2, scale=1.
    Stable merge: O ≈ 1.0 (second chunk underflows to zero weight).
    Mutation M=min(m_c): alpha_1 = exp(2000) overflows float64 (max log ~709);
    L=inf, O=NaN. This is the ONLY class of mutation this test catches:
    because the merge is shift-invariant (any M' = M + delta yields the
    same O after the delta cancels in U/L), M=min/mean/sum on inputs whose
    chunk-max range is < ~709 produce the CORRECT output and would NOT be
    caught. The test is specifically an overflow-vulnerability probe, not
    a general "any non-max M is wrong" probe.
    """
    Q = np.array([[1.0]], dtype=np.float64)
    K = np.array([[1000.0], [-1000.0]], dtype=np.float64)
    V = np.array([[1.0], [2.0]], dtype=np.float64)
    out = flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=1.0)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [[1.0]], rtol=1e-12, atol=1e-12)


def test_large_logits_stay_finite_and_correct():
    """[ROBUST] Fails if exp(raw_scores) is computed without max subtraction
    (float32 overflows at ~89, float64 at ~710), or if a local max is
    chosen that is lower than the true chunk max by more than ~89."""
    N, d, d_v = 8, 4, 4
    Q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    K = np.zeros((N, d), dtype=np.float32)
    K[:, 0] = np.array([100.0, 100.5, 101.0, 100.2, 100.8, 99.5, 100.1, 100.9])
    V = np.arange(N, dtype=np.float32)[:, None] * np.ones((1, d_v), dtype=np.float32)
    scale = 1.0
    out = flash_decode_forward(Q, K, V, num_chunks=4, sm_scale=scale)
    assert np.isfinite(out).all()
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_extreme_finite_scores_raise_clear_error():
    """[ROBUST] Fails if Q @ K.T overflows to inf/nan and the overflow is
    allowed to propagate through exp and produce a cryptic downstream error.

    Finite inputs 1e308 * 1e308 overflow float64 immediately. The
    implementation must detect this at the score stage and raise
    FloatingPointError with a message that names the failing stage.
    """
    Q = np.array([[1e308]], dtype=np.float64)
    K = np.array([[1e308], [1e308]], dtype=np.float64)
    V = np.array([[1.0], [2.0]], dtype=np.float64)
    with pytest.raises(FloatingPointError, match=r"score overflow"):
        flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=1.0)


def test_large_values_with_finite_expected_output():
    """[ROBUST] Fails if the local numerator e @ V_c overflows when the
    mathematically correct normalized output is representable.

    Q=[[0]], K=[[0],[0]], V=[[1e308],[1e308]], C=1, scale=1.
    Both scores zero -> P=[0.5, 0.5] -> O=1e308, representable.
    Naive unnormalized numerator 1e308 + 1e308 = 2e308 overflows.
    Correct per-chunk normalization w_c @ V_c stays at 1e308.
    """
    Q = np.array([[0.0]], dtype=np.float64)
    K = np.array([[0.0], [0.0]], dtype=np.float64)
    V = np.array([[1e308], [1e308]], dtype=np.float64)
    out = flash_decode_forward(Q, K, V, num_chunks=1, sm_scale=1.0)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [[1e308]], rtol=1e-12)


def test_default_scale_equals_inverse_sqrt_d():
    """[ROBUST] Fails if the default sm_scale is not 1/sqrt(d)."""
    rng = np.random.default_rng(3)
    Q, K, V = _rand_arrays(rng, N=8, d=4, d_v=4)
    out_default = flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=None)
    out_explicit = flash_decode_forward(
        Q, K, V, num_chunks=2, sm_scale=1.0 / np.sqrt(4)
    )
    np.testing.assert_allclose(out_default, out_explicit, rtol=1e-6, atol=1e-6)


def test_custom_scale_honored():
    """[ROBUST] Fails if a supplied sm_scale is ignored or overridden."""
    rng = np.random.default_rng(4)
    Q, K, V = _rand_arrays(rng, N=8, d=4, d_v=4)
    scale = 0.37
    out = flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=scale)
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_deterministic_repeated_calls():
    """[ROBUST] Fails if the function has hidden state or randomness within
    a single execution environment.

    Scope: same-environment repeatability only. Cross-BLAS bitwise
    reproducibility is not claimed by the implementation and is not tested
    here (BLAS reduction order can differ across backends).
    """
    Q, K, V = _rand_arrays(np.random.default_rng(13), N=16, d=8, d_v=8)
    outs = [flash_decode_forward(Q, K, V, num_chunks=4) for _ in range(5)]
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


def test_permutation_invariance():
    """[ROBUST] Fails if output depends on the order of KV rows, i.e. if a
    wrong implementation slices K and V inconsistently, or if the merge
    implicitly weights by position rather than by content."""
    rng = np.random.default_rng(17)
    Q, K, V = _rand_arrays(rng, N=16, d=4, d_v=4)
    out_orig = flash_decode_forward(Q, K, V, num_chunks=4)
    perm = rng.permutation(16)
    out_perm = flash_decode_forward(Q, K[perm], V[perm], num_chunks=4)
    np.testing.assert_allclose(out_orig, out_perm, rtol=1e-5, atol=1e-5)


def test_single_element_sequence():
    """[ROBUST] Fails if N=1 is mishandled (softmax over 1 element = 1)."""
    Q = np.array([[1.0, 0.0]], dtype=np.float32)
    K = np.array([[2.0, 0.0]], dtype=np.float32)
    V = np.array([[3.0, 4.0]], dtype=np.float32)
    out = flash_decode_forward(Q, K, V, num_chunks=1)
    np.testing.assert_allclose(out, V, rtol=1e-6, atol=1e-6)


def test_one_chunk_per_sequence_element():
    """[ROBUST] Fails if num_chunks == N is mishandled."""
    rng = np.random.default_rng(14)
    Q, K, V = _rand_arrays(rng, N=8, d=4, d_v=4)
    scale = 1.0 / np.sqrt(4)
    out = flash_decode_forward(Q, K, V, num_chunks=8, sm_scale=scale)
    ref = _stable_full_attention(Q, K, V, scale)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_numpy_integer_num_chunks_accepted():
    """[ROBUST] Fails if numpy integer chunk counts are rejected (documented
    accepted runtime type)."""
    rng = np.random.default_rng(15)
    Q, K, V = _rand_arrays(rng, N=8, d=4, d_v=4)
    out_int = flash_decode_forward(Q, K, V, num_chunks=2)
    out_np = flash_decode_forward(Q, K, V, num_chunks=np.int64(2))
    np.testing.assert_array_equal(out_int, out_np)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_reject_invalid_num_chunks(bad):
    """[ROBUST] Fails if non-positive / bool / float num_chunks is accepted."""
    Q, K, V = _rand_arrays(np.random.default_rng(0), N=4, d=2, d_v=2)
    with pytest.raises((TypeError, ValueError)):
        flash_decode_forward(Q, K, V, num_chunks=bad)


def test_reject_num_chunks_exceeding_n():
    """[ROBUST] Fails if num_chunks > N is accepted (would create empty chunks)."""
    Q, K, V = _rand_arrays(np.random.default_rng(0), N=3, d=2, d_v=2)
    with pytest.raises(ValueError):
        flash_decode_forward(Q, K, V, num_chunks=4)


def test_reject_empty_sequence():
    """[ROBUST] Fails if N=0 is accepted."""
    Q = np.zeros((1, 4), dtype=np.float32)
    K = np.zeros((0, 4), dtype=np.float32)
    V = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        flash_decode_forward(Q, K, V, num_chunks=1)


@pytest.mark.parametrize(
    "q_shape,k_shape,v_shape",
    [
        ((1, 4), (8, 5), (8, 4)),  # K feature mismatch
        ((1, 4), (8, 4), (7, 4)),  # K/V seq mismatch
        ((2, 4), (8, 4), (8, 4)),  # Q not (1, d)
    ],
)
def test_reject_bad_shapes(q_shape, k_shape, v_shape):
    """[ROBUST] Fails if a shape mismatch is accepted."""
    Q = np.zeros(q_shape, dtype=np.float32)
    K = np.zeros(k_shape, dtype=np.float32)
    V = np.zeros(v_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        flash_decode_forward(Q, K, V, num_chunks=2)


@pytest.mark.parametrize("bad_dtype", [np.int32, np.float16, object])
def test_reject_bad_dtypes(bad_dtype):
    """[ROBUST] Fails if int, float16, or object dtypes are accepted
    (float16 accumulation can overflow before max-subtraction and is
    outside the documented contract)."""
    Q = np.ones((1, 2), dtype=bad_dtype)
    K = np.ones((4, 2), dtype=bad_dtype)
    V = np.ones((4, 2), dtype=bad_dtype)
    with pytest.raises(TypeError):
        flash_decode_forward(Q, K, V, num_chunks=2)


def test_reject_mixed_dtypes():
    """[ROBUST] Fails if Q/K/V dtype mixing is accepted."""
    Q = np.ones((1, 2), dtype=np.float32)
    K = np.ones((4, 2), dtype=np.float64)
    V = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        flash_decode_forward(Q, K, V, num_chunks=2)


@pytest.mark.parametrize("which", ["Q", "K", "V"])
@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_reject_nan_or_inf(which, bad_val):
    """[ROBUST] Fails if NaN/inf is accepted in any of Q, K, V. Each tensor
    is poisoned independently, so a finiteness check that validates only
    one of the three is caught."""
    Q, K, V = _rand_arrays(np.random.default_rng(9), N=4, d=2, d_v=2)
    target = {"Q": Q, "K": K, "V": V}[which]
    target[0, 0] = bad_val
    with pytest.raises(ValueError, match=which):
        flash_decode_forward(Q, K, V, num_chunks=2)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.nan, np.inf])
def test_reject_invalid_sm_scale(bad_scale):
    """[ROBUST] Fails if sm_scale is non-positive or non-finite."""
    Q, K, V = _rand_arrays(np.random.default_rng(0), N=4, d=2, d_v=2)
    with pytest.raises(ValueError):
        flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=bad_scale)


def test_reject_scale_that_cannot_be_converted_to_float():
    """[ROBUST] Fails if a Python int too large for float() raises
    OverflowError instead of the documented ValueError."""
    Q = np.ones((1, 2), dtype=np.float64)
    K = np.ones((2, 2), dtype=np.float64)
    V = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        flash_decode_forward(Q, K, V, num_chunks=2, sm_scale=10**10000)


def test_inputs_not_mutated():
    """[ROBUST] Fails if the function mutates caller-owned arrays.

    Categorized as ROBUST (not CONTRACT) because it catches a specific
    wrong implementation that writes back into Q/K/V; no interface change
    is involved.
    """
    Q, K, V = _rand_arrays(np.random.default_rng(12), N=8, d=4, d_v=4)
    Q0, K0, V0 = Q.copy(), K.copy(), V.copy()
    flash_decode_forward(Q, K, V, num_chunks=2)
    np.testing.assert_array_equal(Q, Q0)
    np.testing.assert_array_equal(K, K0)
    np.testing.assert_array_equal(V, V0)


# ---------------------------------------------------------------------------
# [CONTRACT] — public interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_shape_dtype_and_finiteness(dtype):
    """[CONTRACT] Fails if output shape, dtype, or finiteness drifts."""
    Q, K, V = _rand_arrays(np.random.default_rng(11), N=12, d=6, d_v=5, dtype=dtype)
    out = flash_decode_forward(Q, K, V, num_chunks=3)
    assert out.shape == (1, 5)
    assert out.dtype == dtype
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# PROPERTY-TEST STRATEGY (documentation only — NOT executable)
# ---------------------------------------------------------------------------
#
# These are strategy specifications, not executable tests. They do not
# contribute to any pass count or coverage claim. They are retained here
# because the Day 60 spec requires at least one Hypothesis-style property
# strategy to be documented even when the dependency is unavailable.
#
# PROPERTY 1 — chunked output equals stable full attention.
#
#   @given(
#       N=st.integers(min_value=1, max_value=64),
#       C=st.integers(min_value=1, max_value=64),
#       d=st.integers(min_value=1, max_value=16),
#       d_v=st.integers(min_value=1, max_value=16),
#       scale=st.floats(min_value=1e-3, max_value=10.0,
#                       allow_nan=False, allow_infinity=False),
#       data=st.data(),
#   )
#   def test_property_chunked_matches_full(N, C, d, d_v, scale, data):
#       assume(C <= N)
#       Q = data.draw(finite_float_array((1, d)))
#       K = data.draw(finite_float_array((N, d)))
#       V = data.draw(finite_float_array((N, d_v)))
#       out = flash_decode_forward(Q, K, V, num_chunks=C, sm_scale=scale)
#       ref = _stable_full_attention(Q, K, V, scale)
#       assert np.allclose(out, ref, rtol=1e-4, atol=1e-4)
#
# PROPERTY 2 — chunk partition covers [0, N) exactly once.
#
#   @given(N=st.integers(1, 256), C=st.integers(1, 256))
#   def test_property_partition(N, C):
#       assume(C <= N)
#       bounds = _chunk_bounds(N, C)
#       covered = []
#       for s, e in bounds:
#           covered.extend(range(s, e))
#       assert covered == list(range(N))
#
# STRATEGY (finite_float_array): floats with allow_nan=False,
# allow_infinity=False, width=32, bounded by ±1e3; shapes bounded as above.
# Bound magnitudes so exp() stays representable after max-subtraction.

# Not tested (and why):
# - GPU parallelism / wall-clock speedup: sequential NumPy simulator;
#   no real concurrency is exercised, so any speedup claim would be
#   fabricated.
# - Batched / multi-head / MQA / GQA inputs: out of scope for the starter
#   API; enforced by test_reject_bad_shapes.
# - float16 / bfloat16: rejected by dtype contract; float16 accumulation
#   in Q@K.T can overflow before max-subtraction, and supporting it would
#   need a separate numerical-stability analysis.
# - Cross-machine bitwise reproducibility: BLAS reduction order differs
#   between backends; only tolerance equivalence is claimed by the module
#   docstring.
# - Performance at N > 10^5: sequential loop is too slow to exercise here;
#   complexity is argued analytically in the module docstring.
# - Hypothesis property tests: strategy specifications only (see the
#   PROPERTY-TEST STRATEGY block above); they are NOT executable tests and
#   do not contribute to the pass count. The deterministic parametrized
#   tests are the executable stand-in.
# - Behavioral verification of the alternative "filter empty chunks when
#   C > N" policy: the chosen policy is rejection, so the alternative is
#   not exercised.
# - Cross-magnitude sweep across the overflow boundary (700 < range < 800):
#   the only overflow test uses range 2000, which is well past the ~709
#   threshold. A finer sweep would give earlier detection of borderline
#   M choices, but is not exercised here.
# - Numerical behavior under non-IEEE-754 BLAS backends: not exercised.
# - Exhaustive extreme-magnitude V combinations: only the specific case
#   1e308 + 1e308 is exercised by test_large_values_with_finite_expected_output.
#   General float64 accumulation behavior for arbitrary extreme-magnitude
#   values is not proven, only the specific case is demonstrated.
# - Cross-BLAS bitwise determinism: only same-environment repeatability is
#   tested by test_deterministic_repeated_calls.
# - Chunk-level local-max invariant (m_c == max(s_c)): not directly
#   observable through the public API. If m_c were chosen too low by less
#   than ~89 (float32) / ~709 (float64), the arithmetic shift would cancel
#   and the output would still be correct; only overflows are caught by
#   test_large_logits_stay_finite_and_correct. A targeted test would
#   require exposing local summaries, which the public API does not.
