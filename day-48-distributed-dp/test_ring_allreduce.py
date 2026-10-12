"""
Test suite for `day48_ring_allreduce.py`.

Tags:
    [ROBUST]   fails against a specific wrong implementation
    [CONTRACT] fails only if the public interface changes
    [GUARD]    fails if a defensive validation is removed

18 [ROBUST], 2 [CONTRACT], 6 [GUARD]. Every test's docstring names the
wrong implementation it catches.

Scope: these tests validate the returned mean and the input contract.
They do NOT assert the round-by-round ring schedule; a schedule bug
that still produces the correct mean for the tested (N, M) pairs would
not be caught here. See the "Not tested" block at the bottom.
"""

import numpy as np
import pytest
from day48_ring_allreduce import DistributedDataParallelEngine, RingAllReduceWorker

# ---------------------------------------------------------------------------
# [ROBUST] — arithmetic correctness
# ---------------------------------------------------------------------------


def test_n4_divisible_matches_hand_computed_mean():
    """
    [ROBUST] Mean of [[1,2,3],[3,4,5],[5,6,7],[7,8,9]] is [4,5,6].
    Catches sum-instead-of-mean, double division, and dropped or
    duplicated workers.
    """
    out = DistributedDataParallelEngine(4).ring_allreduce(
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 4.0, 5.0]),
            np.array([5.0, 6.0, 7.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
    )
    np.testing.assert_allclose(out, [4.0, 5.0, 6.0], atol=1e-12)


def test_n1_returns_input_unchanged():
    """[ROBUST] Catches a broken single-worker case (no ring rounds)."""
    g = np.array([1.5, -2.25, 0.0, 7.0])
    out = DistributedDataParallelEngine(1).ring_allreduce([g])
    np.testing.assert_array_equal(out, g)


def test_n2_hand_computed():
    """[ROBUST] Mean of [1,3] and [5,7] is [3,5]. Smallest non-trivial ring."""
    out = DistributedDataParallelEngine(2).ring_allreduce(
        [np.array([1.0, 3.0]), np.array([5.0, 7.0])]
    )
    np.testing.assert_allclose(out, [3.0, 5.0], atol=1e-12)


def test_uneven_length_n3_preserves_positional_correspondence():
    """
    [ROBUST] M=5, N=3 -> chunks [2,2,1]. Catches dropped/duplicated
    tail elements (shape changes) and reversed chunk indices (values
    permute). Worker i holds 10*i + j at position j, mean = 10 + j.
    """
    out = DistributedDataParallelEngine(3).ring_allreduce(
        [
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        ]
    )
    assert out.shape == (5,), "tail element dropped or duplicated"
    np.testing.assert_allclose(out, [10.0, 11.0, 12.0, 13.0, 14.0], atol=1e-12)


def test_uneven_length_n5_m7():
    """
    [ROBUST] M=7, N=5 -> chunks [2,2,1,1,1]. Constant-valued inputs
    detect boundary bugs that change the mean or shape, but not
    arbitrary reordering (all chunks carry the same value); the
    varying-input test above handles reordering.
    """
    out = DistributedDataParallelEngine(5).ring_allreduce(
        [np.full(7, float(i)) for i in range(5)]
    )
    np.testing.assert_allclose(out, np.full(7, 2.0), atol=1e-12)


def test_vector_smaller_than_world_size():
    """[ROBUST] M=2 < N=4, so two chunks are empty. Catches M<N rejection."""
    out = DistributedDataParallelEngine(4).ring_allreduce(
        [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([7.0, 8.0]),
        ]
    )
    np.testing.assert_allclose(out, [4.0, 5.0], atol=1e-12)


def test_empty_vector_supported():
    """[ROBUST] Catches M=0 rejection and empty concatenate."""
    out = DistributedDataParallelEngine(3).ring_allreduce(
        [np.zeros(0), np.zeros(0), np.zeros(0)]
    )
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_negative_and_fractional_gradients():
    """[ROBUST] Catches sign errors and integer truncation."""
    grads = [
        np.array([-1.5, 0.25, -0.75]),
        np.array([2.5, 0.75, -0.25]),
        np.array([-0.5, -0.5, 1.75]),
    ]
    expected = np.mean(np.stack(grads, axis=0), axis=0)
    out = DistributedDataParallelEngine(3).ring_allreduce(grads)
    np.testing.assert_allclose(out, expected, atol=1e-15, rtol=0)


def test_all_equal_inputs_return_that_value():
    """
    [ROBUST] All workers give g -> mean is g. Catches a dropped worker
    (result (N-1)/N * g) or a duplicated one ((N+1)/N * g).
    """
    g = np.array([1.1, -2.2, 3.3, 4.4, -5.5])
    out = DistributedDataParallelEngine(4).ring_allreduce([g.copy() for _ in range(4)])
    np.testing.assert_allclose(out, g, atol=1e-14)


def test_repeated_calls_do_not_reuse_stale_state():
    """
    [ROBUST] Catches worker buffers that accumulate across calls: call
    2's output must reflect only call 2's input.
    """
    engine = DistributedDataParallelEngine(3)
    engine.ring_allreduce([np.full(4, 100.0) for _ in range(3)])
    out = engine.ring_allreduce(
        [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([3.0, 3.0, 3.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]
    )
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0], atol=1e-15, rtol=0)


def test_valid_call_after_validation_failure():
    """
    [ROBUST] A valid call works after a validation failure. Note: the
    failure here happens before worker buffers are touched, so this
    does not test recovery from a mid-reduction failure.
    """
    engine = DistributedDataParallelEngine(3)
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.zeros(4), np.zeros(4)])  # too few
    out = engine.ring_allreduce(
        [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
        ]
    )
    np.testing.assert_allclose(out, [2.0, 2.0], atol=1e-15, rtol=0)


def test_caller_owned_inputs_not_mutated():
    """[ROBUST] Catches aliasing of caller memory into the ring's buffers."""
    g0 = np.array([1.0, 2.0, 3.0, 4.0])
    g1 = np.array([5.0, 6.0, 7.0, 8.0])
    g2 = np.array([9.0, 10.0, 11.0, 12.0])
    snapshots = [g.copy() for g in (g0, g1, g2)]
    DistributedDataParallelEngine(3).ring_allreduce([g0, g1, g2])
    for g, snap in zip((g0, g1, g2), snapshots):
        np.testing.assert_array_equal(g, snap)


def test_result_dtype_and_shape_match_contract():
    """[ROBUST] Catches wrong output dtype or a scalar/stacked return."""
    out = DistributedDataParallelEngine(3).ring_allreduce(
        [np.arange(5, dtype=np.float32) for _ in range(3)]
    )
    assert out.dtype == np.float64
    assert out.shape == (5,)


def test_float64_accumulation_is_exact_for_large_magnitudes():
    """
    [ROBUST] Inputs [1e8, 1e8, 1e8, 1e8+4] have mean exactly 1e8+1.
    In float32 the ULP at 4e8 is 32, so the +4 would be rounded away.
    Catches a silent regression to float32 accumulation.
    """
    out = DistributedDataParallelEngine(4).ring_allreduce(
        [
            np.array([1e8]),
            np.array([1e8]),
            np.array([1e8]),
            np.array([1e8 + 4.0]),
        ]
    )
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.array([1e8 + 1.0]))


def test_intermediate_overflow_raises_controlled_error():
    """
    [ROBUST] Inputs [1e308, 1e308, -1e308] have a representable mean
    but overflow in this schedule's addition order. Catches a silent
    return of inf/NaN; the contract is to raise FloatingPointError.
    """
    with pytest.raises(FloatingPointError):
        DistributedDataParallelEngine(3).ring_allreduce(
            [
                np.array([1e308]),
                np.array([1e308]),
                np.array([-1e308]),
            ]
        )


def test_mean_matches_numpy_reference_across_nd():
    """
    [ROBUST] Sweeps (N, M) to catch chunk-boundary bugs that only show
    up for particular combinations. Fixed seed for reproducibility.
    """
    rng = np.random.RandomState(99)
    for n in (1, 2, 3, 4, 5, 7):
        for m in (1, 2, 3, 8, 13):
            grads = [rng.standard_normal(m) for _ in range(n)]
            expected = np.mean(np.stack(grads, axis=0), axis=0)
            out = DistributedDataParallelEngine(n).ring_allreduce(grads)
            np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-12)


def test_shift_invariance():
    """
    [ROBUST] Adding c to every worker's input shifts the mean by c.
    Catches a wrong divisor: the offset would scale with N instead.
    """
    rng = np.random.RandomState(11)
    grads = [rng.standard_normal(5) for _ in range(3)]
    c = rng.standard_normal(5)
    base = DistributedDataParallelEngine(3).ring_allreduce(grads)
    shifted = DistributedDataParallelEngine(3).ring_allreduce([g + c for g in grads])
    np.testing.assert_allclose(shifted, base + c, atol=1e-12)


def test_accepts_generator_of_gradients():
    """
    [ROBUST] Catches implementations that call len() on the input or
    otherwise reject a one-shot iterable. The contract is "iterable".
    """
    out = DistributedDataParallelEngine(3).ring_allreduce(
        np.full(2, float(i)) for i in range(3)
    )
    np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
# [CONTRACT] — public interface
# ---------------------------------------------------------------------------


def test_worker_set_local_gradient_is_a_defensive_copy():
    """[CONTRACT] Catches a worker that aliases caller memory."""
    src = np.array([1.0, 2.0, 3.0])
    w = RingAllReduceWorker(rank=0, world_size=1)
    w.set_local_gradient(src)
    src[0] = 99.0
    assert w.local_gradient[0] == 1.0


def test_worker_local_gradient_is_replaced_not_appended():
    """[CONTRACT] Catches a worker that concatenates instead of replacing."""
    w = RingAllReduceWorker(rank=0, world_size=1)
    w.set_local_gradient(np.array([1.0, 2.0]))
    w.set_local_gradient(np.array([9.0]))
    assert w.local_gradient.shape == (1,)
    np.testing.assert_array_equal(w.local_gradient, [9.0])


# ---------------------------------------------------------------------------
# [GUARD] — defensive validation
# ---------------------------------------------------------------------------


def test_world_size_rejects_bool():
    """[GUARD] Catches isinstance(True, int) accepting True as 1."""
    with pytest.raises(TypeError):
        DistributedDataParallelEngine(world_size=True)


def test_world_size_rejects_float():
    """[GUARD] Catches 3.0 being silently accepted as 3."""
    with pytest.raises(TypeError):
        DistributedDataParallelEngine(world_size=3.0)


def test_world_size_rejects_zero_or_negative():
    """[GUARD] Catches world_size=0 reaching a later division by zero."""
    with pytest.raises(ValueError):
        DistributedDataParallelEngine(world_size=0)
    with pytest.raises(ValueError):
        DistributedDataParallelEngine(world_size=-1)


def test_rejects_wrong_count_of_gradients():
    """[GUARD] Catches zip() silently truncating or dropping extras."""
    engine = DistributedDataParallelEngine(world_size=3)
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.zeros(4), np.zeros(4)])
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.zeros(4)] * 4)


def test_rejects_2d_and_mismatched_and_nonfinite():
    """
    [GUARD] Catches each dropped check: 2-D input, mismatched shapes,
    NaN, +/-inf, and non-numeric input. The first failure to raise
    stops the test, so a failure here says one of the six checks is
    broken, not which.
    """
    engine = DistributedDataParallelEngine(world_size=2)
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.zeros((2, 3)), np.zeros((2, 3))])
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.zeros(4), np.zeros(5)])
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.array([1.0, np.nan]), np.array([1.0, 2.0])])
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.array([1.0, np.inf]), np.array([1.0, 2.0])])
    with pytest.raises(ValueError):
        engine.ring_allreduce([np.array([1.0, -np.inf]), np.array([1.0, 2.0])])
    with pytest.raises((TypeError, ValueError)):
        engine.ring_allreduce([["a", "b"], ["c", "d"]])


def test_worker_rejects_invalid_rank():
    """[GUARD] Catches rank outside [0, world_size) or a bool rank."""
    with pytest.raises(ValueError):
        RingAllReduceWorker(rank=3, world_size=3)
    with pytest.raises(ValueError):
        RingAllReduceWorker(rank=-1, world_size=3)
    with pytest.raises(TypeError):
        RingAllReduceWorker(rank=True, world_size=3)


# Not tested (and why):
# - Sockets / NCCL / CUDA / MPI: single-process simulation; mocking a
#   transport would test the mock, not the code.
# - Compute/communication overlap: no local backward pass implemented.
# - Wall-clock performance: measured numbers describe CPython, not the
#   ring protocol.
# - Fault tolerance: the schedule assumes every rank participates in
#   every round.
# - Compression / mixed precision: not part of Ring-AllReduce.
# - Round-by-round wire-level chunk provenance: last_trace was removed
#   as dead weight. Adding an instrumented schedule test is the
#   recommended next task if the protocol itself is the assessed
#   artifact; see TEST_STRATEGY.md §10.
# - Integer inputs above 2**53: silently rounded by astype; a test
#   would only restate the documented rule.
# - Controlled float64 overflow of a *result* (not an intermediate
#   sum): needs magnitudes near np.finfo(float64).max and is fragile
#   across platforms.
# - N in {6, 8}: the sweep covers (1,2,3,4,5,7). The schedule formula
#   is uniform in N; the chosen N's cover odd/even and prime/composite.
# - success -> failed -> valid: only failed -> valid is tested. A mid-
#   reduction failure does not leave stale state (the next call
#   reassigns every buffer), but a regression could break that.
# - Cross-NumPy-version bit-equality: SIMD lowering can change
#   intermediate rounding; tolerances absorb this.
#
# PROPERTY (Hypothesis-style; dependency not assumed):
#   @given(n=st.integers(1, 8), m=st.integers(0, 32), data=st.data())
#   def test_property_ring_allreduce_equals_numpy_mean(n, m, data):
#       # For any N in [1,8], M in [0,32], and N finite equal-length
#       # 1-D vectors of bounded magnitude, output matches np.mean of
#       # the stacked vectors within dtype tolerance. Magnitudes bounded
#       # to [-1e6, 1e6] so inputs do not overflow by construction;
#       # extreme magnitudes are covered separately by
#       # test_float64_accumulation_is_exact_for_large_magnitudes and
#       # test_intermediate_overflow_raises_controlled_error.
#       grads = [
#           np.array(data.draw(st.lists(
#               st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
#               min_size=m, max_size=m)), dtype=np.float64)
#           for _ in range(n)
#       ]
#       expected = np.mean(np.stack(grads, axis=0), axis=0)
#       actual = DistributedDataParallelEngine(n).ring_allreduce(grads)
#       np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
