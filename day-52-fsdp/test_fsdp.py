"""
Tests for day52_fsdp.

The test names describe what each one catches. Where a test exists to
catch a specific wrong implementation, the docstring names that
implementation in one line — not a full paragraph. The complete
mutation → test map lives in TEST_STRATEGY.md.

Hypothesis is optional. Without it the three property tests are
skipped, not errored; everything else runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from day52_fsdp import (
    FSDPModule,
    FSDPOptimizer,
    MemoryAccounting,
    all_gather,
    reduce_scatter_avg,
    reduce_scatter_sum,
    shard_offsets,
    shard_sizes,
)

try:
    from hypothesis import given, settings  # type: ignore[import-not-found]
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    HAVE_HYPOTHESIS = True
except ImportError:
    HAVE_HYPOTHESIS = False


# --- partition helpers -----------------------------------------------------


def test_shard_sizes_distributes_remainder_to_lowest_ranks():
    """Catches: dropping the remainder, or giving it to the last rank."""
    assert shard_sizes(10, 3) == [4, 3, 3]
    assert shard_sizes(7, 4) == [2, 2, 2, 1]
    assert shard_sizes(8, 2) == [4, 4]
    assert shard_sizes(0, 4) == [0, 0, 0, 0]
    assert sum(shard_sizes(10, 3)) == 10


def test_shard_sizes_allows_empty_shards_when_P_less_than_N():
    """Catches: rejecting or mis-sizing P < N."""
    assert shard_sizes(2, 4) == [1, 1, 0, 0]
    assert shard_sizes(1, 5) == [1, 0, 0, 0, 0]


def test_shard_offsets_cover_total_without_gaps():
    """Catches: missing the trailing offset, or wrong accumulation."""
    assert shard_offsets(10, 3) == [0, 4, 7, 10]
    assert shard_offsets(7, 4) == [0, 2, 4, 6, 7]
    assert shard_offsets(8, 2) == [0, 4, 8]


# --- all_gather ------------------------------------------------------------


def test_all_gather_reconstructs_exact_weights():
    """Catches: wrong order, dropped element, wrong dtype."""
    full = np.arange(16, dtype=np.float32).reshape(4, 4)
    off = shard_offsets(full.size, 4)
    flat = full.reshape(-1)
    shards = [flat[off[r] : off[r + 1]].copy() for r in range(4)]
    out = all_gather(shards, full.shape, full.dtype)
    np.testing.assert_array_equal(out, full)
    assert out.dtype == full.dtype


def test_all_gather_does_not_alias_input_shards():
    """Catches: returning a view of an input shard."""
    shards = [
        np.arange(0, 4, dtype=np.float32),
        np.arange(4, 8, dtype=np.float32),
    ]
    out = all_gather(shards, (8,), np.float32)
    out[0] = -1.0
    assert shards[0][0] == 0.0


def test_all_gather_handles_non_divisible_counts():
    """Catches: assuming P divisible by N and truncating the remainder."""
    full = np.arange(7, dtype=np.float64).reshape(7, 1)
    off = shard_offsets(7, 3)
    flat = full.reshape(-1)
    shards = [flat[off[r] : off[r + 1]].copy() for r in range(3)]
    np.testing.assert_array_equal(all_gather(shards, full.shape, full.dtype), full)


def test_all_gather_rejects_shard_with_wrong_length():
    """Catches: silently accepting a shard length that disagrees with N."""
    with pytest.raises(ValueError, match="length"):
        all_gather([np.zeros(4, dtype=np.float32)], (2, 4), np.float32)


def test_all_gather_rejects_mixed_shard_dtypes():
    """Catches: silent upcast via concatenate (f32 + f64 → f64)."""
    with pytest.raises(TypeError, match="dtype"):
        all_gather(
            [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float64)],
            (4,),
            np.float32,
        )


def test_all_gather_rejects_non_flat_shard():
    """Catches: accepting a 2-D shard that would reshape incorrectly."""
    with pytest.raises(ValueError, match="1-D"):
        all_gather(
            [np.zeros((2, 2), dtype=np.float32), np.zeros(4, dtype=np.float32)],
            (8,),
            np.float32,
        )


# --- world size and rank validation ----------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_world_size_rejects_non_positive(bad):
    """Catches: N ≤ 0 reaching divmod."""
    with pytest.raises(ValueError, match="world_size"):
        shard_sizes(8, bad)


@pytest.mark.parametrize("bad", [True, False])
def test_world_size_rejects_bool(bad):
    """Catches: True/False silently treated as 1/0."""
    with pytest.raises(TypeError, match="int"):
        shard_sizes(8, bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 4, 100])
def test_rank_rejects_out_of_range(bad):
    """Catches: rank < 0 or rank >= N reaching the slice."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    with pytest.raises(ValueError, match="rank"):
        FSDPModule.from_full(W, rank=bad, world_size=4)


def test_rank_rejects_bool():
    """Catches: True accepted as rank 1."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    with pytest.raises(TypeError, match="int"):
        FSDPModule.from_full(W, rank=True, world_size=2)  # type: ignore[arg-type]


def test_module_rejects_bool_shape_dim():
    """Catches: silent int(True) == 1 in a shape."""
    with pytest.raises(TypeError, match="int"):
        FSDPModule(
            local_weight_shard=np.zeros(2, dtype=np.float32),
            full_weight_shape=(True, 2),  # type: ignore[arg-type]
            rank=0,
            world_size=2,
        )


# --- module construction ---------------------------------------------------


def test_module_holds_only_local_shard_not_full_weights():
    """
    Catches: caching the full parameter array on the instance or class.

    Known gap: this scan covers instance and class attributes. It does
    not catch __slots__, closures, or arrays reachable only through
    helper objects — proving absence of hidden state in arbitrary Python
    is not tractable. Documented in TEST_STRATEGY.md.
    """
    full = np.arange(16, dtype=np.float32).reshape(4, 4)
    m = FSDPModule.from_full(full, rank=0, world_size=4)
    assert m.local_weight_shard.size == 4
    for src_name, src in (("instance", vars(m)), ("class", vars(type(m)))):
        for name, val in src.items():
            if isinstance(val, np.ndarray):
                assert val.size < full.size, (
                    f"{src_name} attribute {name!r} holds {val.size} "
                    f"elements; full is {full.size}"
                )


def test_from_full_does_not_mutate_the_caller_array():
    """Catches: from_full normalizing or sorting full_weights in place."""
    full = np.arange(16, dtype=np.float32).reshape(4, 4)
    snapshot = full.copy()
    for r in range(4):
        FSDPModule.from_full(full, rank=r, world_size=4)
    np.testing.assert_array_equal(full, snapshot)


def test_from_full_accepts_non_contiguous_weights():
    """
    Catches: relying on reshape(-1) to silently copy a strided view.
    The implementation calls ascontiguousarray first, so this passes
    regardless of the input's strides.
    """
    base = np.arange(32, dtype=np.float32).reshape(4, 8)
    strided = base[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    peers = [FSDPModule.from_full(strided, rank=r, world_size=3) for r in range(3)]
    np.testing.assert_array_equal(peers[0].all_gather_weights(peers), strided)


def test_module_rejects_shard_length_mismatch():
    """Catches: trusting the caller's shard length."""
    with pytest.raises(ValueError, match="length"):
        FSDPModule(
            local_weight_shard=np.zeros(5, dtype=np.float32),
            full_weight_shape=(2, 4),
            rank=0,
            world_size=2,
        )


def test_sharding_and_gather_round_trip():
    """Catches: any bug in from_full or all_gather_weights together."""
    full = np.arange(16, dtype=np.float32).reshape(4, 4)
    peers = [FSDPModule.from_full(full, rank=r, world_size=3) for r in range(3)]
    np.testing.assert_array_equal(peers[0].all_gather_weights(peers), full)


# --- forward ---------------------------------------------------------------


def test_forward_matches_dense_at_realistic_shape():
    """Catches: wrong order, excluded peer, or wrong shard used."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 4)).astype(np.float32)
    X = rng.standard_normal((2, 8)).astype(np.float32)
    peers = [FSDPModule.from_full(W, rank=r, world_size=4) for r in range(4)]
    np.testing.assert_allclose(peers[0].forward(X, peers), X @ W, rtol=1e-6, atol=1e-6)


def test_forward_world_size_one_matches_dense():
    """Catches: an N ≥ 2 assumption, or dropping the only shard."""
    rng = np.random.default_rng(1)
    W = rng.standard_normal((5, 3)).astype(np.float32)
    X = rng.standard_normal((4, 5)).astype(np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    np.testing.assert_allclose(peers[0].forward(X, peers), X @ W, rtol=1e-6, atol=1e-6)


def test_forward_observes_every_peer_shard():
    """
    Catches: a cached full weight, or ignoring a specific peer.
    X = [1, 1] is picked so BOTH rows of W contribute; with X = [1, 0]
    the test would pass even if rank 1's shard were unused.
    """
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    X = np.array([[1.0, 1.0]], dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=r, world_size=2) for r in range(2)]
    Y0 = peers[0].forward(X, peers)
    peers[1].local_weight_shard = (peers[1].local_weight_shard + 1.0).astype(np.float32)
    Y1 = peers[0].forward(X, peers)
    assert not np.allclose(Y0, Y1)


def test_forward_does_not_mutate_shards():
    """Catches: an in-place write inside forward."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((4, 3)).astype(np.float32)
    X = rng.standard_normal((2, 4)).astype(np.float32)
    peers = [FSDPModule.from_full(W, rank=r, world_size=2) for r in range(2)]
    before = [m.local_weight_shard.copy() for m in peers]
    peers[0].forward(X, peers)
    for i, m in enumerate(peers):
        np.testing.assert_array_equal(m.local_weight_shard, before[i])


def test_forward_with_bias_matches_dense():
    """Catches: bias gathered with the wrong policy or broadcast shape."""
    rng = np.random.default_rng(2)
    W = rng.standard_normal((8, 4)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    X = rng.standard_normal((3, 8)).astype(np.float32)
    peers = [FSDPModule.from_full(W, rank=r, world_size=3, bias=b) for r in range(3)]
    np.testing.assert_allclose(
        peers[1].forward(X, peers), X @ W + b, rtol=1e-6, atol=1e-6
    )


def test_forward_rejects_wrong_feature_dim():
    """Catches: an X mismatch surfacing as a cryptic matmul error."""
    peers = [FSDPModule.from_full(np.zeros((4, 2), dtype=np.float32), 0, 1)]
    with pytest.raises(ValueError, match="feature"):
        peers[0].forward(np.zeros((2, 5), dtype=np.float32), peers)


def test_forward_rejects_wrong_dtype():
    """Catches: silent float32→float64 upcast of X."""
    peers = [FSDPModule.from_full(np.zeros((4, 2), dtype=np.float32), 0, 1)]
    with pytest.raises(TypeError, match="dtype"):
        peers[0].forward(np.zeros((2, 4), dtype=np.float64), peers)


def test_forward_rejects_misordered_peers():
    """Catches: trusting list order without checking peer.rank."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    peers = [FSDPModule.from_full(W, rank=r, world_size=4) for r in range(4)]
    peers[1], peers[2] = peers[2], peers[1]
    with pytest.raises(ValueError, match="rank"):
        peers[0].forward(np.zeros((1, 2), dtype=np.float32), peers)


def test_forward_rejects_missing_peers():
    """Catches: proceeding with fewer peers than declared world_size."""
    W = np.zeros((2, 4), dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=r, world_size=4) for r in range(4)]
    with pytest.raises(ValueError, match="peers"):
        peers[0].forward(np.zeros((1, 2), dtype=np.float32), peers[:3])


def test_forward_rejects_peer_with_different_world_size():
    """Catches: accepting a peer that declares a different N."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    peers = [FSDPModule.from_full(W, rank=r, world_size=4) for r in range(4)]
    peers[1] = FSDPModule.from_full(W, rank=1, world_size=2)
    with pytest.raises(ValueError, match="world_size"):
        peers[0].forward(np.zeros((1, 2), dtype=np.float32), peers)


def test_forward_rejects_caller_not_in_peer_list():
    """
    Catches: an intruder module — same shapes and dtypes — invoking
    forward on a peer list that doesn't contain it by identity. Without
    the identity check every other validation would pass, and the
    intruder's own shard would silently be excluded from the gather.
    """
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    peers = [FSDPModule.from_full(W, rank=r, world_size=4) for r in range(4)]
    intruder = FSDPModule.from_full(W, rank=0, world_size=4)
    with pytest.raises(ValueError, match="not present"):
        intruder.forward(np.zeros((1, 2), dtype=np.float32), peers)


# --- reduce_scatter --------------------------------------------------------


def test_reduce_scatter_sum_matches_hand_computed():
    """Catches: averaging instead of summing, or a wrong partition."""
    contribs = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    ]
    shards = reduce_scatter_sum(contribs, (4,), np.float32)
    np.testing.assert_array_equal(shards[0], np.array([11.0, 22.0], np.float32))
    np.testing.assert_array_equal(shards[1], np.array([33.0, 44.0], np.float32))


def test_reduce_scatter_avg_differs_from_sum():
    """Catches: reduce_scatter_avg delegating to sum with no 1/N."""
    contribs = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
    ]
    s = reduce_scatter_sum(contribs, (4,), np.float32)
    a = reduce_scatter_avg(contribs, (4,), np.float32)
    np.testing.assert_allclose(a[0], s[0] / 2.0)
    assert not np.allclose(a[0], s[0])


def test_reduce_scatter_avg_rejects_integer_dtype():
    """Catches: silent truncation of a non-integral average."""
    with pytest.raises(TypeError, match="floating"):
        reduce_scatter_avg(
            [np.array([1], np.int32), np.array([2], np.int32)],
            (1,),
            np.int32,
        )


def test_reduce_scatter_preserves_every_element():
    """Catches: dropping or duplicating an element (P=7, N=2)."""
    contribs = [
        np.arange(7, dtype=np.float32),
        np.arange(7, dtype=np.float32) * 10,
    ]
    shards = reduce_scatter_sum(contribs, (7,), np.float32)
    np.testing.assert_array_equal(
        np.concatenate(shards), np.arange(7, dtype=np.float32) * 11
    )


def test_reduce_scatter_rejects_wrong_contribution_shape():
    """Catches: silent truncation via reshape(-1)."""
    with pytest.raises(ValueError, match="shape"):
        reduce_scatter_sum(
            [np.zeros((2, 4), np.float32), np.zeros((4, 2), np.float32)],
            (2, 4),
            np.float32,
        )


# --- gradient helpers ------------------------------------------------------


def test_compute_weight_grad_rejects_mixed_dtypes():
    """Catches: silent upcast when X and dY disagree on dtype."""
    with pytest.raises(TypeError, match="dtype"):
        FSDPModule.compute_weight_grad(
            np.ones((2, 3), np.float32), np.ones((2, 4), np.float64)
        )


# --- optimizer -------------------------------------------------------------


def test_sgd_step_matches_hand_computed():
    """Catches: sign flip, silent divide by N, wrong lr."""
    W = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    FSDPOptimizer(peers[0], lr=0.1).step(np.full(8, 0.5, dtype=np.float32))
    np.testing.assert_allclose(
        peers[0].local_weight_shard, W.reshape(-1) - 0.05, rtol=1e-6
    )


def test_optimizer_touches_only_its_own_module():
    """Catches: an optimizer reaching into peers to update them."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    peers = [FSDPModule.from_full(W, rank=r, world_size=2) for r in range(2)]
    before = peers[1].local_weight_shard.copy()
    FSDPOptimizer(peers[0], lr=0.5).step(np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(peers[1].local_weight_shard, before)


def test_optimizer_on_rank1_updates_rank1_slice():
    """Catches: updating the wrong rank's slice regardless of the binding."""
    W = np.arange(8, dtype=np.float32).reshape(2, 4)
    peers = [FSDPModule.from_full(W, rank=r, world_size=2) for r in range(2)]
    FSDPOptimizer(peers[1], lr=1.0).step(np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(
        peers[0].local_weight_shard, np.array([0, 1, 2, 3], np.float32)
    )
    np.testing.assert_array_equal(
        peers[1].local_weight_shard, np.array([3, 4, 5, 6], np.float32)
    )


def test_optimizer_step_is_atomic_on_bad_bias_grad():
    """Catches: committing the weight update before bias validation."""
    W = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([0.5, 0.5], dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1, bias=b)]
    opt = FSDPOptimizer(peers[0], lr=0.1)
    w0 = peers[0].local_weight_shard.copy()
    b0 = peers[0].local_bias_shard.copy()
    with pytest.raises(ValueError, match="shape"):
        opt.step(np.ones(2, np.float32), np.ones(3, np.float32))
    np.testing.assert_array_equal(peers[0].local_weight_shard, w0)
    np.testing.assert_array_equal(peers[0].local_bias_shard, b0)


def test_optimizer_step_does_not_mutate_grad_inputs():
    """Catches: an in-place `g *= lr` corrupting the caller's buffer."""
    W = np.array([[1.0, 2.0]], dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    g = np.array([0.3, -0.7], dtype=np.float32)
    snapshot = g.copy()
    FSDPOptimizer(peers[0], lr=0.1).step(g)
    np.testing.assert_array_equal(g, snapshot)


def test_optimizer_rejects_overflowing_update():
    """Catches: silent inf/nan reaching the shard after an overflow."""
    W = np.ones((1, 4), dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    opt = FSDPOptimizer(peers[0], lr=1e38)
    with pytest.raises(FloatingPointError, match="overflow"):
        opt.step(np.full(4, 10.0, np.float32))


def test_optimizer_step_is_atomic_when_bias_overflows():
    """
    Catches: committing the weight update, then discovering the bias
    update overflows. Weight here is perfectly updatable; only bias
    overflows. Both shards must be unchanged after the exception.
    """
    f32 = np.float32
    W = np.ones((1, 2), dtype=f32)
    b = np.full(2, -3e38, dtype=f32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1, bias=b)]
    opt = FSDPOptimizer(peers[0], lr=1.0)
    w0 = peers[0].local_weight_shard.copy()
    b0 = peers[0].local_bias_shard.copy()
    with pytest.raises(FloatingPointError, match="overflow"):
        opt.step(np.ones(2, f32), np.full(2, 3e38, f32))
    np.testing.assert_array_equal(peers[0].local_weight_shard, w0)
    np.testing.assert_array_equal(peers[0].local_bias_shard, b0)


def test_optimizer_accepts_large_but_finite_updates():
    """Catches: overflow guards rejecting legitimate large values."""
    f32 = np.float32
    W = np.full((1, 4), 1e30, dtype=f32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    FSDPOptimizer(peers[0], lr=1.0).step(np.full(4, 1e30, f32))
    np.testing.assert_array_equal(peers[0].local_weight_shard, np.zeros(4, f32))


def test_optimizer_accumulates_across_multiple_steps():
    """Catches: re-reading an initial snapshot on subsequent calls."""
    W = np.array([[1.0, 2.0]], dtype=np.float32)
    peers = [FSDPModule.from_full(W, rank=0, world_size=1)]
    opt = FSDPOptimizer(peers[0], lr=0.1)
    g = np.ones(2, dtype=np.float32)
    for _ in range(3):
        opt.step(g)
    np.testing.assert_allclose(
        peers[0].local_weight_shard,
        np.array([0.7, 1.7], np.float32),
        rtol=1e-6,
    )


def test_optimizer_rejects_full_shape_gradient():
    """Catches: accepting an un-reduced full gradient by broadcasting."""
    peers = [FSDPModule.from_full(np.zeros((2, 4), np.float32), 0, 1)]
    with pytest.raises(ValueError, match="shape"):
        FSDPOptimizer(peers[0], lr=0.1).step(np.zeros((2, 4), np.float32))


def test_optimizer_rejects_wrong_grad_dtype():
    """Catches: a float64 gradient silently upcasting the shard."""
    peers = [FSDPModule.from_full(np.zeros((2, 4), np.float32), 0, 1)]
    with pytest.raises(TypeError, match="dtype"):
        FSDPOptimizer(peers[0], lr=0.1).step(np.zeros(8, np.float64))


def test_optimizer_rejects_nonfinite_grad():
    """Catches: NaN in a gradient reaching the parameter."""
    peers = [FSDPModule.from_full(np.zeros((2, 4), np.float32), 0, 1)]
    bad = np.zeros(8, np.float32)
    bad[0] = np.nan
    with pytest.raises(FloatingPointError):
        FSDPOptimizer(peers[0], lr=0.1).step(bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("inf"), float("nan")])
def test_optimizer_rejects_invalid_lr(bad):
    """Catches: lr = 0, negative, inf, or nan."""
    peers = [FSDPModule.from_full(np.zeros((2, 4), np.float32), 0, 1)]
    with pytest.raises(ValueError, match="lr"):
        FSDPOptimizer(peers[0], lr=bad)


# --- end to end ------------------------------------------------------------


def test_full_training_step_matches_dense_reference():
    """
    Catches: any drift between the sharded pipeline (forward, gradient,
    Reduce-Scatter, update, reconstruct) and a plain NumPy reference on
    identical inputs.

    Scope note: this test uses the SAME full gradient N times and
    averages it. That is algebraically equivalent to the dense
    reference, but it does not simulate per-rank minibatch gradients.
    """
    rng = np.random.default_rng(7)
    W = rng.standard_normal((6, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    X = rng.standard_normal((4, 6)).astype(np.float32)
    dY = rng.standard_normal((4, 3)).astype(np.float32)
    lr, N = 0.05, 3

    # Dense reference
    dW_ref = X.T @ dY
    db_ref = dY.sum(axis=0)
    W_ref_new = W - lr * dW_ref
    b_ref_new = b - lr * db_ref

    # FSDP pipeline
    peers = [FSDPModule.from_full(W, rank=r, world_size=N, bias=b) for r in range(N)]
    assert np.allclose(peers[0].forward(X, peers), X @ W + b, atol=1e-6)

    dW = FSDPModule.compute_weight_grad(X, dY)
    db = FSDPModule.compute_bias_grad(dY)
    dW_shards = reduce_scatter_avg([dW] * N, W.shape, W.dtype)
    db_shards = reduce_scatter_avg([db] * N, b.shape, b.dtype)

    for r in range(N):
        FSDPOptimizer(peers[r], lr=lr).step(dW_shards[r], db_shards[r])

    np.testing.assert_allclose(
        peers[0].all_gather_weights(peers), W_ref_new, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        peers[0].all_gather_bias(peers), b_ref_new, rtol=1e-6, atol=1e-6
    )


def test_identical_fresh_runs_produce_identical_results():
    """
    Catches: any hidden state that makes two independent runs diverge.

    Scope note: this checks reproducibility across runs, not hidden
    state accumulating within one optimizer instance. That is covered
    by test_optimizer_accumulates_across_multiple_steps.
    """
    rng = np.random.default_rng(11)
    W0 = rng.standard_normal((4, 3)).astype(np.float32)
    g = rng.standard_normal(12).astype(np.float32)

    def run():
        peers = [FSDPModule.from_full(W0, rank=0, world_size=1)]
        opt = FSDPOptimizer(peers[0], lr=0.1)
        for _ in range(3):
            opt.step(g)
        return peers[0].local_weight_shard.copy()

    np.testing.assert_array_equal(run(), run())


# --- memory accounting -----------------------------------------------------


def test_memory_accounting_uses_shard_not_full_size():
    """Catches: using P rather than P/N, or omitting a byte term."""
    ma = MemoryAccounting(
        num_params=8,
        world_size=4,
        bytes_per_param=4,
        bytes_per_grad=4,
        bytes_per_state=8,
    )
    assert ma.persistent_per_rank() == {
        "param_shard": 8,
        "grad_shard": 8,
        "state_shard": 16,
    }
    assert ma.standard_dp_per_rank() == 128


def test_memory_accounting_uses_max_shard_for_uneven_P():
    """Catches: using the truncated average ⌊P/N⌋ instead of ⌈P/N⌉."""
    ma = MemoryAccounting(
        num_params=10,
        world_size=3,
        bytes_per_param=4,
        bytes_per_grad=4,
        bytes_per_state=8,
    )
    # shard_sizes(10, 3) == [4, 3, 3] → max = 4
    assert ma.persistent_per_rank()["param_shard"] == 16


def test_memory_accounting_reports_zero_state_for_zero_input():
    """
    Catches: conflating "bytes per state" with "state exists". This
    module's SGD has no state arrays; bytes_per_state=0 must produce a
    zero state shard.
    """
    ma = MemoryAccounting(
        num_params=8,
        world_size=4,
        bytes_per_param=4,
        bytes_per_grad=4,
        bytes_per_state=0,
    )
    assert ma.persistent_per_rank()["state_shard"] == 0


def test_memory_accounting_rejects_zero_world_size():
    """Catches: N=0 reaching persistent_per_rank's division."""
    with pytest.raises(ValueError, match="world_size"):
        MemoryAccounting(
            num_params=8,
            world_size=0,
            bytes_per_param=4,
            bytes_per_grad=4,
            bytes_per_state=8,
        )


# --- property tests (skipped without Hypothesis) ---------------------------

if HAVE_HYPOTHESIS:

    @given(
        P=st.integers(min_value=0, max_value=50),
        N=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_shard_round_trip(P, N):
        vec = np.arange(P, dtype=np.float64).reshape(P, 1)
        peers = [FSDPModule.from_full(vec, rank=r, world_size=N) for r in range(N)]
        np.testing.assert_array_equal(peers[0].all_gather_weights(peers), vec)

    @given(
        P=st.integers(min_value=0, max_value=30),
        N=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_reduce_scatter_preserves_ordering(P, N):
        g = np.arange(P, dtype=np.float32)
        shards = reduce_scatter_sum([g.copy() for _ in range(N)], (P,), np.float32)
        np.testing.assert_array_equal(
            np.concatenate(shards), (N * g).astype(np.float32)
        )

    @given(
        rows=st.integers(min_value=1, max_value=4),
        in_f=st.integers(min_value=1, max_value=6),
        out_f=st.integers(min_value=1, max_value=6),
        N=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_forward_equals_dense(rows, in_f, out_f, N):
        rng = np.random.default_rng(rows * 1000 + in_f * 100 + out_f * 10 + N)
        W = rng.standard_normal((in_f, out_f)).astype(np.float32)
        X = rng.standard_normal((rows, in_f)).astype(np.float32)
        peers = [FSDPModule.from_full(W, rank=r, world_size=N) for r in range(N)]
        np.testing.assert_allclose(
            peers[0].forward(X, peers), X @ W, rtol=1e-5, atol=1e-5
        )

else:

    def test_property_shard_round_trip():
        pytest.skip("hypothesis not installed")

    def test_property_reduce_scatter_preserves_ordering():
        pytest.skip("hypothesis not installed")

    def test_property_forward_equals_dense():
        pytest.skip("hypothesis not installed")
