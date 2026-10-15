"""
test_tensor_parallelism.py

Tests for the two tensor-parallel linear layers.

How to read the tags:

    [ROBUST]   Fails if a specific wrong implementation of the layer is
               used. This is the majority of the tests.

    [GUARD]    Fails only if a defensive input check is removed. Each one
               names a specific kind of bad input.

The test-to-mutation map and tag counts live in TEST_STRATEGY.md.
"""

import numpy as np
import pytest
from day50_tensor_parallelism import ColumnParallelLinear, RowParallelLinear

# ---------------------------------------------------------------------------
# Small, hand-checkable cases.
#
# The numbers are small enough to check by hand on paper. That makes it
# obvious when a test breaks whether the implementation or the test is
# wrong.
# ---------------------------------------------------------------------------


def _column_case():
    """Return a small column-parallel case with fixed numbers."""
    in_features, out_features, world_size = 3, 4, 2
    full_weight = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        dtype=np.float32,
    )
    full_bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    x = np.array([[1, 0, -1], [0.5, 2, -0.5]], dtype=np.float32)
    return in_features, out_features, world_size, full_weight, full_bias, x


def _row_case():
    """Return a small row-parallel case with fixed numbers."""
    in_features, out_features, world_size = 4, 2, 2
    full_weight = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=np.float32,
    )
    full_bias = np.array([0.5, -0.5], dtype=np.float32)
    x = np.array([[1, 2, 3, 4], [0.5, -1, 2, 0]], dtype=np.float32)
    return in_features, out_features, world_size, full_weight, full_bias, x


def _make_column_shards(in_features, out_features, world_size, full_weight, full_bias):
    """Build one ColumnParallelLinear per rank, all from the same weights."""
    return [
        ColumnParallelLinear(
            in_features,
            out_features,
            world_size,
            r,
            full_weight=full_weight,
            full_bias=full_bias,
        )
        for r in range(world_size)
    ]


def _make_row_shards(in_features, out_features, world_size, full_weight, full_bias):
    """Build one RowParallelLinear per rank, all from the same weights."""
    return [
        RowParallelLinear(
            in_features,
            out_features,
            world_size,
            r,
            full_weight=full_weight,
            full_bias=full_bias,
        )
        for r in range(world_size)
    ]


# ---------------------------------------------------------------------------
# Dense-reference equivalence.
#
# These are the most important tests: they compare the sharded result
# against the plain dense computation X @ W + b. If the math is wrong,
# these fail.
# ---------------------------------------------------------------------------


# [ROBUST] Fails if the column shards use the wrong weight axis, the wrong
# rank offset, or the wrong bias slice. The concatenated output must equal
# the dense reference numerically.
def test_column_parallel_concatenation_matches_dense_with_fixed_weights():
    in_features, out_features, world_size, full_weight, full_bias, x = _column_case()
    shards = _make_column_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    got = np.concatenate([s.forward(x) for s in shards], axis=1)
    expected = x @ full_weight + full_bias
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


# [ROBUST] Fails if the row shards split the wrong way, use the wrong rank
# offset, add bias per rank, or sum in the wrong order. The aggregated
# output must equal the dense reference numerically.
def test_row_parallel_all_reduce_matches_dense_with_fixed_weights():
    in_features, out_features, world_size, full_weight, full_bias, x = _row_case()
    local_in = in_features // world_size
    x_shards = [x[:, r * local_in : (r + 1) * local_in] for r in range(world_size)]
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    local_outputs = [shards[r].forward(x_shards[r]) for r in range(world_size)]
    got = RowParallelLinear.all_reduce_sum(local_outputs, full_bias)
    expected = x @ full_weight + full_bias
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


# [ROBUST] Fails if forward accepts an input split along the wrong axis.
# Splitting on the batch axis gives a shard of shape (1, in_features)
# instead of (batch, local_in_features), and forward must reject it.
def test_row_wrong_input_split_axis_is_rejected():
    in_features, out_features, world_size, full_weight, full_bias, x = _row_case()
    wrong_x_shards = [x[0:1], x[1:2]]
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    with pytest.raises(ValueError):
        shards[0].forward(wrong_x_shards[0])


# ---------------------------------------------------------------------------
# Shard offsets and shard consistency.
#
# These check that rank i gets the exact slice it should, and that no
# slice is duplicated or skipped.
# ---------------------------------------------------------------------------


# [ROBUST] Fails if the column slices for rank 0 and rank 1 are shifted,
# duplicated, or overlapping.
def test_column_rank_offsets_are_non_overlapping_and_correct():
    in_features, out_features, world_size, full_weight, full_bias, _ = _column_case()
    shards = _make_column_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    np.testing.assert_array_equal(shards[0].weight, full_weight[:, :2])
    np.testing.assert_array_equal(shards[1].weight, full_weight[:, 2:])


# [ROBUST] Fails if the row slices for rank 0 and rank 1 are shifted,
# duplicated, or overlapping.
def test_row_rank_offsets_are_non_overlapping_and_correct():
    in_features, out_features, world_size, full_weight, full_bias, _ = _row_case()
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    np.testing.assert_array_equal(shards[0].weight, full_weight[:2, :])
    np.testing.assert_array_equal(shards[1].weight, full_weight[2:, :])


# [ROBUST] Fails if each rank creates its own unrelated random matrix
# instead of slicing one shared matrix. Two ranks with the same seed and
# the same dimensions must receive disjoint slices of one matrix.
def test_same_seed_across_ranks_yields_disjoint_slices_of_same_full_matrix():
    in_features, out_features, world_size, seed = 4, 6, 3, 42
    shards = [
        ColumnParallelLinear(in_features, out_features, world_size, r, seed=seed)
        for r in range(world_size)
    ]
    reconstructed = np.concatenate([s.weight for s in shards], axis=1)
    # Build the logical full matrix by asking a world-size-1 instance for
    # the same seed. If the ranks were building unrelated matrices, the
    # concat above would not match this.
    full = ColumnParallelLinear(in_features, out_features, 1, 0, seed=seed).weight
    np.testing.assert_array_equal(reconstructed, full)


# [ROBUST] Fails if the column shards are ordered wrong or their
# boundaries are off. Concatenating them must rebuild the input matrix.
def test_column_shards_reconstruct_full_weight():
    in_features, out_features, world_size, full_weight, full_bias, _ = _column_case()
    shards = _make_column_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    reconstructed = np.concatenate([s.weight for s in shards], axis=1)
    np.testing.assert_array_equal(reconstructed, full_weight)


# [ROBUST] Same as above but for the row axis.
def test_row_shards_reconstruct_full_weight():
    in_features, out_features, world_size, full_weight, full_bias, _ = _row_case()
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    reconstructed = np.concatenate([s.weight for s in shards], axis=0)
    np.testing.assert_array_equal(reconstructed, full_weight)


# ---------------------------------------------------------------------------
# Bias handling.
# ---------------------------------------------------------------------------


# [ROBUST] Fails if RowParallelLinear.forward adds bias. Only the
# collective should add bias, and only once.
def test_row_forward_does_not_add_bias():
    in_features, out_features, world_size, full_weight, full_bias, _ = _row_case()
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    x_shard = np.ones((1, shards[0].local_in_features), dtype=np.float32)
    z = shards[0].forward(x_shard)
    np.testing.assert_allclose(z, x_shard @ shards[0].weight, rtol=1e-6, atol=1e-6)
    assert not np.allclose(z, x_shard @ shards[0].weight + shards[0].bias)


# [ROBUST] Fails if the All-Reduce helper drops one rank's contribution.
def test_all_reduce_omitting_a_partial_differs_from_dense():
    in_features, out_features, world_size, full_weight, full_bias, x = _row_case()
    local_in = in_features // world_size
    x_shards = [x[:, r * local_in : (r + 1) * local_in] for r in range(world_size)]
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    partials = [shards[r].forward(x_shards[r]) for r in range(world_size)]

    correct = RowParallelLinear.all_reduce_sum(partials, full_bias)
    omitted = RowParallelLinear.all_reduce_sum(partials[:1], full_bias)
    assert not np.allclose(correct, omitted, rtol=1e-5, atol=1e-6)


# [ROBUST] Fails if the All-Reduce helper counts one rank's contribution
# twice.
def test_all_reduce_duplicating_a_partial_differs_from_dense():
    in_features, out_features, world_size, full_weight, full_bias, x = _row_case()
    local_in = in_features // world_size
    x_shards = [x[:, r * local_in : (r + 1) * local_in] for r in range(world_size)]
    shards = _make_row_shards(
        in_features, out_features, world_size, full_weight, full_bias
    )
    partials = [shards[r].forward(x_shards[r]) for r in range(world_size)]

    correct = RowParallelLinear.all_reduce_sum(partials, full_bias)
    duplicated = RowParallelLinear.all_reduce_sum([partials[0], partials[0]], full_bias)
    assert not np.allclose(correct, duplicated, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Degenerate case: world_size == 1.
#
# With one rank, both parallel forms must reduce to the plain dense layer.
# ---------------------------------------------------------------------------


# [ROBUST] Fails if the column layer does not behave like a dense layer
# when there is only one rank.
def test_world_size_one_column_reduces_to_dense():
    in_features, out_features, _, full_weight, full_bias, x = _column_case()
    layer = ColumnParallelLinear(
        in_features,
        out_features,
        1,
        0,
        full_weight=full_weight,
        full_bias=full_bias,
    )
    np.testing.assert_allclose(
        layer.forward(x),
        x @ full_weight + full_bias,
        rtol=1e-5,
        atol=1e-6,
    )


# [ROBUST] Fails if the row layer does not behave like a dense layer
# when there is only one rank.
def test_world_size_one_row_reduces_to_dense():
    in_features, out_features, _, full_weight, full_bias, x = _row_case()
    layer = RowParallelLinear(
        in_features,
        out_features,
        1,
        0,
        full_weight=full_weight,
        full_bias=full_bias,
    )
    z = layer.forward(x)
    got = RowParallelLinear.all_reduce_sum([z], full_bias)
    np.testing.assert_allclose(got, x @ full_weight + full_bias, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Constructor validation.
#
# Every guard test below fails if the corresponding defensive check is
# removed. Each layer is tested separately because the two constructors
# validate their own dimensions; the shared helpers do not make the row
# constructor's checks automatically run when only the column constructor
# is exercised.
# ---------------------------------------------------------------------------


# [GUARD] Column: out_features must be divisible by world_size.
def test_column_rejects_non_divisible_out_features():
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 5, 2, 0, seed=0)


# [GUARD] Row: in_features must be divisible by world_size.
def test_row_rejects_non_divisible_in_features():
    with pytest.raises(ValueError):
        RowParallelLinear(5, 3, 2, 0, seed=0)


# [GUARD] Column: True is not an integer dimension, even though bool is a
# subclass of int in Python.
def test_column_constructor_rejects_boolean_dimensions():
    with pytest.raises(TypeError):
        ColumnParallelLinear(True, 4, 2, 0, seed=0)
    with pytest.raises(TypeError):
        ColumnParallelLinear(3, 4, True, 0, seed=0)


# [GUARD] Row: same as above for the row constructor.
def test_row_constructor_rejects_boolean_dimensions():
    with pytest.raises(TypeError):
        RowParallelLinear(True, 4, 2, 0, seed=0)
    with pytest.raises(TypeError):
        RowParallelLinear(4, 4, True, 0, seed=0)


# [GUARD] Column: world_size == 0 is rejected.
def test_column_constructor_rejects_zero_world_size():
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 4, 0, 0, seed=0)


# [GUARD] Row: world_size == 0 is rejected.
def test_row_constructor_rejects_zero_world_size():
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 0, 0, seed=0)


# [GUARD] Column: negative dimensions are rejected.
def test_column_constructor_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        ColumnParallelLinear(-3, 4, 2, 0, seed=0)


# [GUARD] Row: negative dimensions are rejected.
def test_row_constructor_rejects_negative_dimensions():
    with pytest.raises(ValueError):
        RowParallelLinear(4, -2, 2, 0, seed=0)


# [GUARD] Column: float dimensions are rejected.
def test_column_constructor_rejects_non_integer_dimensions():
    with pytest.raises(TypeError):
        ColumnParallelLinear(3.0, 4, 2, 0, seed=0)
    with pytest.raises(TypeError):
        ColumnParallelLinear(3, 4, 2.0, 0, seed=0)


# [GUARD] Row: float dimensions are rejected.
def test_row_constructor_rejects_non_integer_dimensions():
    with pytest.raises(TypeError):
        RowParallelLinear(4.0, 2, 2, 0, seed=0)
    with pytest.raises(TypeError):
        RowParallelLinear(4, 2, 2.0, 0, seed=0)


# [GUARD] Column: rank < 0 is rejected.
def test_column_constructor_rejects_negative_rank():
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 4, 2, -1, seed=0)


# [GUARD] Row: rank < 0 is rejected.
def test_row_constructor_rejects_negative_rank():
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 2, -1, seed=0)


# [GUARD] Column: rank == world_size is rejected.
def test_column_constructor_rejects_rank_equal_to_world_size():
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 4, 2, 2, seed=0)


# [GUARD] Row: rank == world_size is rejected.
def test_row_constructor_rejects_rank_equal_to_world_size():
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 2, 2, seed=0)


# [GUARD] Column: True is not a valid rank.
def test_column_constructor_rejects_boolean_rank():
    with pytest.raises(TypeError):
        ColumnParallelLinear(3, 4, 2, True, seed=0)


# [GUARD] Row: True is not a valid rank.
def test_row_constructor_rejects_boolean_rank():
    with pytest.raises(TypeError):
        RowParallelLinear(4, 2, 2, True, seed=0)


# [GUARD] Column: a full_weight with the wrong shape is rejected.
def test_column_constructor_rejects_mismatched_full_weight_shape():
    bad = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 4, 2, 0, full_weight=bad)


# [GUARD] Row: a full_weight with the wrong shape is rejected.
def test_row_constructor_rejects_mismatched_full_weight_shape():
    bad = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 2, 0, full_weight=bad)


# [GUARD] Column: a full_bias with the wrong length is rejected.
def test_column_constructor_rejects_mismatched_full_bias_shape():
    with pytest.raises(ValueError):
        ColumnParallelLinear(
            3, 4, 2, 0, seed=0, full_bias=np.zeros(3, dtype=np.float32)
        )


# [GUARD] Row: a full_bias with the wrong length is rejected.
def test_row_constructor_rejects_mismatched_full_bias_shape():
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 2, 0, seed=0, full_bias=np.zeros(3, dtype=np.float32))


# [GUARD] Column: the constructor refuses to build without a seed or a
# full_weight — there would be no way to get weights.
def test_column_constructor_requires_seed_or_full_weight():
    with pytest.raises(ValueError):
        ColumnParallelLinear(3, 4, 2, 0)


# [GUARD] Row: same as above for the row constructor.
def test_row_constructor_requires_seed_or_full_weight():
    with pytest.raises(ValueError):
        RowParallelLinear(4, 2, 2, 0)


# ---------------------------------------------------------------------------
# Forward input validation.
#
# Each layer's forward must reject the same classes of bad input. Both are
# tested explicitly because the shared helper does not guarantee that a
# given constructor calls it for a given argument.
# ---------------------------------------------------------------------------


# [GUARD] Column: input feature size must match in_features.
def test_column_forward_rejects_wrong_in_features():
    layer = ColumnParallelLinear(3, 4, 2, 0, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((2, 4), dtype=np.float32))


# [GUARD] Row: shard feature size must match local_in_features.
def test_row_forward_rejects_wrong_shard_in_features():
    layer = RowParallelLinear(4, 2, 2, 0, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((2, 3), dtype=np.float32))


# [GUARD] Column: 1-D input is rejected.
def test_column_forward_rejects_non_2d_input():
    layer = ColumnParallelLinear(3, 4, 2, 0, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((3,), dtype=np.float32))


# [GUARD] Row: 1-D input is rejected.
def test_row_forward_rejects_non_2d_input():
    layer = RowParallelLinear(4, 2, 2, 0, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((2,), dtype=np.float32))


# [GUARD] Column: float64 input is rejected, not silently cast.
def test_column_forward_rejects_non_float32_dtype():
    layer = ColumnParallelLinear(3, 4, 2, 0, seed=0)
    with pytest.raises(TypeError):
        layer.forward(np.zeros((2, 3), dtype=np.float64))


# [GUARD] Row: float64 input is rejected.
def test_row_forward_rejects_non_float32_dtype():
    layer = RowParallelLinear(4, 2, 2, 0, seed=0)
    with pytest.raises(TypeError):
        layer.forward(np.zeros((2, 2), dtype=np.float64))


# [GUARD] Column: NaN and Inf inputs are rejected.
def test_column_forward_rejects_non_finite_input():
    layer = ColumnParallelLinear(3, 4, 2, 0, seed=0)
    x = np.zeros((2, 3), dtype=np.float32)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        layer.forward(x)
    x[0, 0] = np.inf
    with pytest.raises(ValueError):
        layer.forward(x)


# [GUARD] Row: NaN and Inf inputs are rejected.
def test_row_forward_rejects_non_finite_input():
    layer = RowParallelLinear(4, 2, 2, 0, seed=0)
    x_shard = np.zeros((2, 2), dtype=np.float32)
    x_shard[0, 0] = np.nan
    with pytest.raises(ValueError):
        layer.forward(x_shard)
    x_shard[0, 0] = np.inf
    with pytest.raises(ValueError):
        layer.forward(x_shard)


# [ROBUST] Fails if finite inputs that overflow to Inf during the matmul
# are returned silently. The contract is finite-in, finite-out: a product
# beyond float32's range must raise.
def test_column_forward_rejects_overflowed_output():
    weight = np.array([[1e20]], dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    x = np.array([[1e20]], dtype=np.float32)
    layer = ColumnParallelLinear(
        1,
        1,
        1,
        0,
        full_weight=weight,
        full_bias=bias,
    )
    with pytest.raises(ValueError, match="finite"):
        layer.forward(x)


# [ROBUST] Same contract as above, for the row-parallel path.
def test_row_forward_rejects_overflowed_output():
    weight = np.array([[1e20]], dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    x_shard = np.array([[1e20]], dtype=np.float32)
    layer = RowParallelLinear(
        1,
        1,
        1,
        0,
        full_weight=weight,
        full_bias=bias,
    )
    with pytest.raises(ValueError, match="finite"):
        layer.forward(x_shard)


# ---------------------------------------------------------------------------
# Collective validation.
# ---------------------------------------------------------------------------


# [GUARD] An empty list of partial outputs is rejected.
def test_all_reduce_sum_rejects_empty_sequence():
    with pytest.raises(ValueError):
        RowParallelLinear.all_reduce_sum([], np.zeros(2, dtype=np.float32))


# [GUARD] A generator (not a Sequence) is rejected with a TypeError, so
# the caller can distinguish it from an empty-list ValueError.
def test_all_reduce_sum_rejects_wrong_container_type():
    partial = np.zeros((1, 2), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    with pytest.raises(TypeError, match="sequence"):
        RowParallelLinear.all_reduce_sum(iter([partial]), bias)


# [GUARD] Partial outputs with different shapes are rejected.
def test_all_reduce_sum_rejects_mismatched_shapes():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        RowParallelLinear.all_reduce_sum([a, b], np.zeros(2, dtype=np.float32))


# [GUARD] A bias with the wrong output dimension is rejected.
def test_all_reduce_sum_rejects_invalid_bias_shape():
    a = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        RowParallelLinear.all_reduce_sum([a], np.zeros(3, dtype=np.float32))


# [GUARD] A non-finite partial is rejected.
def test_all_reduce_sum_rejects_non_finite_partial():
    a = np.zeros((2, 2), dtype=np.float32)
    a[0, 0] = np.nan
    with pytest.raises(ValueError):
        RowParallelLinear.all_reduce_sum([a], np.zeros(2, dtype=np.float32))


# [GUARD] A non-finite bias is rejected even when every partial is finite.
def test_all_reduce_sum_rejects_non_finite_bias():
    partial = np.zeros((1, 2), dtype=np.float32)
    bias = np.array([0.0, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        RowParallelLinear.all_reduce_sum([partial], bias)


# ---------------------------------------------------------------------------
# Reproducibility, dtype, and side-effect contracts.
# ---------------------------------------------------------------------------


# [ROBUST] Same seed produces the same weights.
def test_same_seed_produces_same_shard_weights():
    a = ColumnParallelLinear(3, 4, 2, 0, seed=123)
    b = ColumnParallelLinear(3, 4, 2, 0, seed=123)
    np.testing.assert_array_equal(a.weight, b.weight)
    np.testing.assert_array_equal(a.bias, b.bias)


# [ROBUST] Different seeds produce different weights for this specific
# pair. This is one check, not a proof that no seed pair ever collides.
def test_different_seeds_produce_different_shard_weights():
    a = ColumnParallelLinear(3, 4, 2, 0, seed=123)
    b = ColumnParallelLinear(3, 4, 2, 0, seed=456)
    assert not np.array_equal(a.weight, b.weight)


# [ROBUST] The constructor must not read or advance the global NumPy RNG
# state. It uses a local `default_rng(seed)` object. If it instead used
# `np.random.seed(...)` or `np.random.randn(...)`, the second draw here
# would not match the first.
def test_constructor_does_not_disturb_global_numpy_rng():
    np.random.seed(42)
    expected = np.random.random()
    np.random.seed(42)
    ColumnParallelLinear(3, 4, 2, 0, seed=123)
    RowParallelLinear(4, 2, 2, 0, seed=123)
    got = np.random.random()
    assert got == expected


# [ROBUST] Calling forward does not mutate the layer's weights, its bias,
# or the input array.
def test_repeated_forward_does_not_mutate_state_or_input():
    in_features, out_features, world_size, full_weight, full_bias, x = _column_case()
    layer = ColumnParallelLinear(
        in_features,
        out_features,
        world_size,
        0,
        full_weight=full_weight,
        full_bias=full_bias,
    )
    w_before = layer.weight.copy()
    b_before = layer.bias.copy()
    x_before = x.copy()
    y1 = layer.forward(x)
    y2 = layer.forward(x)
    np.testing.assert_array_equal(layer.weight, w_before)
    np.testing.assert_array_equal(layer.bias, b_before)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y1, y2)


# [ROBUST] all_reduce_sum does not mutate any of its inputs.
def test_all_reduce_sum_does_not_mutate_inputs():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.full((2, 2), 2.0, dtype=np.float32)
    bias = np.array([0.5, 0.5], dtype=np.float32)
    a_before, b_before, bias_before = a.copy(), b.copy(), bias.copy()
    RowParallelLinear.all_reduce_sum([a, b], bias)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
    np.testing.assert_array_equal(bias, bias_before)


# [ROBUST] The column constructor does not keep a reference to the
# caller's full_weight or full_bias.
def test_column_constructor_does_not_alias_caller_arrays():
    in_features, out_features, world_size, full_weight, full_bias, _ = _column_case()
    layer = ColumnParallelLinear(
        in_features,
        out_features,
        world_size,
        0,
        full_weight=full_weight,
        full_bias=full_bias,
    )
    full_weight[:] = 999.0
    full_bias[:] = 999.0
    assert not np.all(layer.weight == 999.0)
    assert not np.all(layer.bias == 999.0)


# [ROBUST] The row constructor does not keep a reference to the caller's
# full_weight or full_bias.
def test_row_constructor_does_not_alias_caller_arrays():
    in_features, out_features, world_size, full_weight, full_bias, _ = _row_case()
    layer = RowParallelLinear(
        in_features,
        out_features,
        world_size,
        0,
        full_weight=full_weight,
        full_bias=full_bias,
    )
    full_weight[:] = 999.0
    full_bias[:] = 999.0
    assert not np.all(layer.weight == 999.0)
    assert not np.all(layer.bias == 999.0)


# [ROBUST] The column output is float32.
def test_column_output_dtype_is_float32():
    layer = ColumnParallelLinear(3, 4, 2, 0, seed=0)
    y = layer.forward(np.zeros((2, 3), dtype=np.float32))
    assert y.dtype == np.float32


# [ROBUST] The row output is float32.
def test_row_output_dtype_is_float32():
    layer = RowParallelLinear(4, 2, 2, 0, seed=0)
    z = layer.forward(np.zeros((2, 2), dtype=np.float32))
    assert z.dtype == np.float32


# [ROBUST] The All-Reduce output is float32.
def test_all_reduce_sum_output_dtype_is_float32():
    a = np.zeros((2, 2), dtype=np.float32)
    out = RowParallelLinear.all_reduce_sum([a], np.zeros(2, dtype=np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Sweep over several small valid configurations.
#
# A single fixed case can hide a bug that only appears at other
# dimensions. This sweeps a small grid.
# ---------------------------------------------------------------------------


# [ROBUST] Numerical equivalence must hold across the tested combinations
# of world_size, in_features, and out_features.
def test_column_and_row_match_dense_for_grid_of_small_configs():
    rng = np.random.default_rng(7)
    for world_size in (1, 2, 3):
        for in_features in (2, 4, 6):
            for out_features in (3, 6, 9):
                if out_features % world_size != 0:
                    continue
                batch = 2
                fw = rng.standard_normal((in_features, out_features)).astype(np.float32)
                fb = rng.standard_normal(out_features).astype(np.float32)
                x = rng.standard_normal((batch, in_features)).astype(np.float32)

                col = _make_column_shards(in_features, out_features, world_size, fw, fb)
                got = np.concatenate([s.forward(x) for s in col], axis=1)
                np.testing.assert_allclose(got, x @ fw + fb, rtol=1e-5, atol=1e-6)

                if in_features % world_size != 0:
                    continue
                local_in = in_features // world_size
                x_shards = [
                    x[:, r * local_in : (r + 1) * local_in] for r in range(world_size)
                ]
                row = _make_row_shards(in_features, out_features, world_size, fw, fb)
                partials = [row[r].forward(x_shards[r]) for r in range(world_size)]
                got = RowParallelLinear.all_reduce_sum(partials, fb)
                np.testing.assert_allclose(got, x @ fw + fb, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Hypothesis property tests (only run if Hypothesis is installed).
#
# The properties below describe the same math as the fixed tests, but
# over generated dimensions.
#
# PROPERTY: for any valid (batch >= 1, in_features > 0, out_features > 0,
# world_size > 0) with out_features % world_size == 0, and finite float32
# X, W, b, the concatenated column-parallel output equals X @ W + b within
# float32 tolerance.
# PROPERTY: for any valid (batch >= 1, in_features > 0, out_features > 0,
# world_size > 0) with in_features % world_size == 0, the row-parallel
# All-Reduce output equals X @ W + b within float32 tolerance.
# PROPERTY: concatenating all column shards along axis 1 reconstructs W;
# concatenating all row shards along axis 0 reconstructs W.
# ---------------------------------------------------------------------------

try:
    from hypothesis import assume, given, settings  # type: ignore[import-not-found]
    from hypothesis import strategies as st  # type: ignore[import-not-found]

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


if HYPOTHESIS_AVAILABLE:
    SMALL_DIM = st.integers(min_value=1, max_value=4)
    SMALL_WS = st.integers(min_value=1, max_value=2)

    @settings(max_examples=20, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=3),
        in_features=SMALL_DIM,
        out_features=SMALL_DIM,
        world_size=SMALL_WS,
    )
    # [ROBUST] Column equivalence over generated cases.
    def test_property_column_matches_dense(
        batch, in_features, out_features, world_size
    ):
        assume(out_features % world_size == 0)
        rng = np.random.default_rng(0)
        fw = rng.standard_normal((in_features, out_features)).astype(np.float32)
        fb = rng.standard_normal(out_features).astype(np.float32)
        x = rng.standard_normal((batch, in_features)).astype(np.float32)
        outputs = [
            ColumnParallelLinear(
                in_features,
                out_features,
                world_size,
                r,
                full_weight=fw,
                full_bias=fb,
            ).forward(x)
            for r in range(world_size)
        ]
        got = np.concatenate(outputs, axis=1)
        np.testing.assert_allclose(got, x @ fw + fb, rtol=1e-5, atol=1e-6)

    @settings(max_examples=20, deadline=None)
    @given(
        batch=st.integers(min_value=1, max_value=3),
        in_features=SMALL_DIM,
        out_features=SMALL_DIM,
        world_size=SMALL_WS,
    )
    # [ROBUST] Row equivalence over generated cases.
    def test_property_row_matches_dense(batch, in_features, out_features, world_size):
        assume(in_features % world_size == 0)
        rng = np.random.default_rng(0)
        fw = rng.standard_normal((in_features, out_features)).astype(np.float32)
        fb = rng.standard_normal(out_features).astype(np.float32)
        x = rng.standard_normal((batch, in_features)).astype(np.float32)
        local_in = in_features // world_size
        x_shards = [x[:, r * local_in : (r + 1) * local_in] for r in range(world_size)]
        row = [
            RowParallelLinear(
                in_features,
                out_features,
                world_size,
                r,
                full_weight=fw,
                full_bias=fb,
            )
            for r in range(world_size)
        ]
        partials = [row[r].forward(x_shards[r]) for r in range(world_size)]
        got = RowParallelLinear.all_reduce_sum(partials, fb)
        np.testing.assert_allclose(got, x @ fw + fb, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main(["-v", __file__])


# Not tested, and why:
#
# - Real multi-GPU execution. This is a single-process NumPy simulation.
# - Backward pass, autograd, or optimizers. Forward only.
# - Full transformer blocks (attention + MLP wiring). Out of scope.
# - Non-float32 dtypes. The contract is float32-only; other dtypes raise.
# - Subnormal float32 values. Overflow is tested, subnormals are not.
# - Communication latency, bandwidth, overlap, or collective failure.
#   The All-Reduce here is a local sum, not a distributed collective.
# - Inputs with more than 2 dimensions. The contract is (batch, features).
# - GPU placement. Not applicable in NumPy.
# - A bias argument that is not an ndarray (a list, for example). The
#   validator rejects it as a TypeError, but no test exercises that path.
# - A boolean seed (`seed=True`). `_require_int_seed` rejects it, but no
#   test exercises that path.
# - Empirical mutation testing. The tests here are predicted to catch the
#   named mutations, but no automated mutation run was performed. That is
#   the next step if the claims need to be turned into measurements.
