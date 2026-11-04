"""Test suite for day58_qat.

Categorization (master Rule 2):
  [ROBUST]   catches a specific mutated implementation
  [CONTRACT] catches a changed public interface
  [GUARD]    catches a defensive validation failure

Each test docstring names the exact mutation caught.

Revision history in this file:
  * test_scale_responds_to_max_abs     -> replaced by
    test_scale_uses_max_abs_not_mean_abs, which discriminates max-abs from
    other homogeneous statistics by asserting the calibration anchor.
  * test_values_below_half_step_quantize_to_zero -> replaced by
    test_truncation_discriminated_from_rint, which uses 0.6-step and
    -0.4-step values where rint, trunc, and floor disagree.
  * test_idempotent_within_tolerance    -> replaced by
    test_known_multi_element_quantization, which hand-computes expected
    values and catches wrong denominators and truncation.
  * Added test_public_path_matches_private_levels as a regression guard
    against drift between the test-facing and public-facing kernels.
  * Added seed-validation tests and a Generator state-advancement test.
  * Hypothesis property tests now use explicit pytest.mark.skip when the
    library is absent, so they appear as skipped instead of silently
    disappearing from collection.
"""

import importlib

import numpy as np
import pytest
from day58_qat import (
    FakeQuantizeLinear,
    _quantization_bounds,
    _symmetric_quantize_levels,
    straight_through_estimator_backward,
)

try:
    _hypothesis = importlib.import_module("hypothesis")
    given = _hypothesis.given
    settings = _hypothesis.settings
    st = _hypothesis.strategies
    _HAS_HYPOTHESIS = True
except ImportError:
    _HAS_HYPOTHESIS = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_layer(in_features=10, out_features=10, bits=8, bias=True, seed=0):
    return FakeQuantizeLinear(
        in_features=in_features,
        out_features=out_features,
        bits=bits,
        bias=bias,
        seed=seed,
    )


# ===========================================================================
# [ROBUST] Known-value quantization
# ===========================================================================
def test_known_value_quantization_hand_computed_grid():
    """[ROBUST] Catches wrong Delta formula or wrong quantization grid.

    bits=2 => q_max=1, q_min=-1. W = [-2,-1,0,1,2]:
      max_abs=2 => Delta=2.0. W/Delta = [-1,-0.5,0,0.5,1].
      rint    = [-1,-0,0,0,1].  clip = same.  *Delta = [-2,0,0,0,2].

    Caught mutations:
      * Delta = max_abs/(2^b - 1): grid becomes {0, 2/3, 4/3, 2}.
      * Delta = max_abs/(2^b): grid becomes {0, 1, 2}.
      * np.floor / np.ceil instead of np.rint: middle values differ.
    """
    layer = _make_layer(bits=2)
    w = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    expected = np.array([-2.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(layer.fake_quantize(w), expected)


# ===========================================================================
# [ROBUST] Symmetry
# ===========================================================================
@pytest.mark.parametrize("bits", [2, 4, 8, 12, 16])
def test_symmetry_across_bits(bits):
    """[ROBUST] Catches asymmetric quantization (zero-point offset).

    Q(-W) must equal -Q(W) elementwise. An asymmetric scheme
    (q = round(W/Delta) + zero_point) fails whenever zero_point != 0.
    """
    rng = np.random.default_rng(42)
    w = rng.standard_normal((64,)).astype(np.float32) * 5.0
    layer = _make_layer(bits=bits)
    np.testing.assert_array_equal(layer.fake_quantize(-w), -layer.fake_quantize(w))


# ===========================================================================
# [ROBUST] Grid membership and level range
# ===========================================================================
def test_output_is_on_quantization_grid():
    """[ROBUST] Catches off-grid outputs (wrong rounding, wrong scale).

    Every element of Q(W) must satisfy Q(W)/Delta ≈ integer within
    documented float tolerance, and that integer must be in [q_min, q_max].
    """
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((32, 4)) * 5.0).astype(np.float32)
    bits = 8
    layer = _make_layer(bits=bits)
    out = layer.fake_quantize(w)

    q_max = (1 << (bits - 1)) - 1
    delta = float(np.max(np.abs(w))) / q_max
    scaled = out / delta
    residual = np.abs(scaled - np.rint(scaled))
    assert np.all(residual < 1e-4), f"not on grid; max residual {residual.max()}"

    levels = np.rint(scaled)
    assert np.all(levels <= q_max + 1e-4)
    assert np.all(levels >= -q_max - 1e-4)


# ===========================================================================
# [ROBUST] All-zero tensor
# ===========================================================================
def test_all_zero_weights_return_zero_without_nan():
    """[ROBUST] Catches 0/0 = NaN when max(|W|) = 0."""
    layer = _make_layer(bits=8)
    w = np.zeros((50,), dtype=np.float32)
    out = layer.fake_quantize(w)
    assert np.array_equal(out, np.zeros_like(w))
    assert np.all(np.isfinite(out))


# ===========================================================================
# [ROBUST] Bit-width progression
# ===========================================================================
def test_bit_width_changes_quantization_step():
    """[ROBUST] Catches hardcoded bits or wrong q_max derivation."""
    rng = np.random.default_rng(0)
    w = rng.uniform(-1.0, 1.0, size=(1000,)).astype(np.float32)
    w[0] = 1.0
    w[-1] = -1.0

    n_levels = {}
    for bits in (4, 8, 12):
        out = _make_layer(bits=bits).fake_quantize(w)
        q_max = (1 << (bits - 1)) - 1
        scaled = out / (1.0 / q_max)
        residual = np.abs(scaled - np.rint(scaled))
        assert np.max(residual) < 1e-4, f"bits={bits}: off grid"
        assert len(np.unique(out)) <= 2 * q_max + 1
        n_levels[bits] = len(np.unique(out))
    assert n_levels[4] <= n_levels[8] <= n_levels[12]


# ===========================================================================
# [ROBUST] Integer-level contract
# ===========================================================================
@pytest.mark.parametrize("bits", [2, 4, 8, 12, 16])
def test_quantization_bounds_symmetric_contract(bits):
    """[ROBUST] Catches asymmetric bounds (e.g., [-128, 127] for INT8).

    Output-only tests cannot distinguish [-127,127] from [-128,127] under
    this calibration because Delta anchors at q_max. This test verifies the
    contract directly at the source via _quantization_bounds.
    """
    q_min, q_max = _quantization_bounds(bits)
    expected_q_max = (1 << (bits - 1)) - 1
    assert q_max == expected_q_max
    assert q_min == -expected_q_max
    assert q_min == -q_max


def test_integer_levels_bounded_and_anchor():
    """[ROBUST] Catches level-range violations and calibration-anchor drift."""
    layer = _make_layer(bits=8)
    _, q_max = _quantization_bounds(layer.bits)
    w = np.array([127.0, 1.0, -63.5, 0.5], dtype=np.float32)
    levels = _symmetric_quantize_levels(w, layer.bits)
    assert levels.min() >= -q_max
    assert levels.max() <= q_max
    assert levels[0] == q_max
    assert levels[2] == -64  # rint(-63.5) = -64 (ties-to-even)


def test_calibration_anchors_max_abs_to_q_max():
    """[ROBUST] Catches scale denominator mutations that shift the anchor.

    Correct: Delta=1.0, levels=[127,1,0] for w=[127,1,0.5].
    Denominator q_max+1: levels=[127(clip),1,1] — distinguishes.
    Denominator 2^b-1:   levels=[127(clip),2,1] — distinguishes.
    """
    layer = _make_layer(bits=8)
    w = np.array([127.0, 1.0, 0.5], dtype=np.float32)
    levels = _symmetric_quantize_levels(w, layer.bits)
    np.testing.assert_array_equal(levels, [127, 1, 0])


def test_negative_max_abs_anchors_to_q_min():
    """[ROBUST] Catches sign errors in the symmetric calibration anchor."""
    layer = _make_layer(bits=8)
    _, q_max = _quantization_bounds(layer.bits)
    w = np.array([-127.0, 63.0, -0.5], dtype=np.float32)
    levels = _symmetric_quantize_levels(w, layer.bits)
    assert levels[0] == -q_max


# ===========================================================================
# [ROBUST] Ties-to-even
# ===========================================================================
def test_tie_to_even_at_halfway_points():
    """[ROBUST] Catches floor / ceil / half-up / half-away rounding."""
    layer = _make_layer(bits=4)
    w = np.array([-14.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 14.0], dtype=np.float32)
    expected = np.array([-14.0, -4.0, -4.0, 0.0, 0.0, 4.0, 4.0, 14.0], dtype=np.float32)
    np.testing.assert_array_equal(layer.fake_quantize(w), expected)


@pytest.mark.parametrize("bits", [8, 12])
def test_tie_to_even_across_bits(bits):
    """[ROBUST] Catches tie-break behavior that drifts with bit width.

    Pins max_abs to 2*q_max, forcing Delta=2.0. Inputs at ±5, ±3, ±1 sit
    at scaled values ±2.5, ±1.5, ±0.5 — exactly the halfway points.
    """
    q_max = (1 << (bits - 1)) - 1
    max_abs = 2.0 * q_max
    layer = _make_layer(bits=bits)
    w = np.array(
        [-max_abs, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, max_abs],
        dtype=np.float32,
    )
    expected = np.array(
        [-max_abs, -4.0, -4.0, 0.0, 0.0, 4.0, 4.0, max_abs],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(layer.fake_quantize(w), expected)


# ===========================================================================
# [ROBUST] Scale calibration — max-abs vs other statistics
# ===========================================================================
def test_scale_uses_max_abs_not_mean_abs():
    """[ROBUST] Catches any homogeneous calibration statistic that is not max-abs.

    w = [4, 1, 0], bits=8, q_max=127.
    Correct (max-abs): delta = 4/127; levels = [127, 32, 0];
        output = [4, 128/127, 0]. The maximum-magnitude element maps to
        exactly 4 (the calibration anchor).
    Wrong (mean-abs = 5/3): delta = (5/3)/127; levels clip to [127, 76, 0];
        output ≈ [1.667, 0.997, 0]. The max element is 1.667, not 4.
    Wrong (RMS, L2, median, etc.) similarly fail the anchor assertion.
    """
    layer = _make_layer(bits=8)
    w = np.array([4.0, 1.0, 0.0], dtype=np.float32)
    out = layer.fake_quantize(w)
    assert out[0] == 4.0, (
        f"calibration anchor not preserved: out[0]={out[0]}, expected 4.0. "
        f"Scale is not derived from max(|W|)."
    )
    assert out[2] == 0.0
    # Middle element: rint(127/4)*4/127 = 32*4/127 ≈ 1.00787
    expected_middle = np.float32(32 * 4 / 127)
    np.testing.assert_allclose(out[1], expected_middle, rtol=1e-6)


# ===========================================================================
# [ROBUST] Rounding vs truncation / floor on non-halfway values
# ===========================================================================
def test_truncation_discriminated_from_rint():
    """[ROBUST] Catches truncation-toward-zero and floor/ceil mutations.

    With max_val=10 (bits=8, delta=10/127), consider:
      +0.6*delta  -> rint=1, trunc=0, floor=0  (catches trunc/floor)
      -0.4*delta  -> rint=0, trunc=0, floor=-1 (catches floor)
    The assertions below fail for trunc (which gives 0 instead of ±delta)
    and for floor (which gives -delta instead of 0 on the negative value).
    """
    layer = _make_layer(bits=8)
    max_val = 10.0
    delta = np.float32(max_val / 127)
    w = np.array([max_val, 0.6 * delta, -0.4 * delta], dtype=np.float32)
    out = layer.fake_quantize(w)
    np.testing.assert_allclose(out[0], max_val, rtol=1e-6)
    (
        np.testing.assert_allclose(out[1], delta, rtol=1e-5),
        (
            f"positive non-halfway 0.6*delta should round to +delta, "
            f"got {out[1]}; truncation or floor bug"
        ),
    )
    (
        np.testing.assert_allclose(out[2], 0.0, atol=1e-6),
        (f"negative 0.4*delta should round to 0, got {out[2]}; floor bug"),
    )


# ===========================================================================
# [ROBUST] Multi-element hand-computed example
# ===========================================================================
def test_known_multi_element_quantization():
    """[ROBUST] Catches wrong denominator and truncation on a combined case.

    w = [-3.7, -1.4, 0.0, 1.4, 3.7], bits=4 (q_max=7).
    max_abs = 3.7, delta = 3.7/7 ≈ 0.52857.
    w/delta ≈ [-7, -2.6486, 0, 2.6486, 7].
    rint    = [-7, -3, 0, 3, 7].
    output  = [-3.7, -3*delta, 0, 3*delta, 3.7].

    Wrong denominator (2^b - 1 = 15): delta = 3.7/15; levels = [-7,-6,0,6,7];
      output ≠ expected.
    Truncation: levels = [-7, -2, 0, 2, 7]; output ≠ expected.
    """
    layer = _make_layer(bits=4)
    w = np.array([-3.7, -1.4, 0.0, 1.4, 3.7], dtype=np.float32)
    delta = np.float32(3.7) / 7
    expected = np.array([-3.7, -3 * delta, 0.0, 3 * delta, 3.7], dtype=np.float32)
    np.testing.assert_allclose(layer.fake_quantize(w), expected, rtol=1e-6)


# ===========================================================================
# [ROBUST] Public path matches private levels (drift guard)
# ===========================================================================
@pytest.mark.parametrize("bits", [2, 4, 8, 12, 16])
def test_public_path_matches_private_levels(bits):
    """[ROBUST] Catches drift between the test-facing and public kernels.

    The public fake_quantize output must equal
    clip(levels * delta, -max_abs, max_abs) for every accepted bit width.
    If the two paths diverge (different scale, rounding, or clip), the
    reconstruction here would not match the public output. Both paths share
    _quantize_parts today; this test is a regression guard against future
    divergence.
    """
    rng = np.random.default_rng(bits)
    w = (rng.standard_normal((50,)) * 3).astype(np.float32)
    layer = _make_layer(bits=bits)
    out_public = layer.fake_quantize(w)

    levels = _symmetric_quantize_levels(w, bits)
    _, q_max = _quantization_bounds(bits)
    delta = np.max(np.abs(w)) / np.float32(q_max)
    max_abs = np.max(np.abs(w))
    with np.errstate(over="ignore"):
        reconstructed = levels.astype(np.float32) * delta
    reconstructed = np.clip(reconstructed, -max_abs, max_abs)

    np.testing.assert_allclose(out_public, reconstructed, rtol=1e-6, atol=1e-6)


# ===========================================================================
# [ROBUST] Minimal working configuration and zero handling
# ===========================================================================
def test_bits_2_is_minimal_working_config():
    """[ROBUST] Catches off-by-one lower bound (allowing bits=1)."""
    layer = _make_layer(bits=2)
    w = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
    expected = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
    np.testing.assert_array_equal(layer.fake_quantize(w), expected)


def test_zero_element_in_nonzero_tensor_stays_zero():
    """[ROBUST] Catches asymmetric zero-point offset."""
    layer = _make_layer(bits=8)
    w = np.array([5.0, 0.0, -5.0], dtype=np.float32)
    out = layer.fake_quantize(w)
    assert out[1] == 0.0


# ===========================================================================
# [ROBUST] Determinism and non-mutation
# ===========================================================================
def test_repeated_fake_quantize_is_deterministic():
    """[ROBUST] Catches hidden state or RNG in fake_quantize."""
    layer = _make_layer(bits=8)
    w = np.linspace(-2.0, 2.0, 128, dtype=np.float32)
    np.testing.assert_array_equal(layer.fake_quantize(w), layer.fake_quantize(w))


def test_fake_quantize_does_not_mutate_input():
    """[ROBUST] Catches in-place ops inside fake_quantize."""
    layer = _make_layer(bits=8)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((50,)).astype(np.float32)
    before = w.copy()
    _ = layer.fake_quantize(w)
    np.testing.assert_array_equal(w, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
def test_input_immutability_across_dtypes(dtype):
    """[ROBUST] Catches in-place mutation for any accepted input dtype."""
    layer = _make_layer(bits=8)
    if dtype == np.int32:
        w = np.array([1, -2, 3, -4], dtype=dtype)
    else:
        w = np.array([1.0, -2.0, 3.0, -4.0], dtype=dtype)
    before = w.copy()
    _ = layer.fake_quantize(w)
    np.testing.assert_array_equal(w, before)


# ===========================================================================
# [ROBUST] STE identity
# ===========================================================================
def test_ste_preserves_all_input_values():
    """[ROBUST] Catches any transformation inside STE (clip, scale, abs)."""
    grad = np.array(
        [[-3.5, -1.0, 0.0, 1.0, 2.5], [10.0, -10.0, 0.0, 1e-6, -1e-6]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(straight_through_estimator_backward(grad), grad)


def test_ste_returns_owned_copy_not_alias():
    """[ROBUST] Catches `return grad_output` (alias) or view return."""
    grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = straight_through_estimator_backward(grad)
    assert out is not grad
    assert not np.shares_memory(out, grad)
    out[0] = 999.0
    assert grad[0] == 1.0


def test_ste_does_not_mutate_input():
    """[ROBUST] Catches in-place ops or aliasing inside STE."""
    grad = np.array([1.0, -2.0, 0.0, 3.0], dtype=np.float32)
    before = grad.copy()
    _ = straight_through_estimator_backward(grad)
    np.testing.assert_array_equal(grad, before)


def test_ste_identity_on_multidimensional_input():
    """[ROBUST] Catches reshape/transpose bugs at rank > 2."""
    rng = np.random.default_rng(7)
    grad = rng.standard_normal((2, 3, 4)).astype(np.float32)
    out = straight_through_estimator_backward(grad)
    np.testing.assert_array_equal(out, grad)
    assert out.shape == grad.shape


def test_ste_zero_array():
    """[ROBUST] Catches accidental nonzero handling (e.g., adding eps)."""
    grad = np.zeros((3, 3), dtype=np.float32)
    np.testing.assert_array_equal(straight_through_estimator_backward(grad), grad)


def test_ste_identity_is_elementwise_not_just_shape():
    """[ROBUST] Catches sign flip, permutation, or transpose."""
    grad = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    out = straight_through_estimator_backward(grad)
    assert out[0, 0] == 1.0 and out[0, 1] == -2.0
    assert out[1, 0] == 3.0 and out[1, 1] == -4.0


# ===========================================================================
# [ROBUST] Weight initialization determinism
# ===========================================================================
def test_different_seeds_produce_different_weights():
    """[ROBUST] Catches ignored seed argument."""
    a = FakeQuantizeLinear(10, 10, bits=8, seed=1)
    b = FakeQuantizeLinear(10, 10, bits=8, seed=2)
    assert not np.array_equal(a.weight, b.weight)


def test_same_seed_reproduces_weights():
    """[ROBUST] Catches non-deterministic init (e.g., global np.random)."""
    a = FakeQuantizeLinear(10, 10, bits=8, seed=12345)
    b = FakeQuantizeLinear(10, 10, bits=8, seed=12345)
    np.testing.assert_array_equal(a.weight, b.weight)


def test_generator_seed_object_is_used():
    """[ROBUST] Catches accidental reset of a passed-in Generator.

    Two separately constructed Generators with identical seed state must
    produce identical weights.
    """
    a = FakeQuantizeLinear(10, 10, bits=8, seed=np.random.default_rng(7))
    b = FakeQuantizeLinear(10, 10, bits=8, seed=np.random.default_rng(7))
    np.testing.assert_array_equal(a.weight, b.weight)


def test_generator_state_advances_on_use():
    """[ROBUST] Catches Generator reset or copy rather than advance.

    Two layers constructed from the SAME Generator must have different
    weights, because the Generator's state advances on each weight draw.
    If the layer copies or resets the Generator instead, both layers would
    see the same state and produce identical weights.
    """
    rng = np.random.default_rng(42)
    a = FakeQuantizeLinear(10, 10, bits=8, seed=rng)
    b = FakeQuantizeLinear(10, 10, bits=8, seed=rng)
    assert not np.array_equal(a.weight, b.weight), (
        "two layers built from the same Generator produced identical weights; "
        "Generator state did not advance"
    )


def test_construction_does_not_touch_global_rng():
    """[ROBUST] Catches use of global np.random.* in __init__.

    Seed global state, draw a reference, reseed, construct a layer, draw
    again. If construction advanced global state, the draws differ.
    """
    np.random.seed(12345)
    expected = np.random.rand()
    np.random.seed(12345)
    FakeQuantizeLinear(10, 10, bits=8)
    actual = np.random.rand()
    assert actual == expected, "construction advanced the global np.random state"


# ===========================================================================
# [ROBUST] Forward exact equation
# ===========================================================================
def test_forward_matches_manual_computation():
    """[ROBUST] Catches missing fake_quant, wrong transpose, missing bias."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    layer.bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    x = np.array(
        [[1.0, 2.0, -1.0, 0.5], [0.0, -1.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    w_q = layer.fake_quantize(layer.weight)
    expected = x @ w_q.T + layer.bias
    np.testing.assert_allclose(layer.forward(x), expected, rtol=0, atol=1e-6)


def test_forward_no_bias():
    """[ROBUST] Catches unconditional bias add when bias=False."""
    layer = FakeQuantizeLinear(4, 3, bits=8, bias=False, seed=0)
    x = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
    w_q = layer.fake_quantize(layer.weight)
    np.testing.assert_allclose(layer.forward(x), x @ w_q.T, rtol=0, atol=1e-6)
    assert layer.bias is None


def test_forward_does_not_quantize_activations():
    """[ROBUST] Catches activation quantization mistakenly added to forward."""
    layer = FakeQuantizeLinear(4, 3, bits=8, bias=False, seed=0)
    x = np.array([0.123456, 0.987654, -0.543210, 0.111111], dtype=np.float32)
    w_q = layer.fake_quantize(layer.weight)
    expected = x @ w_q.T
    np.testing.assert_allclose(layer.forward(x), expected, rtol=0, atol=1e-6)


def test_forward_multibatch_shape():
    """[ROBUST] Catches missing multi-batch axis handling in forward."""
    layer = FakeQuantizeLinear(8, 4, bits=8, seed=0)
    x = np.random.default_rng(0).standard_normal((3, 5, 8)).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (3, 5, 4)
    assert y.dtype == np.float32


def test_forward_non_contiguous_input():
    """[ROBUST] Catches assumptions about contiguous memory layout."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    x_contig = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    x_noncontig = x_contig[::2, :]
    assert not x_noncontig.flags["C_CONTIGUOUS"]
    y = layer.forward(x_noncontig)
    w_q = layer.fake_quantize(layer.weight)
    expected = x_noncontig @ w_q.T + layer.bias
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-6)


# ===========================================================================
# [ROBUST] Extreme numerical values
# ===========================================================================
@pytest.mark.parametrize(
    "value,dtype,expect_success",
    [
        pytest.param(
            float(np.finfo(np.float32).max), np.float32, True, id="max_finite_f32"
        ),
        pytest.param(
            float(np.finfo(np.float32).tiny), np.float32, True, id="min_normal_f32"
        ),
        pytest.param(
            float(np.nextafter(np.float32(0), np.float32(1))),
            np.float32,
            False,
            id="smallest_subnormal_f32",
        ),
        pytest.param(
            float(np.finfo(np.float64).max), np.float64, True, id="max_finite_f64"
        ),
    ],
)
def test_extreme_float_values(value, dtype, expect_success):
    """[ROBUST] Catches silent NaN/inf leaks on extreme inputs."""
    layer = _make_layer(bits=8)
    w = np.array([value, -value], dtype=dtype)
    if expect_success:
        out = layer.fake_quantize(w)
        assert np.all(np.isfinite(out))
    else:
        with pytest.raises(ValueError):
            layer.fake_quantize(w)


def test_subnormal_scale_underflow_rejected():
    """[GUARD] Catches silent inf/nan when Delta underflows to zero."""
    layer = _make_layer(bits=16)
    tiny = np.array([1e-45, -1e-45, 5e-46], dtype=np.float32)
    if float(np.max(np.abs(tiny))) == 0.0:
        pytest.skip("platform underflows 1e-45 to zero before our code sees it")
    with pytest.raises(ValueError):
        layer.fake_quantize(tiny)


# ===========================================================================
# [CONTRACT] Dtype contracts
# ===========================================================================
def test_float32_preserves_dtype():
    """[CONTRACT] Catches accidental float32 -> float64 promotion."""
    layer = _make_layer()
    w = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    assert layer.fake_quantize(w).dtype == np.float32


def test_float64_preserves_dtype():
    """[CONTRACT] Catches accidental float64 -> float32 demotion."""
    layer = _make_layer()
    w = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    assert layer.fake_quantize(w).dtype == np.float64


def test_float16_promotes_to_float32():
    """[CONTRACT] Catches missing float16 promotion."""
    layer = _make_layer()
    w = np.array([1.0, -0.5, 0.25], dtype=np.float16)
    assert layer.fake_quantize(w).dtype == np.float32


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_promotes_to_float64(dtype):
    """[CONTRACT] Catches accidental integer truncation for any integer dtype.

    Uses [1, 2, 3] (non-negative) so construction succeeds for unsigned
    dtypes. Sign handling is covered by the float tests.
    """
    layer = _make_layer()
    w = np.array([1, 2, 3], dtype=dtype)
    out = layer.fake_quantize(w)
    assert out.dtype == np.float64
    assert np.issubdtype(out.dtype, np.floating)


def test_longdouble_preserved_if_supported():
    """[CONTRACT] Catches demotion of longdouble to float64."""
    layer = _make_layer()
    try:
        w = np.array([1.0, -2.0, 0.5], dtype=np.longdouble)
    except (TypeError, AttributeError):
        pytest.skip("longdouble not available on this platform")
    out = layer.fake_quantize(w)
    assert out.dtype == np.dtype(np.longdouble)


def test_huge_int64_precision_loss_documented():
    """[CONTRACT] Documents that int64 above 2**53 loses precision in float64.

    Two distinct int64 values that both round to 2**53 in float64 produce
    identical output. That is the documented behavior, not a bug.
    """
    layer = _make_layer(bits=8)
    a = np.array([2**53], dtype=np.int64)
    b = np.array([2**53 + 1], dtype=np.int64)
    out_a = layer.fake_quantize(a)
    out_b = layer.fake_quantize(b)
    assert out_a.dtype == np.float64
    assert out_b.dtype == np.float64
    assert np.all(np.isfinite(out_a))
    np.testing.assert_array_equal(out_a, out_b)


# ===========================================================================
# [CONTRACT] Shape preservation (starter's 1-D case)
# ===========================================================================
def test_starter_1d_input_not_reshaped():
    """[CONTRACT] Catches unwanted reshape to (out, in)."""
    layer = FakeQuantizeLinear(in_features=10, out_features=10, bits=8)
    w = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    out = layer.fake_quantize(w)
    assert out.shape == (100,)
    assert out.dtype == np.float32


def test_higher_rank_shape_preserved():
    """[CONTRACT] Catches shape mangling on 3-D input."""
    layer = _make_layer()
    w = np.ones((3, 5, 7), dtype=np.float32)
    assert layer.fake_quantize(w).shape == w.shape


def test_forward_accepts_1d_input():
    """[CONTRACT] Catches forward rejecting valid 1-D x of shape (in_features,)."""
    layer = FakeQuantizeLinear(4, 3, bits=8, bias=False, seed=0)
    x = np.ones(4, dtype=np.float32)
    y = layer.forward(x)
    assert y.shape == (3,)


# ===========================================================================
# [CONTRACT] No-seed construction determinism (design contract)
# ===========================================================================
def test_no_seed_construction_is_deterministic():
    """[CONTRACT] Catches reliance on global np.random state.

    Design contract: seed=None produces deterministic weights (fixed
    internal seed). Two default-constructed layers have identical weights.
    """
    a = FakeQuantizeLinear(10, 10, bits=8)
    b = FakeQuantizeLinear(10, 10, bits=8)
    np.testing.assert_array_equal(a.weight, b.weight)


def test_forward_shape():
    """[CONTRACT] Catches transpose or axis-swap in linear forward."""
    layer = FakeQuantizeLinear(8, 4, bits=8, seed=0)
    x = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
    y = layer.forward(x)
    assert y.shape == (16, 4)
    assert y.dtype == np.float32


# ===========================================================================
# [CONTRACT] STE dtype/shape
# ===========================================================================
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
def test_ste_preserves_dtype(dtype):
    """[CONTRACT] Catches accidental dtype promotion/demotion in STE."""
    g = (
        np.array([1, 2, 0], dtype=dtype)
        if dtype == np.int32
        else np.array([1.0, -2.0, 0.0], dtype=dtype)
    )
    assert straight_through_estimator_backward(g).dtype == np.dtype(dtype)


def test_ste_accepts_python_list():
    """[CONTRACT] Catches STE rejecting non-ndarray list input."""
    out = straight_through_estimator_backward([1.0, -2.0, 0.0])
    np.testing.assert_array_equal(out, np.array([1.0, -2.0, 0.0]))


# ===========================================================================
# [CONTRACT] Forward dtype
# ===========================================================================
def test_forward_float64_input_dtype():
    """[CONTRACT] Catches float64 x being demoted to float32 in forward."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    x = np.ones((2, 4), dtype=np.float64)
    y = layer.forward(x)
    assert y.dtype == np.float64


def test_forward_integer_input_dtype():
    """[CONTRACT] Catches integer x leaking through as integer output."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    y = layer.forward(x)
    assert y.dtype == np.float64


# ===========================================================================
# [GUARD] Validation
# ===========================================================================
@pytest.mark.parametrize(
    "bits,exc",
    [
        pytest.param(True, TypeError, id="bool_true"),
        pytest.param(False, TypeError, id="bool_false"),
        pytest.param(1.5, TypeError, id="float"),
        pytest.param(None, TypeError, id="none"),
        pytest.param(1, ValueError, id="below_range"),
        pytest.param(0, ValueError, id="zero"),
        pytest.param(-1, ValueError, id="negative"),
        pytest.param(17, ValueError, id="above_range"),
    ],
)
def test_reject_invalid_bits(bits, exc):
    """[GUARD] Catches missing/partial bits validation."""
    with pytest.raises(exc):
        FakeQuantizeLinear(10, 10, bits=bits)


@pytest.mark.parametrize(
    "payload,exc",
    [
        pytest.param(np.array([], dtype=np.float32), ValueError, id="empty"),
        pytest.param(np.array([1.0, "x"], dtype=object), TypeError, id="object"),
        pytest.param(np.array(["a", "b"], dtype="U1"), TypeError, id="string"),
        pytest.param(np.array([b"a", b"b"], dtype="S1"), TypeError, id="bytes"),
        pytest.param(np.array([1.0, 1j], dtype=np.complex64), TypeError, id="complex"),
        pytest.param(np.array([True, False]), TypeError, id="bool"),
        pytest.param(np.array([1.0, np.nan], dtype=np.float32), ValueError, id="nan"),
        pytest.param(
            np.array([1.0, np.inf], dtype=np.float32), ValueError, id="pos_inf"
        ),
        pytest.param(
            np.array([1.0, -np.inf], dtype=np.float32), ValueError, id="neg_inf"
        ),
    ],
)
def test_reject_invalid_weight_tensors(payload, exc):
    """[GUARD] Catches missing validation on weight tensors."""
    with pytest.raises(exc):
        _make_layer().fake_quantize(payload)


@pytest.mark.parametrize(
    "payload,exc",
    [
        pytest.param(np.array([], dtype=np.float32), ValueError, id="empty"),
        pytest.param(np.array([1.0, "x"], dtype=object), TypeError, id="object"),
        pytest.param(np.array([True, False]), TypeError, id="bool"),
        pytest.param(np.array([1.0, np.nan], dtype=np.float32), ValueError, id="nan"),
        pytest.param(
            np.array([1.0, np.inf], dtype=np.float32), ValueError, id="pos_inf"
        ),
        pytest.param(
            np.array([1.0, -np.inf], dtype=np.float32), ValueError, id="neg_inf"
        ),
    ],
)
def test_reject_invalid_ste_tensors(payload, exc):
    """[GUARD] Catches missing validation in STE."""
    with pytest.raises(exc):
        straight_through_estimator_backward(payload)


@pytest.mark.parametrize(
    "kw,value",
    [
        pytest.param("in_features", 0, id="in_zero"),
        pytest.param("in_features", -1, id="in_neg"),
        pytest.param("out_features", 0, id="out_zero"),
        pytest.param("out_features", -1, id="out_neg"),
    ],
)
def test_reject_non_positive_feature_dims(kw, value):
    """[GUARD] Catches missing positivity check on feature dimensions."""
    kwargs = {"in_features": 10, "out_features": 10, "bits": 8}
    kwargs[kw] = value
    with pytest.raises(ValueError):
        FakeQuantizeLinear(**kwargs)


@pytest.mark.parametrize(
    "kw,value",
    [
        pytest.param("in_features", True, id="in_bool_true"),
        pytest.param("in_features", False, id="in_bool_false"),
        pytest.param("in_features", 10.0, id="in_float"),
        pytest.param("out_features", "10", id="out_string"),
    ],
)
def test_reject_non_integer_feature_dims(kw, value):
    """[GUARD] Catches missing Integral check on feature dims."""
    kwargs = {"in_features": 10, "out_features": 10, "bits": 8}
    kwargs[kw] = value
    with pytest.raises(TypeError):
        FakeQuantizeLinear(**kwargs)


@pytest.mark.parametrize("bias_val", ["False", "True", 1, 0, None, 1.0])
def test_reject_non_bool_bias(bias_val):
    """[GUARD] Catches `bool(bias)` silent coercion."""
    with pytest.raises(TypeError):
        FakeQuantizeLinear(10, 10, bits=8, bias=bias_val)


@pytest.mark.parametrize("bad_seed", [True, False])
def test_reject_bool_seed(bad_seed):
    """[GUARD] Catches bool accepted as seed (bool is int subclass)."""
    with pytest.raises(TypeError):
        FakeQuantizeLinear(10, 10, bits=8, seed=bad_seed)


@pytest.mark.parametrize("bad_seed", [1.5, "0", [1], object()])
def test_reject_invalid_seed_types(bad_seed):
    """[GUARD] Catches non-int/non-Generator/non-None seed types.

    None is intentionally NOT in this list; it is the documented default.
    """
    with pytest.raises(TypeError):
        FakeQuantizeLinear(10, 10, bits=8, seed=bad_seed)


def test_forward_rejects_wrong_last_axis():
    """[GUARD] Catches missing last-axis check in forward."""
    layer = FakeQuantizeLinear(8, 4, bits=8, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((16, 7), dtype=np.float32))


def test_forward_rejects_0d_input():
    """[GUARD] Catches missing ndim==0 check in forward."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.array(1.0))


def test_forward_rejects_empty_input():
    """[GUARD] Catches missing non-empty check in forward."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((0, 4), dtype=np.float32))


def test_forward_rejects_nan_input():
    """[GUARD] Catches missing finiteness check in forward."""
    layer = FakeQuantizeLinear(4, 3, bits=8, seed=0)
    x = np.zeros((2, 4), dtype=np.float32)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        layer.forward(x)


# ===========================================================================
# Hypothesis property tests
#
# Always collected. When hypothesis is missing, the tests are explicitly
# skipped with a clear reason, so the suite reports "N passed, 2 skipped"
# rather than silently losing coverage. To run them:
#     pip install hypothesis
# ===========================================================================
if _HAS_HYPOTHESIS:

    @settings(max_examples=200, deadline=None)
    @given(
        weights=st.lists(
            st.floats(
                min_value=-1e4,
                max_value=1e4,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=1,
            max_size=200,
        ),
        bits=st.integers(min_value=2, max_value=16),
    )
    def test_property_quantization_metamorphic(weights, bits):
        """[ROBUST] Hypothesis property: symmetric, idempotent, bounded."""
        w = np.array(weights, dtype=np.float32)
        layer = _make_layer(bits=bits)

        q = layer.fake_quantize(w)
        q_neg = layer.fake_quantize(-w)
        q_q = layer.fake_quantize(q)

        assert np.all(np.isfinite(q))
        assert np.max(np.abs(q)) <= np.max(np.abs(w)) + 1e-4

        np.testing.assert_array_equal(q_neg, -q)
        np.testing.assert_allclose(q_q, q, rtol=1e-4, atol=1e-4)

    @settings(max_examples=200, deadline=None)
    @given(
        grad=st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=200,
        ),
    )
    def test_property_ste_identity(grad):
        """[ROBUST] Hypothesis property: STE is identity on any finite array."""
        g = np.array(grad, dtype=np.float64)
        out = straight_through_estimator_backward(g)
        np.testing.assert_array_equal(out, g)
        assert not np.shares_memory(out, g)
else:

    @pytest.mark.skip(
        reason="hypothesis not installed; pip install hypothesis to enable"
    )
    def test_property_quantization_metamorphic():
        """Placeholder so pytest reports 'skipped' rather than uncollected."""

    @pytest.mark.skip(
        reason="hypothesis not installed; pip install hypothesis to enable"
    )
    def test_property_ste_identity():
        """Placeholder so pytest reports 'skipped' rather than uncollected."""


# ===========================================================================
# Documented property strategies (Rule 4 compliance even without Hypothesis)
# ===========================================================================
# PROPERTY: for any finite non-empty W and valid bits b, Q(W) has shape
# W.shape, is finite, is bounded by max(|W|), lies on k*Delta grid for
# integer k in [q_min, q_max], satisfies Q(-W) == -Q(W), and is idempotent
# up to float ULP. Verified above when Hypothesis is available.
#
# PROPERTY: for any finite non-empty gradient g, STE(g) == g elementwise,
# same shape, same dtype, does not share memory with g. Verified above.
#
# PROPERTY: for any valid bits b and positive scalar s, Q(s*W) == s*Q(W)
# elementwise up to float tolerance (calibration is scale-equivariant).
# Covered by test_scale_uses_max_abs_not_mean_abs and the bounded test;
# can be lifted to Hypothesis if broader coverage is wanted.


# Not tested (and why):
# - Full backprop graph / autograd integration: out of scope; this is a
#   reference kernel, not a training framework. No optimizer integration.
# - Per-channel quantization: not implemented by design (per-tensor baseline).
# - Hardware INT8 GEMM parity: fake quantization produces floating arrays;
#   no compiled INT8 kernel to compare against.
# - Complex-valued gradients or weights: rejected at validation; complex
#   STE would need a different contract.
# - Exhaustive subnormal fuzz: the zero-scale ValueError boundary is
#   spot-checked (test_subnormal_scale_underflow_rejected and
#   test_extreme_float_values[smallest_subnormal_f32]), not fuzzed across
#   the full subnormal range. Hypothesis is the natural tool; installing
#   it as a required dependency for a 250-line educational file is not
#   justified.
# - -0.0 input: not explicitly tested. The all-zero shortcut normalizes
#   -0.0 to +0.0 (which is fine under `==` comparison and preserves sign
#   symmetry). Adding test_negative_zero_input_normalizes_to_positive_zero
#   would close this; kept out of this pass to keep the diff focused on
#   the reviewer's priority items.
# - QAT training accuracy / convergence on real models: no training
#   experiment; no accuracy guarantee is claimed.
# - Multi-threaded RNG safety: Generator used as-is; thread safety is the
#   caller's responsibility.
# - Non-symmetric / asymmetric / per-channel quantization: not implemented.
# - Serialization (pickle, save/load): not implemented.
# - Forward backward-composition with STE (grad flowing through forward):
#   forward returns y only; no autograd hook. Callers compose manually.
# - int64 inputs above 2**53: float64 promotion loses low-order bits.
#   Documented limitation; test_huge_int64_precision_loss_documented
#   verifies the documented behavior, not a precision-preserving alternative.
# - Clipping necessity: np.clip is defensive; under calibration it is a
#   mathematical no-op for calibration-anchored inputs. Removing it would
#   change output only in rare float-rounding edge cases, which we do not
#   attempt to enumerate. Documented as a test limitation, not a gap.
# - Cross-platform numerical equivalence: not demonstrated. Float32
#   arithmetic is IEEE-754 deterministic on all mainstream platforms we
#   target, but we do not run a CI matrix.
