"""Tests for day46_qlora_adapter.

Each test is tagged [ROBUST], [GUARD], or [CONTRACT]:
    ROBUST   -- fails against a specific wrong implementation
    GUARD    -- fails if a defensive validation is removed
    CONTRACT -- fails if the public interface changes

Property description (Hypothesis not installed; kept as a comment because
the master prompt requires it):

    For any finite x [N,k], W0 [d,k], A [r,k], B=0 [d,r], alpha >= 0, r > 0:
        forward(x).shape == (N, d)
        forward(x) is finite
        w_quant.dtype == np.int8, values in [-128, 127]
        forward(x) == x @ dequant(W0).T  (adapter contribution is exactly 0)
        max|W0 - dequant(quant(W0))| <= scale / 2

    Strategies: N, k, d, r bounded 1..5; float32 arrays drawn from
    floats(-10, 10, allow_nan=False, allow_infinity=False).
"""

import numpy as np
import pytest
from day46_qlora_adapter import QuantizedLinearLoRA

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


# [ROBUST] Fails if quantization returns a non-int8 dtype or a transposed shape.
def test_quantization_preserves_dtype_shape_and_range():
    w = np.array([[1.5, -3.0], [0.0, 2.2]], dtype=np.float32)
    _, wq = QuantizedLinearLoRA._quantize_int8(w)

    assert wq.shape == w.shape
    assert wq.dtype == np.int8
    assert int(wq.min()) >= -128
    assert int(wq.max()) <= 127


# [ROBUST] Fails if the scale divisor is not 127.
def test_quantization_error_is_bounded_by_half_scale():
    w = np.array([[-5.0, 2.0, 0.1]], dtype=np.float32)
    scale, wq = QuantizedLinearLoRA._quantize_int8(w)
    w_deq = wq.astype(np.float32) * np.float32(scale)

    assert scale == pytest.approx(float(np.max(np.abs(w))) / 127.0)
    assert np.all(np.abs(w - w_deq) <= scale / 2.0 + 1e-6)


# [ROBUST] Fails if the all-zero fallback uses scale = 0 or divides by zero.
def test_zero_weight_matrix_uses_unit_scale_and_exact_zero():
    w = np.zeros((2, 3), dtype=np.float32)
    scale, wq = QuantizedLinearLoRA._quantize_int8(w)

    assert scale == 1.0
    assert wq.dtype == np.int8
    assert np.array_equal(wq, np.zeros((2, 3), dtype=np.int8))
    assert np.array_equal(wq.astype(np.float32) * np.float32(scale), w)


# [ROBUST] Fails if the scale is exactly 127 (the ±max weight maps to ±127).
def test_max_magnitude_weight_uses_scale_of_one():
    w = np.array([[127.0, -127.0]], dtype=np.float32)
    scale, wq = QuantizedLinearLoRA._quantize_int8(w)

    assert scale == pytest.approx(1.0)
    assert np.array_equal(wq, np.array([[127, -127]], dtype=np.int8))


# [GUARD] Fails if empty, NaN, or Inf weights slip through the quantizer.
def test_quantization_rejects_empty_and_nonfinite_weights():
    with pytest.raises(ValueError):
        QuantizedLinearLoRA._quantize_int8(np.empty((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        QuantizedLinearLoRA._quantize_int8(np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError):
        QuantizedLinearLoRA._quantize_int8(np.array([[np.inf]], dtype=np.float32))


# [ROBUST] Fails if a tiny subnormal weight underflows the float32 scale to 0.
def test_quantization_rejects_scale_that_underflows_float32():
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    with pytest.raises(ValueError):
        QuantizedLinearLoRA._quantize_int8(np.array([[tiny]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Construction and reproducibility
# ---------------------------------------------------------------------------


# [ROBUST] Fails if same-seed construction is not bit-identical.
def test_same_seed_reproduces_initialization():
    a = QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=123)
    b = QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=123)

    assert np.array_equal(a.w_quant, b.w_quant)
    assert a.w_scale == b.w_scale
    assert np.array_equal(a.lora_A, b.lora_A)
    assert np.array_equal(a.lora_B, b.lora_B)


# [ROBUST] Fails if the seed is ignored for either the base or the adapter.
def test_different_seeds_change_initialization():
    a = QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=1)
    b = QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=2)

    assert not np.array_equal(a.lora_A, b.lora_A)
    assert not np.array_equal(a.w_quant, b.w_quant)


# [ROBUST] Fails if a supplied Generator is not actually used for initialization.
def test_supplied_rng_determines_initialization():
    a = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, rng=np.random.default_rng(42))
    b = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, rng=np.random.default_rng(42))
    c = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=42)

    assert np.array_equal(a.w_quant, b.w_quant)
    assert np.array_equal(a.lora_A, b.lora_A)
    # The Generator and seed paths draw from the same BitGenerator.
    assert np.array_equal(a.w_quant, c.w_quant)
    assert np.array_equal(a.lora_A, c.lora_A)


# [CONTRACT] Fails if seed and rng are both accepted instead of rejected.
def test_constructor_rejects_both_seed_and_rng():
    with pytest.raises(ValueError):
        QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=1, rng=np.random.default_rng(2))


# [ROBUST] Fails if construction consumes or resets NumPy's global RNG.
def test_construction_does_not_touch_global_rng_state():
    before = np.random.get_state()
    QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=42)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


# [ROBUST] Fails if B is randomly initialized (adapter must be zero at step 0).
def test_zero_B_makes_initial_forward_equal_dequantized_base():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float32)

    assert np.array_equal(
        layer.forward(x), (x @ layer._dequantize_w0().T).astype(np.float32)
    )


# [ROBUST] Fails if r=0 computes alpha/r or allocates adapter arrays.
def test_rank_zero_has_no_adapters_and_uses_base_only():
    layer = QuantizedLinearLoRA(3, 2, r=0, alpha=16.0, seed=0)

    assert layer.lora_A is None
    assert layer.lora_B is None

    x = np.ones((2, 3), dtype=np.float32)
    assert np.array_equal(layer.forward(x), x @ layer._dequantize_w0().T)


# [GUARD] Fails if negative rank or non-positive dimensions are accepted.
def test_constructor_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        QuantizedLinearLoRA(3, 2, r=-1, alpha=4.0, seed=0)
    with pytest.raises(ValueError):
        QuantizedLinearLoRA(0, 2, r=2, alpha=4.0, seed=0)
    with pytest.raises(ValueError):
        QuantizedLinearLoRA(3, 0, r=2, alpha=4.0, seed=0)


# [GUARD] Fails if bool slips through as an int (True == 1 would pass isinstance).
def test_constructor_rejects_boolean_dimensions_and_rank():
    with pytest.raises(TypeError):
        QuantizedLinearLoRA(True, 3, r=2, alpha=4.0, seed=0)
    with pytest.raises(TypeError):
        QuantizedLinearLoRA(3, True, r=2, alpha=4.0, seed=0)
    with pytest.raises(TypeError):
        QuantizedLinearLoRA(3, 2, r=True, alpha=4.0, seed=0)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


# [ROBUST] Fails if the forward expression uses the wrong transpose or scale.
def test_manual_forward_matches_hand_computed_expression():
    layer = QuantizedLinearLoRA(2, 2, r=1, alpha=2.0, seed=0)
    layer.w_scale = 0.5
    layer.w_quant = np.array([[1, 2], [3, 4]], dtype=np.int8)
    layer.lora_A = np.array([[1.0, -1.0]], dtype=np.float32)
    layer.lora_B = np.array([[2.0], [3.0]], dtype=np.float32)

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    # base = x @ (0.5 * [[1,2],[3,4]]).T = [[-1.5, -0.5]]
    # adapter = (2/1) * (x @ A.T) @ B.T = 2 * [0] * [[2],[3]].T = [[0, 0]]
    expected = np.array([[-1.5, -0.5]], dtype=np.float32)
    assert np.allclose(layer.forward(x), expected, rtol=0.0, atol=1e-6)


# [ROBUST] Fails if scaling uses alpha*r, alpha, alpha/k, or r alone.
# With r=2, k=4, alpha in {2,4}: correct is alpha/r -> [[8]] and [[16]].
def test_scaling_is_alpha_over_r():
    x = np.ones((1, 4), dtype=np.float32)
    A = np.ones((2, 4), dtype=np.float32)
    B = np.ones((1, 2), dtype=np.float32)

    def make(alpha):
        layer = QuantizedLinearLoRA(4, 1, r=2, alpha=alpha, seed=0)
        layer.w_scale = 1.0
        layer.w_quant = np.zeros((1, 4), dtype=np.int8)
        layer.lora_A = A.copy()
        layer.lora_B = B.copy()
        return layer

    assert np.allclose(make(2.0).forward(x), [[8.0]], rtol=0.0, atol=1e-6)
    assert np.allclose(make(4.0).forward(x), [[16.0]], rtol=0.0, atol=1e-6)


# [GUARD] Fails if forward accepts non-finite input or the wrong feature count.
def test_forward_rejects_bad_inputs():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)

    with pytest.raises(ValueError):
        layer.forward(np.ones((2, 4), dtype=np.float32))

    bad = np.ones((2, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        layer.forward(bad)


# [ROBUST] Fails if empty batches crash or return the wrong shape.
def test_forward_empty_batch_returns_zero_rows():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    out = layer.forward(np.empty((0, 3), dtype=np.float32))

    assert out.shape == (0, 2)
    assert out.dtype == np.float32


# [GUARD] Fails if a corrupted lora_B shape is silently broadcast.
def test_forward_rejects_mutated_lora_shape():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_B = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        layer.forward(np.ones((2, 3), dtype=np.float32))


# [ROBUST] Fails if a corrupted base is consumed by forward without validation.
def test_forward_rejects_corrupted_base():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)

    layer.w_quant = np.zeros((5, 3), dtype=np.int8)
    with pytest.raises(ValueError):
        layer.forward(x)

    layer.w_quant = np.zeros((2, 4), dtype=np.int8)
    with pytest.raises(ValueError):
        layer.forward(x)

    layer.w_quant = np.zeros((2, 3), dtype=np.float32)  # wrong dtype
    with pytest.raises(ValueError):
        layer.forward(x)

    layer.w_quant = None
    with pytest.raises(TypeError):
        layer.forward(x)


# [ROBUST] Fails if a non-positive or non-finite w_scale is used silently.
def test_forward_rejects_nonpositive_or_nonfinite_scale():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)

    for bad in (-1.0, 0.0, float("nan"), float("inf")):
        layer.w_scale = bad
        with pytest.raises(ValueError):
            layer.forward(x)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


# [ROBUST] Fails if backward accumulates instead of overwriting gradients.
def test_backward_overwrites_gradients():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)

    layer.backward(x, np.ones((2, 2), dtype=np.float32))
    layer.backward(x, np.zeros((2, 2), dtype=np.float32))

    assert np.array_equal(layer.grad_A, np.zeros_like(layer.grad_A))
    assert np.array_equal(layer.grad_B, np.zeros_like(layer.grad_B))


# [ROBUST] Fails if the first backward pass (B=0) produces nonzero grad_A.
def test_first_backward_with_zero_B_has_zero_grad_A():
    layer = QuantizedLinearLoRA(4, 3, r=2, alpha=4.0, seed=0)
    x = np.array([[0.2, 0.3, -0.5, 0.7], [0.1, -0.4, 0.6, -0.2]], dtype=np.float32)
    g = np.array([[1.0, -0.5, 0.25], [0.3, 0.8, -0.7]], dtype=np.float32)

    layer.backward(x, g)

    assert np.allclose(layer.grad_A, 0.0)
    assert np.any(layer.grad_B != 0.0)


# [ROBUST] Fails if either gradient formula, sign, or scaling is wrong.
def test_gradients_match_finite_differences():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_A = np.array([[0.2, -0.4, 0.1], [-0.3, 0.5, 0.2]], dtype=np.float32)
    layer.lora_B = np.array([[0.7, -0.2], [0.3, 0.9]], dtype=np.float32)

    x = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float32)
    g = np.array([[0.7, -0.4], [0.2, 0.8]], dtype=np.float32)

    def loss():
        return float(np.sum(layer.forward(x) * g))

    eps = 1e-3

    def numeric(param):
        grad = np.zeros_like(param)
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                old = param[i, j]
                param[i, j] = old + eps
                plus = loss()
                param[i, j] = old - eps
                minus = loss()
                param[i, j] = old
                grad[i, j] = (plus - minus) / (2.0 * eps)
        return grad

    A0 = layer.lora_A.copy()
    B0 = layer.lora_B.copy()
    num_A = numeric(layer.lora_A)
    num_B = numeric(layer.lora_B)
    layer.lora_A, layer.lora_B = A0, B0

    layer.backward(x, g)
    assert np.allclose(layer.grad_A, num_A, rtol=1e-2, atol=1e-3)
    assert np.allclose(layer.grad_B, num_B, rtol=1e-2, atol=1e-3)


# [GUARD] Fails if backward accepts a mismatched grad_output shape.
def test_backward_rejects_mismatched_grad_output_shape():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        layer.backward(x, np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        layer.backward(x, np.ones((2, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Adapter updates
# ---------------------------------------------------------------------------


# [ROBUST] Fails if the update adds gradients or uses the wrong magnitude.
def test_positive_lr_moves_adapters_against_gradient():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_B = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=np.float32)

    x = np.ones((2, 3), dtype=np.float32)
    layer.backward(x, np.ones((2, 2), dtype=np.float32))
    A0, B0 = layer.lora_A.copy(), layer.lora_B.copy()
    gA, gB = layer.grad_A.copy(), layer.grad_B.copy()

    layer.update_adapters(0.1)

    assert np.allclose(layer.lora_A, A0 - 0.1 * gA, rtol=1e-6, atol=1e-7)
    assert np.allclose(layer.lora_B, B0 - 0.1 * gB, rtol=1e-6, atol=1e-7)


# [ROBUST] Fails if zero learning rate mutates adapters.
def test_zero_lr_leaves_adapters_unchanged():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)

    layer.backward(x, np.ones((2, 2), dtype=np.float32))
    A0, B0 = layer.lora_A.copy(), layer.lora_B.copy()
    layer.update_adapters(0.0)

    assert np.array_equal(layer.lora_A, A0)
    assert np.array_equal(layer.lora_B, B0)


# [GUARD] Fails if negative, NaN, or Inf learning rates are accepted.
def test_update_rejects_bad_learning_rates():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)

    with pytest.raises(ValueError):
        layer.update_adapters(-0.1)
    with pytest.raises(ValueError):
        layer.update_adapters(float("nan"))
    with pytest.raises(ValueError):
        layer.update_adapters(float("inf"))


# [GUARD] Fails if bool is accepted (float(True) == 1.0).
def test_update_rejects_boolean_learning_rate():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    with pytest.raises(TypeError):
        layer.update_adapters(True)
    with pytest.raises(TypeError):
        layer.update_adapters(False)


# [ROBUST] Fails if update-before-backward treats the zero gradients as stale.
def test_update_before_backward_is_a_noop():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    A0, B0 = layer.lora_A.copy(), layer.lora_B.copy()

    layer.update_adapters(0.5)

    assert np.array_equal(layer.lora_A, A0)
    assert np.array_equal(layer.lora_B, B0)


# [ROBUST] Fails if update_adapters touches w_quant or w_scale.
def test_update_leaves_frozen_base_unchanged():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_B = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=np.float32)

    x = np.ones((2, 3), dtype=np.float32)
    layer.backward(x, np.ones((2, 2), dtype=np.float32))
    wq0, ws0 = layer.w_quant.copy(), layer.w_scale

    layer.update_adapters(0.1)

    assert np.array_equal(layer.w_quant, wq0)
    assert layer.w_scale == ws0


# [GUARD] Fails if corrupted gradient shapes are silently broadcast.
def test_update_rejects_corrupted_gradient_shapes():
    a = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    a.grad_A = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        a.update_adapters(0.1)

    b = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    b.grad_B = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        b.update_adapters(0.1)


# [ROBUST] Fails if update reassigns new arrays instead of mutating in place.
def test_update_mutates_adapters_in_place():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_B = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=np.float32)

    ref_A, ref_B = layer.lora_A, layer.lora_B
    before = ref_B.copy()
    layer.backward(np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    layer.update_adapters(0.1)

    assert layer.lora_A is ref_A
    assert layer.lora_B is ref_B
    assert not np.array_equal(ref_B, before)


# [ROBUST] Fails if rank-zero backward/update allocates or touches adapters.
def test_rank_zero_backward_and_update_are_noops():
    layer = QuantizedLinearLoRA(3, 2, r=0, alpha=16.0, seed=0)
    layer.backward(np.ones((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    layer.update_adapters(0.1)

    assert layer.lora_A is None
    assert layer.lora_B is None
    assert layer.grad_A is None
    assert layer.grad_B is None


# [ROBUST] Fails if alpha=0 still produces a nonzero adapter contribution.
def test_alpha_zero_makes_adapter_and_gradients_zero():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=0.0, seed=0)
    x = np.ones((2, 3), dtype=np.float32)
    g = np.ones((2, 2), dtype=np.float32)

    base = (x @ layer._dequantize_w0().T).astype(np.float32)
    assert np.array_equal(layer.forward(x), base)

    layer.backward(x, g)
    assert np.array_equal(layer.grad_A, np.zeros_like(layer.grad_A))
    assert np.array_equal(layer.grad_B, np.zeros_like(layer.grad_B))


# [ROBUST] Fails if the update sign is wrong: a small step must lower the loss.
def test_gradient_descent_step_lowers_loss():
    layer = QuantizedLinearLoRA(3, 2, r=2, alpha=4.0, seed=0)
    layer.lora_B = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=np.float32)

    x = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=np.float32)
    g = np.array([[0.7, -0.4], [0.2, 0.8]], dtype=np.float32)

    loss = lambda: float(np.sum(layer.forward(x) * g))
    before = loss()

    layer.backward(x, g)
    layer.update_adapters(1e-3)

    assert loss() < before


# Not tested, and why:
# - NF4, double quantization, paged optimizers: not implemented (Int8 only).
# - GPU, mixed precision, distributed training: CPU NumPy only.
# - Adam or any stateful optimizer: the update is plain gradient descent.
# - Training loop, tokenizer, dataset, transformer: out of scope.
# - Executable Hypothesis property tests: Hypothesis is not installed; the
#   property is documented as a comment at the top of this file.
# - Clipping mutation: with scale = max|W|/127 the rounded values are
#   already inside [-128, 127], so clipping appears redundant under the
#   current scale. A test that removed clip would exercise float32
#   arithmetic rather than the quantizer's contract.
# - Thread safety: the layer is mutable and unsynchronized.
# - Cross-process reproducibility: numpy.default_rng is documented to be
#   reproducible across processes; not re-verified here.
# - Finite-difference tolerances (eps=1e-3, rtol=1e-2, atol=1e-3) were
#   chosen analytically, not tuned by measurement. Tune eps first if the
#   gradient test ever fails.
# - Immutability of the frozen base: w_quant and w_scale are public. The
#   shape/dtype/scale validators catch obvious corruption but cannot stop
#   a caller from assigning plausible-looking Int8 values.
