"""
Test suite for the Day 30 regularisation and gradient clipping functions.

These tests verify the mathematical correctness, statistical behaviour,
and robustness to edge cases. They are written to be easy to understand
and to show that the code works as intended.
"""

import numpy as np
import pytest
from regularization import (
    Dropout,
    apply_l1_l2_gradients,
    clip_gradients_norm,
    clip_gradients_value,
    compute_l1_penalty,
    compute_l2_penalty,
)

# ----------------------------------------------------------------------
#  L1 Regularisation tests
# ----------------------------------------------------------------------


class TestL1:
    def test_penalty_calculation(self):
        # L1 penalty for weights [1, -2, 3] and [-4, 0] with λ=0.1
        # Expected: 0.1 * (1+2+3+4+0) = 1.0
        params = [np.array([1.0, -2.0, 3.0]), np.array([-4.0, 0.0])]
        l1 = compute_l1_penalty(params, l1_lambda=0.1)
        assert np.isclose(l1, 1.0)

    def test_positive_weights(self):
        w = np.array([0.5, 1.5, 2.5])
        l1 = compute_l1_penalty([w], l1_lambda=2.0)
        # 2.0 * (0.5+1.5+2.5) = 9.0
        assert np.isclose(l1, 9.0)

    def test_negative_weights(self):
        w = np.array([-1.0, -2.0, -3.0])
        l1 = compute_l1_penalty([w], l1_lambda=0.5)
        # 0.5 * (1+2+3) = 3.0
        assert np.isclose(l1, 3.0)

    def test_zero_weights(self):
        w = np.array([0.0, 0.0, 0.0])
        l1 = compute_l1_penalty([w], l1_lambda=10.0)
        assert l1 == 0.0

    def test_zero_lambda(self):
        w = np.array([1.0, -2.0])
        l1 = compute_l1_penalty([w], l1_lambda=0.0)
        assert l1 == 0.0

    def test_gradient_contribution(self):
        # For w = [2, -3, 0], L1 gradient is λ*sign(w)
        w = np.array([2.0, -3.0, 0.0])
        dw = np.array([1.0, 1.0, 1.0])
        apply_l1_l2_gradients([w], [dw], l1_lambda=0.5, l2_lambda=0.0)
        expected = np.array([1.5, 0.5, 1.0])  # base + 0.5*sign
        np.testing.assert_allclose(dw, expected)


# ----------------------------------------------------------------------
#  L2 Regularisation tests
# ----------------------------------------------------------------------


class TestL2:
    def test_penalty_calculation(self):
        # L2 penalty for [1, -2, 3] and [-4, 0] with λ=0.2
        # Sum of squares = 1+4+9+16+0 = 30; penalty = 0.5 * 0.2 * 30 = 3.0
        params = [np.array([1.0, -2.0, 3.0]), np.array([-4.0, 0.0])]
        l2 = compute_l2_penalty(params, l2_lambda=0.2)
        assert np.isclose(l2, 3.0)

    def test_positive_weights(self):
        w = np.array([0.5, 1.5])
        l2 = compute_l2_penalty([w], l2_lambda=4.0)
        # 0.5 * 4.0 * (0.25 + 2.25) = 5.0
        assert np.isclose(l2, 5.0)

    def test_negative_weights(self):
        w = np.array([-1.0, -2.0])
        l2 = compute_l2_penalty([w], l2_lambda=3.0)
        # 0.5 * 3.0 * (1+4) = 7.5
        assert np.isclose(l2, 7.5)

    def test_zero_weights(self):
        w = np.array([0.0, 0.0])
        l2 = compute_l2_penalty([w], l2_lambda=5.0)
        assert l2 == 0.0

    def test_zero_lambda(self):
        w = np.array([1.0, 2.0])
        l2 = compute_l2_penalty([w], l2_lambda=0.0)
        assert l2 == 0.0

    def test_gradient_contribution(self):
        # For w = [2, -3, 0], L2 gradient is λ*w
        w = np.array([2.0, -3.0, 0.0])
        dw = np.array([1.0, 1.0, 1.0])
        apply_l1_l2_gradients([w], [dw], l1_lambda=0.0, l2_lambda=0.1)
        expected = np.array([1.2, 0.7, 1.0])  # base + 0.1*w
        np.testing.assert_allclose(dw, expected)


# ----------------------------------------------------------------------
#  Combined L1 + L2 tests
# ----------------------------------------------------------------------


class TestCombined:
    def test_both_gradients(self):
        w = np.array([2.0, -3.0, 0.0])
        dw = np.array([1.0, 1.0, 1.0])
        apply_l1_l2_gradients([w], [dw], l1_lambda=0.5, l2_lambda=0.1)
        # Expected: base + 0.1*w + 0.5*sign(w)
        expected = np.array([1.7, 0.2, 1.0])
        np.testing.assert_allclose(dw, expected)


# ----------------------------------------------------------------------
#  Dropout tests
# ----------------------------------------------------------------------


class TestDropout:
    def test_training_mode_statistics(self):
        # With drop_rate=0.3, about 30% of activations should be zero,
        # and the mean of the remaining should be about 1.0 (since input is all 1s).
        np.random.seed(42)
        p = 0.3
        dropout = Dropout(drop_rate=p)
        x = np.ones((1000, 100))
        out = dropout.forward(x)
        zero_fraction = np.mean(out == 0)
        assert np.isclose(zero_fraction, p, atol=0.02)
        assert np.isclose(np.mean(out), 1.0, atol=0.05)

    def test_eval_mode_identity(self):
        # In evaluation mode, the layer should just return the input.
        dropout = Dropout(drop_rate=0.5)
        dropout.eval()
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = dropout.forward(x)
        np.testing.assert_array_equal(out, x)

    def test_drop_rate_zero(self):
        # If drop_rate is 0, dropout is effectively disabled.
        dropout = Dropout(drop_rate=0.0)
        dropout.train()
        x = np.random.randn(100, 100)
        out = dropout.forward(x)
        np.testing.assert_array_equal(out, x)
        dout = np.random.randn(100, 100)
        back = dropout.backward(dout)
        np.testing.assert_array_equal(back, dout)

    def test_mask_reuse_with_arbitrary_gradient(self):
        # The backward pass must use the exact same mask as forward.
        # We test this with a non‑uniform gradient to be sure.
        np.random.seed(123)
        dropout = Dropout(drop_rate=0.4)
        dropout.train()
        x = np.ones((4, 4))
        dropout.forward(x)
        mask = dropout.mask.copy()
        dout = np.arange(1.0, 17.0).reshape(4, 4)
        expected = dout * mask
        actual = dropout.backward(dout)
        np.testing.assert_array_equal(actual, expected)

    def test_reproducibility(self):
        # With the same random seed, dropout should produce the same mask.
        np.random.seed(0)
        dropout1 = Dropout(drop_rate=0.5)
        x = np.ones((100, 100))
        out1 = dropout1.forward(x)
        np.random.seed(0)
        dropout2 = Dropout(drop_rate=0.5)
        out2 = dropout2.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_backward_without_forward_raises(self):
        # You cannot call backward() before a training forward.
        dropout = Dropout(0.3)
        dropout.train()
        dout = np.ones((10, 10))
        with pytest.raises(RuntimeError):
            dropout.backward(dout)

    def test_training_after_eval_requires_new_forward(self):
        # Switching back to training invalidates the old mask.
        dropout = Dropout(0.5)
        dropout.train()
        x = np.ones((4, 4))
        dropout.forward(x)  # creates a mask
        dropout.eval()
        dropout.train()  # mask is now stale
        dout = np.ones((4, 4))
        with pytest.raises(RuntimeError):
            dropout.backward(dout)

    def test_backward_shape_mismatch_raises(self):
        # The gradient shape must match the mask shape.
        dropout = Dropout(0.3)
        dropout.train()
        x = np.ones((5, 5))
        dropout.forward(x)
        dout = np.ones((4, 4))  # wrong shape
        with pytest.raises(ValueError, match="shape.*does not match"):
            dropout.backward(dout)


# ----------------------------------------------------------------------
#  Value‑clipping tests
# ----------------------------------------------------------------------


class TestValueClipping:
    def test_positive_overflow(self):
        dw = np.array([5.0, 2.0, -1.0])
        clip_gradients_value([dw], clip_value=3.0)
        np.testing.assert_allclose(dw, [3.0, 2.0, -1.0])

    def test_negative_overflow(self):
        dw = np.array([5.0, -2.0, -10.0])
        clip_gradients_value([dw], clip_value=4.0)
        np.testing.assert_allclose(dw, [4.0, -2.0, -4.0])

    def test_inside_range(self):
        dw = np.array([2.0, -1.0, 0.5])
        clip_gradients_value([dw], clip_value=3.0)
        np.testing.assert_allclose(dw, [2.0, -1.0, 0.5])

    def test_zero_clip_raises(self):
        # clip_value must be strictly positive
        with pytest.raises(ValueError, match="must be > 0"):
            clip_gradients_value([np.array([1.0])], clip_value=0.0)

    def test_negative_clip_raises(self):
        with pytest.raises(ValueError):
            clip_gradients_value([np.array([1.0])], clip_value=-1.0)

    def test_nonfinite_clip_raises(self):
        with pytest.raises(ValueError):
            clip_gradients_value([np.array([1.0])], np.nan)


# ----------------------------------------------------------------------
#  Global norm clipping tests
# ----------------------------------------------------------------------


class TestNormClipping:
    def test_original_norm_returned(self):
        dw1 = np.array([3.0, 0.0])
        dw2 = np.array([0.0, 4.0])
        grads = [dw1, dw2]
        orig = clip_gradients_norm(grads, max_norm=10.0)
        assert np.isclose(orig, 5.0)  # sqrt(3² + 4²)

    def test_clipping_to_target_norm(self):
        dw1 = np.array([3.0, 0.0])
        dw2 = np.array([0.0, 4.0])
        grads = [dw1, dw2]
        orig = clip_gradients_norm(grads, max_norm=2.5)
        assert np.isclose(orig, 5.0)
        new_norm = np.sqrt(np.sum(dw1**2) + np.sum(dw2**2))
        assert np.isclose(new_norm, 2.5)

    def test_direction_preserved(self):
        # Scaling should preserve the ratio between components.
        dw1 = np.array([3.0, 0.0])
        dw2 = np.array([0.0, 4.0])
        grads = [dw1, dw2]
        _ = clip_gradients_norm(grads, max_norm=2.5)
        assert np.isclose(dw1[0] / dw2[1], 3.0 / 4.0)

    def test_exact_scaling_factor(self):
        # With max_norm = original_norm / 2, scale factor should be exactly 0.5.
        g1 = np.array([3.0, 4.0])
        g2 = np.array([6.0])
        original = [g1.copy(), g2.copy()]
        norm = np.sqrt(3**2 + 4**2 + 6**2)
        max_norm = norm / 2.0
        clip_gradients_norm([g1, g2], max_norm)
        np.testing.assert_allclose(g1, original[0] * 0.5)
        np.testing.assert_allclose(g2, original[1] * 0.5)

    def test_no_clipping_below_threshold(self):
        dw = np.array([1.0, 2.0])
        grads = [dw]
        orig = clip_gradients_norm(grads, max_norm=10.0)
        assert np.isclose(orig, np.sqrt(5))
        np.testing.assert_allclose(dw, [1.0, 2.0])  # unchanged

    def test_zero_gradient(self):
        dw = np.array([0.0, 0.0])
        grads = [dw]
        orig = clip_gradients_norm(grads, max_norm=5.0)
        assert orig == 0.0
        np.testing.assert_allclose(dw, [0.0, 0.0])

    def test_zero_max_norm_raises(self):
        # max_norm must be strictly positive
        with pytest.raises(ValueError, match="must be > 0"):
            clip_gradients_norm([np.array([1.0])], max_norm=0.0)

    def test_negative_max_norm_raises(self):
        with pytest.raises(ValueError):
            clip_gradients_norm([np.array([1.0])], max_norm=-1.0)

    def test_stable_norm_extreme_values(self):
        # Large finite values (1e308) would overflow a naive norm calculation,
        # but our stable method handles them correctly.
        large = 1e308
        grads = [np.array([large, -large])]
        max_norm = 1.0
        orig = clip_gradients_norm(grads, max_norm)
        assert np.isfinite(orig)
        assert np.max(np.abs(grads[0])) <= 1.0
        new_norm = np.sqrt(np.sum(grads[0] ** 2))
        assert np.isclose(new_norm, max_norm)


# ----------------------------------------------------------------------
#  Input validation tests
# ----------------------------------------------------------------------


class TestValidation:
    def test_invalid_dropout_rate(self):
        with pytest.raises(ValueError):
            Dropout(drop_rate=1.0)
        with pytest.raises(ValueError):
            Dropout(drop_rate=-0.1)
        with pytest.raises(ValueError):
            Dropout(drop_rate=np.nan)

    def test_nonfinite_l1_lambda(self):
        with pytest.raises(ValueError):
            compute_l1_penalty([np.array([1.0])], np.nan)
        with pytest.raises(ValueError):
            compute_l1_penalty([np.array([1.0])], np.inf)

    def test_nonfinite_l2_lambda(self):
        with pytest.raises(ValueError):
            compute_l2_penalty([np.array([1.0])], np.nan)

    def test_negative_l1_lambda(self):
        with pytest.raises(ValueError):
            compute_l1_penalty([np.array([1.0])], l1_lambda=-1.0)

    def test_negative_l2_lambda(self):
        with pytest.raises(ValueError):
            compute_l2_penalty([np.array([1.0])], l2_lambda=-1.0)

    def test_apply_negative_lambda(self):
        with pytest.raises(ValueError):
            apply_l1_l2_gradients([np.array([1.0])], [np.array([0.0])], l1_lambda=-0.5)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            apply_l1_l2_gradients([np.array([1.0]), np.array([2.0])], [np.array([0.0])])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            apply_l1_l2_gradients([np.ones((2, 2))], [np.ones((2,))], l2_lambda=0.1)

    def test_nonfinite_gradient_in_apply(self):
        with pytest.raises(ValueError):
            apply_l1_l2_gradients(
                [np.array([1.0, 2.0])], [np.array([1.0, np.nan])], l2_lambda=0.1
            )

    def test_nonfinite_parameter_in_apply(self):
        with pytest.raises(ValueError):
            apply_l1_l2_gradients(
                [np.array([1.0, np.nan])], [np.array([1.0, 2.0])], l2_lambda=0.1
            )

    def test_nonfinite_input_dropout_forward(self):
        dropout = Dropout(0.3)
        with pytest.raises(ValueError):
            dropout.forward(np.array([1.0, np.nan]))

    def test_nonfinite_gradient_dropout_backward(self):
        dropout = Dropout(0.3)
        dropout.train()
        dropout.forward(np.ones(5))
        with pytest.raises(ValueError):
            dropout.backward(np.array([1.0, np.nan, 3.0, 4.0, 5.0]))

    def test_params_not_list_in_l1(self):
        with pytest.raises(TypeError, match="params must be a list or tuple"):
            compute_l1_penalty(np.array([1.0, 2.0]), 0.1)

    def test_params_not_list_in_l2(self):
        with pytest.raises(TypeError, match="params must be a list or tuple"):
            compute_l2_penalty(np.array([1.0, 2.0]), 0.1)
