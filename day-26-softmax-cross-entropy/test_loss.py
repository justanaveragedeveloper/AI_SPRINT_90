"""
test_loss.py

Comprehensive test suite for softmax and categorical cross‑entropy loss.
Includes additional checks for gradient correctness and numerical robustness.
"""

import logging
import math
import os
import sys

import pytest

# Ensure the day-24-autograd directory is in the path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../day-24-autograd"))
)

from engine import Value  # type: ignore
from loss_and_ops import categorical_cross_entropy, softmax

# Disable logging warnings during tests to keep output clean
logging.disable(logging.CRITICAL)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def assert_value_close(actual: Value, expected: float, tol: float = 1e-6):
    """Assert that a Value's data is close to expected."""
    assert abs(actual.value - expected) < tol, f"{actual.value} != {expected}"


# ----------------------------------------------------------------------
# Softmax tests
# ----------------------------------------------------------------------


def test_softmax_probabilities_sum_to_one():
    logits = [Value(2.0), Value(1.0), Value(0.1)]
    probs = softmax(logits)

    total = sum(p.value for p in probs)
    assert abs(total - 1.0) < 1e-6
    assert all(0.0 <= p.value <= 1.0 for p in probs)


def test_softmax_equal_logits_produce_equal_probs():
    logits = [Value(0.5), Value(0.5), Value(0.5)]
    probs = softmax(logits)

    expected = 1.0 / 3.0
    for p in probs:
        assert abs(p.value - expected) < 1e-6


def test_softmax_numerical_stability_large_logits():
    large = [Value(1000.0), Value(999.0), Value(1001.0)]
    probs = softmax(large)
    total = sum(p.value for p in probs)
    assert abs(total - 1.0) < 1e-6
    assert all(0.0 <= p.value <= 1.0 for p in probs)
    idx_max = max(range(len(probs)), key=lambda i: probs[i].value)
    assert idx_max == 2


def test_softmax_numerical_stability_negative_logits():
    neg = [Value(-1000.0), Value(-999.0), Value(-1001.0)]
    probs = softmax(neg)
    total = sum(p.value for p in probs)
    assert abs(total - 1.0) < 1e-6
    idx_max = max(range(len(probs)), key=lambda i: probs[i].value)
    assert idx_max == 1


def test_softmax_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        softmax([])


def test_softmax_invalid_type_raises():
    with pytest.raises(TypeError, match="expected Value"):
        softmax([1.0, 2.0])  # not Value


# ----------------------------------------------------------------------
# Cross‑entropy loss tests
# ----------------------------------------------------------------------


def test_cross_entropy_loss_computation():
    logits = [Value(3.0), Value(1.0), Value(0.2)]
    probs = softmax(logits)
    loss = categorical_cross_entropy(probs, target_idx=0)

    denom = math.exp(3) + math.exp(1) + math.exp(0.2)
    expected = -math.log(math.exp(3) / denom)
    assert abs(loss.value - expected) < 1e-6


def test_cross_entropy_epsilon_protection():
    logits = [Value(-1000.0), Value(0.0)]
    probs = softmax(logits)  # probs[0] ~ 0, probs[1] ~ 1
    loss = categorical_cross_entropy(probs, target_idx=0)
    assert not math.isinf(loss.value)
    assert loss.value > 0.0


def test_cross_entropy_gradient_propagation():
    logits = [Value(3.0), Value(1.0), Value(0.2)]
    probs = softmax(logits)
    loss = categorical_cross_entropy(probs, target_idx=1)
    loss.backward()

    for z in logits:
        assert z.gradient != 0.0, f"Gradient for {z.value} is zero"


def test_cross_entropy_gradient_balance():
    logits = [Value(1.5), Value(-0.5), Value(2.1)]
    probs = softmax(logits)
    loss = categorical_cross_entropy(probs, target_idx=2)
    loss.backward()

    grad_sum = sum(z.gradient for z in logits)
    assert abs(grad_sum) < 1e-5, f"Gradient sum = {grad_sum}"


def test_cross_entropy_invalid_target_raises():
    logits = [Value(0.0), Value(1.0)]
    probs = softmax(logits)

    with pytest.raises(IndexError, match="target_idx 2 out of range"):
        categorical_cross_entropy(probs, target_idx=2)

    with pytest.raises(IndexError, match="target_idx -1 out of range"):
        categorical_cross_entropy(probs, target_idx=-1)


def test_cross_entropy_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        categorical_cross_entropy([], target_idx=0)


def test_cross_entropy_invalid_type_raises():
    probs = [Value(0.2), 0.3]  # second is float, not Value
    with pytest.raises(TypeError, match="expected Value"):
        categorical_cross_entropy(probs, target_idx=0)


# ----------------------------------------------------------------------
# Additional (optional) tests for enhanced coverage
# ----------------------------------------------------------------------


def test_cross_entropy_loss_non_negative():
    logits = [Value(0.0), Value(1.0), Value(-0.5)]
    probs = softmax(logits)
    for target in range(len(probs)):
        loss = categorical_cross_entropy(probs, target)
        assert loss.value >= 0.0, f"Loss negative for target {target}: {loss.value}"


def test_cross_entropy_gradient_exact_formula():
    """
    Verify that gradient w.r.t. logits equals (probs - one_hot) for the target.
    This is a known closed‑form derivative of softmax + cross‑entropy.
    """
    logits = [Value(2.0), Value(0.5), Value(-1.0)]
    probs = softmax(logits)
    target = 1

    loss = categorical_cross_entropy(probs, target)
    loss.backward()

    # Expected gradient: dL/dz_i = p_i - δ_{i,target}
    expected_grads = [p.value for p in probs]
    expected_grads[target] -= 1.0

    for z, g_exp in zip(logits, expected_grads):
        assert (
            abs(z.gradient - g_exp) < 1e-6
        ), f"Gradient mismatch for {z.value}: {z.gradient} vs {g_exp}"


def test_cross_entropy_custom_epsilon():
    # When epsilon is very large, it should affect the loss
    logits = [Value(-1000.0), Value(0.0)]
    probs = softmax(logits)
    # With default epsilon, loss ~ 1000
    loss_default = categorical_cross_entropy(probs, target_idx=0)
    # With a larger epsilon (say 1e-3), loss should be smaller (less negative log)
    loss_large_eps = categorical_cross_entropy(probs, target_idx=0, epsilon=1e-3)
    assert loss_large_eps.value < loss_default.value


# ----------------------------------------------------------------------
# End-to-end integration test
# ----------------------------------------------------------------------


def test_full_backward_pass():
    logits = [Value(x) for x in [2.0, -1.0, 0.5]]
    probs = softmax(logits)
    loss = categorical_cross_entropy(probs, target_idx=0)
    loss.backward()

    for z in logits:
        assert z.gradient is not None
        assert z.gradient != 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
