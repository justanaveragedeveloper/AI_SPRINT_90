"""
Unit tests for Day 38 loss functions.

Covers:
- Numerically stable Softmax (max‑subtraction, edge cases)
- Categorical Cross‑Entropy (forward, gradient, input validation)
- Mean Squared Error (forward, gradient, edge cases)
- Binary Cross‑Entropy (forward, gradient, boundary behaviour)

All tests are designed to verify both mathematical correctness and robustness.
"""

import numpy as np
import pytest
from losses import BCELoss, CrossEntropyLoss, MSELoss, softmax


# -----------------------------------------------------------------------------
# Softmax Tests
# -----------------------------------------------------------------------------
def test_softmax_normalization():
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-5)
    assert np.all(probs >= 0)


def test_softmax_order():
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    assert probs[0] > probs[1] > probs[2]


def test_softmax_large_positive():
    logits = np.array([1000.0, 1001.0, 999.0])
    probs = softmax(logits)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-5)
    assert np.all(probs > 0)
    assert probs[1] > probs[0] > probs[2]


def test_softmax_large_negative():
    logits = np.array([-1000.0, -1001.0, -999.0])
    probs = softmax(logits)
    assert np.allclose(np.sum(probs), 1.0, rtol=1e-5)
    assert np.all(probs > 0)
    # Larger logit => larger probability
    assert probs[2] > probs[0] > probs[1]


def test_softmax_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        softmax(np.array([]))


def test_softmax_invalid_type():
    with pytest.raises(TypeError):
        softmax([1, 2, 3])  # list, not ndarray


def test_softmax_nan():
    with pytest.raises(ValueError, match="inf or nan"):
        softmax(np.array([1.0, np.nan, 3.0]))


def test_softmax_batched():
    logits = np.array([[1.0, 2.0], [3.0, 1.0]])
    probs = softmax(logits, axis=-1)
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_softmax_invalid_axis():
    logits = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="axis 2 out of bounds"):
        softmax(logits, axis=2)


def test_softmax_constant_shift_invariance():
    logits = np.array([1.0, 2.0, 3.0])
    probs1 = softmax(logits)
    probs2 = softmax(logits + 1000.0)
    assert np.allclose(probs1, probs2)


def test_softmax_preserves_shape():
    logits = np.zeros((4, 3, 2))
    probs = softmax(logits, axis=-1)
    assert probs.shape == logits.shape
    assert np.allclose(np.sum(probs, axis=-1), 1.0)


# -----------------------------------------------------------------------------
# CrossEntropyLoss Tests
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_logits_1d():
    return np.array([2.0, 1.0, 0.1])


def test_cce_forward_1d(sample_logits_1d):
    criterion = CrossEntropyLoss()
    logits = sample_logits_1d
    target = 0
    loss, _ = criterion.forward(logits, targets=target)
    probs = softmax(logits)
    expected_loss = -np.log(probs[target])
    assert np.allclose(loss, expected_loss, rtol=1e-5)


def test_cce_backward_1d(sample_logits_1d):
    criterion = CrossEntropyLoss()
    logits = sample_logits_1d
    target = 0
    _, backward = criterion.forward(logits, targets=target)
    grad = backward()
    probs = softmax(logits)
    one_hot = np.zeros_like(probs)
    one_hot[target] = 1.0
    expected_grad = probs - one_hot
    assert np.allclose(grad, expected_grad, rtol=1e-5)


def test_cce_target_out_of_bounds():
    criterion = CrossEntropyLoss()
    with pytest.raises(ValueError, match="out of range"):
        criterion.forward(np.array([1.0, 2.0]), targets=2)


def test_cce_batched():
    logits = np.array([[1.0, 2.0], [3.0, 1.0]])
    targets = np.array([1, 0])
    criterion = CrossEntropyLoss()
    loss, backward = criterion.forward(logits, targets=targets)
    probs = softmax(logits, axis=-1)
    expected_loss = -np.mean([np.log(probs[0, 1]), np.log(probs[1, 0])])
    assert np.allclose(loss, expected_loss, rtol=1e-5)
    grad = backward()
    assert grad.shape == (2, 2)
    # Gradient sum should be zero for each sample
    assert np.allclose(np.sum(grad, axis=1), 0.0, atol=1e-5)


def test_cce_empty_logits():
    criterion = CrossEntropyLoss()
    with pytest.raises(ValueError, match="cannot be empty"):
        criterion.forward(np.array([]), targets=0)


def test_cce_gradient_sum_zero():
    logits = np.array([2.0, 1.0, 0.1, -2.0])
    for target in range(len(logits)):
        criterion = CrossEntropyLoss()
        _, backward = criterion.forward(logits, targets=target)
        grad = backward()
        assert np.allclose(np.sum(grad), 0.0, atol=1e-10)


def test_cce_gradient_finite_difference():
    criterion = CrossEntropyLoss()
    logits = np.array([1.0, 2.0, 0.5])
    target = 1
    loss, backward = criterion.forward(logits, targets=target)
    grad_analytic = backward()
    eps = 1e-6
    grad_num = np.zeros_like(logits)
    for i in range(len(logits)):
        logits_plus = logits.copy()
        logits_plus[i] += eps
        loss_plus, _ = criterion.forward(logits_plus, targets=target)
        grad_num[i] = (loss_plus - loss) / eps
    assert np.allclose(grad_analytic, grad_num, atol=1e-4)


def test_cce_batch_gradient_scaling():
    criterion = CrossEntropyLoss()
    logits = np.array([[1.0, 2.0], [3.0, 1.0]])
    targets = np.array([1, 0])
    _, backward = criterion.forward(logits, targets=targets)
    grad = backward()
    probs = softmax(logits, axis=-1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(2), targets] = 1.0
    expected_grad = (probs - one_hot) / 2.0  # divide by batch size
    assert np.allclose(grad, expected_grad)


def test_cce_log_sum_exp_numerical_stability():
    logits = np.array([1000.0, 1001.0, 999.0])
    criterion = CrossEntropyLoss()
    loss, backward = criterion.forward(logits, targets=1)
    assert np.isfinite(loss)
    grad = backward()
    assert np.all(np.isfinite(grad))


def test_cce_wrong_target_length():
    criterion = CrossEntropyLoss()
    logits = np.array([[1.0, 2.0], [2.0, 1.0]])
    targets = np.array([0])  # length 1 instead of 2
    with pytest.raises(ValueError, match="targets must be a 1D array of length batch size"):
        criterion.forward(logits, targets=targets)


def test_cce_invalid_target_dtype():
    criterion = CrossEntropyLoss()
    logits = np.array([[1.0, 2.0], [2.0, 1.0]])
    targets = np.array([0.0, 1.0])  # floats instead of integers
    with pytest.raises(TypeError, match="targets must be integer indices"):
        criterion.forward(logits, targets=targets)


# -----------------------------------------------------------------------------
# MSELoss Tests
# -----------------------------------------------------------------------------
def test_mse_forward():
    criterion = MSELoss()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.5])
    loss, _ = criterion.forward(pred, target)
    expected_loss = np.mean((pred - target) ** 2)
    assert np.allclose(loss, expected_loss)


def test_mse_backward():
    criterion = MSELoss()
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.5])
    _, backward = criterion.forward(pred, target)
    grad = backward()
    expected_grad = 2.0 * (pred - target) / pred.size
    assert np.allclose(grad, expected_grad)


def test_mse_shape_mismatch():
    criterion = MSELoss()
    with pytest.raises(ValueError, match="Shape mismatch"):
        criterion.forward(np.array([1.0, 2.0]), np.array([1.0]))


def test_mse_empty():
    criterion = MSELoss()
    with pytest.raises(ValueError, match="cannot be empty"):
        criterion.forward(np.array([]), np.array([]))


def test_mse_multidimensional():
    criterion = MSELoss()
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.5, 2.5], [3.5, 4.5]])
    loss, backward = criterion.forward(pred, target)
    expected_loss = np.mean((pred - target) ** 2)
    assert np.allclose(loss, expected_loss)
    grad = backward()
    expected_grad = 2.0 * (pred - target) / pred.size
    assert np.allclose(grad, expected_grad)


def test_mse_gradient_finite_difference():
    criterion = MSELoss()
    pred = np.array([1.2, 2.3, 3.1])
    target = np.array([1.0, 2.0, 3.0])
    _, backward = criterion.forward(pred, target)
    analytic = backward()
    eps = 1e-6
    numerical = np.zeros_like(pred)
    for i in range(pred.size):
        plus = pred.copy()
        minus = pred.copy()
        plus[i] += eps
        minus[i] -= eps
        loss_plus, _ = criterion.forward(plus, target)
        loss_minus, _ = criterion.forward(minus, target)
        numerical[i] = (loss_plus - loss_minus) / (2.0 * eps)
    assert np.allclose(analytic, numerical, atol=1e-6)


# -----------------------------------------------------------------------------
# BCELoss Tests
# -----------------------------------------------------------------------------
def test_bce_forward():
    criterion = BCELoss(eps=1e-15)
    p = np.array([0.8, 0.2])
    y = np.array([1.0, 0.0])
    loss, _ = criterion.forward(p, y)
    expected_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert np.allclose(loss, expected_loss)


def test_bce_backward():
    criterion = BCELoss(eps=1e-15)
    p = np.array([0.8, 0.2])
    y = np.array([1.0, 0.0])
    _, backward = criterion.forward(p, y)
    grad = backward()
    denom = p * (1 - p)
    expected_grad = (p - y) / denom / p.size
    assert np.allclose(grad, expected_grad)


def test_bce_probability_bounds():
    criterion = BCELoss(eps=1e-15)
    with pytest.raises(ValueError, match="values must be in \\[0,1\\]"):
        criterion.forward(np.array([1.2]), np.array([0.5]))
    with pytest.raises(ValueError, match="values must be in \\[0,1\\]"):
        criterion.forward(np.array([0.5]), np.array([1.2]))


def test_bce_target_bounds():
    criterion = BCELoss(eps=1e-15)
    with pytest.raises(ValueError, match="values must be in \\[0,1\\]"):
        criterion.forward(np.array([0.5]), np.array([-0.1]))


def test_bce_near_boundaries():
    criterion = BCELoss(eps=1e-8)
    p = np.array([1e-10, 1.0 - 1e-10])
    y = np.array([0.0, 1.0])
    loss, backward = criterion.forward(p, y)
    assert np.isfinite(loss)
    grad = backward()
    assert np.all(np.isfinite(grad))


def test_bce_empty():
    criterion = BCELoss()
    with pytest.raises(ValueError, match="cannot be empty"):
        criterion.forward(np.array([]), np.array([]))


def test_bce_shape_mismatch():
    criterion = BCELoss()
    with pytest.raises(ValueError, match="Shape mismatch"):
        criterion.forward(np.array([0.5]), np.array([0.5, 0.5]))


def test_bce_gradient_finite_difference():
    criterion = BCELoss(eps=1e-8)
    p = np.array([0.2, 0.7, 0.4])
    y = np.array([0.0, 1.0, 0.0])
    _, backward = criterion.forward(p, y)
    analytic = backward()
    eps = 1e-6
    numerical = np.zeros_like(p)
    for i in range(p.size):
        plus = p.copy()
        minus = p.copy()
        plus[i] += eps
        minus[i] -= eps
        loss_plus, _ = criterion.forward(plus, y)
        loss_minus, _ = criterion.forward(minus, y)
        numerical[i] = (loss_plus - loss_minus) / (2.0 * eps)
    assert np.allclose(analytic, numerical, atol=1e-5)


def test_bce_exact_boundaries_are_finite():
    criterion = BCELoss(eps=1e-8)
    p = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    loss, backward = criterion.forward(p, y)
    assert np.isfinite(loss)
    grad = backward()
    assert np.all(np.isfinite(grad))
    # At the boundaries, the gradient should be zero (clip derivative is zero)
    assert np.all(grad == 0.0)