"""
Pytest suite for Day 36 CNN Architecture.

These tests verify:
    - Correct shape propagation through the entire CNN.
    - Gradient accumulation and zeroing.
    - Numerical agreement with finite differences (input, weight, bias).
    - Edge cases (ties in pooling, zero inputs, invalid shapes).
"""

import math

import pytest
from cnn_arch import (
    Conv2DLayer,
    ConvNet,
    DenseLayer,
    FlattenLayer,
    MaxPool2DLayer,
)

# ---------- Fixtures ----------

@pytest.fixture
def sample_input_4x4() -> list[list[list[float]]]:
    """A checkerboard 4×4 input (common test pattern)."""
    return [[
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]]


@pytest.fixture
def gradient_input_4x4() -> list[list[list[float]]]:
    """A smooth gradient 4×4 input to avoid accidental cancellation."""
    return [[
        [float(i + j) for j in range(4)]
        for i in range(4)
    ]]


# ---------- FlattenLayer ----------

def test_flatten_forward_ordering():
    """Check that flatten preserves channel‑major order."""
    flatten = FlattenLayer()
    X = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]  # (2,2,2)
    flat_out, _ = flatten.forward(X)
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert flat_out == expected


def test_flatten_backward_shape_and_values():
    """Flatten backward must reconstruct original shape and route gradients correctly."""
    flatten = FlattenLayer()
    X = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    _, backward = flatten.forward(X)
    dY_flat = [0.1 * (i + 1) for i in range(8)]
    dX = backward(dY_flat)

    assert len(dX) == 2
    assert len(dX[0]) == 2
    assert len(dX[0][0]) == 2
    assert dX[0][0][0] == 0.1
    assert dX[1][1][1] == 0.8


def test_flatten_backward_invalid_gradient_length():
    """Backward should raise error if gradient length does not match flattened size."""
    flatten = FlattenLayer()
    X = [[[1.0, 2.0], [3.0, 4.0]]]  # shape (1,2,2) → length 4
    _, backward = flatten.forward(X)
    with pytest.raises(ValueError, match="length"):
        backward([1.0, 2.0, 3.0])  # only 3 elements


# ---------- Conv2DLayer ----------

def test_conv2d_output_dimensions():
    """With padding=1, stride=1, kernel=3, spatial size should stay the same."""
    conv = Conv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
    X = [[[1.0] * 4 for _ in range(4)]]  # (1,4,4)
    out, _ = conv.forward(X)
    assert len(out) == 2
    assert len(out[0]) == 4
    assert len(out[0][0]) == 4


def test_conv2d_gradient_accumulation():
    """Parameter gradients must accumulate over multiple backward calls."""
    conv = Conv2DLayer(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
    X = [[[1.0, 2.0], [3.0, 4.0]]]  # (1,2,2)
    _, backward = conv.forward(X)
    dY = [[[1.0]]]  # output (1,1,1)
    _ = backward(dY)
    # Check weight and bias gradients after first backward
    assert conv.weight_grad[0][0][0][0] == 1.0
    assert conv.weight_grad[0][0][0][1] == 2.0
    assert conv.weight_grad[0][0][1][0] == 3.0
    assert conv.weight_grad[0][0][1][1] == 4.0
    assert conv.bias_grad[0] == 1.0

    # Second backward should add to existing gradients
    _ = backward(dY)
    assert conv.weight_grad[0][0][0][0] == 2.0


def test_conv2d_zero_grad():
    """zero_grad() should reset all parameter gradients to zero."""
    conv = Conv2DLayer(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
    X = [[[1.0, 2.0], [3.0, 4.0]]]
    _, backward = conv.forward(X)
    dY = [[[1.0]]]
    _ = backward(dY)
    conv.zero_grad()
    assert all(v == 0.0 for v in conv.weight_grad[0][0][0])
    assert conv.bias_grad[0] == 0.0


def test_conv2d_invalid_dimensions():
    """Layer should reject input with wrong number of channels."""
    conv = Conv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    X_bad = [[[1.0] * 4 for _ in range(4)], [[1.0] * 4 for _ in range(4)]]  # (2,4,4)
    with pytest.raises(ValueError, match="Input channels"):
        conv.forward(X_bad)


# ---------- MaxPool2DLayer ----------

def test_maxpool_forward_backward_argmax():
    """Pool should output the maximum and route gradient only to that position."""
    pool = MaxPool2DLayer(kernel_size=2, stride=2)
    X = [[[1.0, 3.0], [2.0, 4.0]]]
    out, backward = pool.forward(X)
    assert out[0][0][0] == 4.0

    dY = [[[1.0]]]
    dX = backward(dY)
    # Only the position (1,1) should receive the gradient
    assert dX[0][0][0] == 0.0
    assert dX[0][0][1] == 0.0
    assert dX[0][1][0] == 0.0
    assert dX[0][1][1] == 1.0


def test_maxpool_tie_uses_first_maximum():
    """When two values are equal, the first encountered (row‑major) wins."""
    pool = MaxPool2DLayer(kernel_size=2, stride=2)
    X = [[
        [5.0, 5.0],
        [1.0, 1.0],
    ]]
    out, backward = pool.forward(X)
    assert out == [[[5.0]]]
    dY = [[[1.0]]]
    dX = backward(dY)
    # First maximum is at (0,0)
    expected = [[1.0, 0.0],
                [0.0, 0.0]]
    assert dX[0] == expected


def test_maxpool_invalid_output_dimensions():
    """Pool should raise an error if kernel/stride make output size non‑positive."""
    pool = MaxPool2DLayer(kernel_size=3, stride=2)
    X = [[[1.0] * 2 for _ in range(2)]]  # too small for kernel 3
    with pytest.raises(ValueError, match="output dimensions"):
        pool.forward(X)


# ---------- DenseLayer ----------

def test_dense_forward_backward_accumulation():
    """Check affine transformation and gradient accumulation."""
    dense = DenseLayer(in_features=3, out_features=2)
    X = [1.0, 2.0, 3.0]
    out, backward = dense.forward(X)
    # Manually compute expected output
    expected0 = dense.bias[0] + sum(X[i] * dense.weight[0][i] for i in range(3))
    expected1 = dense.bias[1] + sum(X[i] * dense.weight[1][i] for i in range(3))
    assert out == [expected0, expected1]

    dY = [1.0, -1.0]
    dX = backward(dY)
    # Bias gradients should equal dY
    assert dense.bias_grad == [1.0, -1.0]
    # Weight gradients = dY[o] * X[i]
    for o in range(2):
        for i in range(3):
            assert dense.weight_grad[o][i] == dY[o] * X[i]
    # Input gradient = sum_o dY[o] * weight[o][i]
    for i in range(3):
        expected_dX = dY[0] * dense.weight[0][i] + dY[1] * dense.weight[1][i]
        assert dX[i] == expected_dX


def test_dense_zero_grad():
    dense = DenseLayer(in_features=3, out_features=2)
    X = [1.0, 2.0, 3.0]
    _, backward = dense.forward(X)
    dY = [1.0, -1.0]
    backward(dY)
    dense.zero_grad()
    assert all(v == 0.0 for row in dense.weight_grad for v in row)
    assert all(v == 0.0 for v in dense.bias_grad)


# ---------- ConvNet (Full Model) ----------

def test_convnet_forward_dimensions(sample_input_4x4):
    """Forward pass should produce 2 logits and 4 closures."""
    model = ConvNet()
    logits, closures = model.forward(sample_input_4x4)
    assert len(logits) == 2
    assert len(closures) == 4


def test_convnet_backward_shape(sample_input_4x4):
    """Backward should return a gradient of the same shape as the input (1,4,4)."""
    model = ConvNet()
    _, closures = model.forward(sample_input_4x4)
    dLoss = [1.0, -1.0]
    dX = model.backward(dLoss, closures)
    assert len(dX) == 1
    assert len(dX[0]) == 4
    assert len(dX[0][0]) == 4


def test_convnet_gradients_nonzero(gradient_input_4x4):
    """
    With a non‑constant input and dLoss = [1.0, 0.0], conv weight and bias
    gradients must be non‑zero (no cancellation).
    """
    model = ConvNet()
    model.zero_grad()

    X = gradient_input_4x4
    _, closures = model.forward(X)
    dLoss = [1.0, 0.0]  # only first logit gets gradient
    _ = model.backward(dLoss, closures)

    # Dense bias should be exactly [1.0, 0.0]
    assert model.fc1.bias_grad == [1.0, 0.0]

    # At least one conv weight gradient should be non‑zero
    total_abs_w = sum(
        abs(model.conv1.weight_grad[o][c][m][n])
        for o in range(2)
        for c in range(1)
        for m in range(3)
        for n in range(3)
    )
    assert total_abs_w > 1e-10, "Conv weight gradients should be non‑zero"

    # Conv bias gradients should be non‑zero too
    total_abs_b = sum(abs(model.conv1.bias_grad[o]) for o in range(2))
    assert total_abs_b > 1e-10, "Conv bias gradients should be non‑zero"


def test_convnet_zero_grad_effect(sample_input_4x4):
    """zero_grad() must clear all parameter gradients in the model."""
    model = ConvNet()
    _, closures = model.forward(sample_input_4x4)
    dLoss = [1.0, -1.0]
    _ = model.backward(dLoss, closures)
    # Ensure some gradients are non‑zero before zeroing
    assert any(v != 0.0 for row in model.fc1.weight_grad for v in row)

    model.zero_grad()
    # All gradients should be zero now
    assert all(v == 0.0 for row in model.fc1.weight_grad for v in row)
    assert model.fc1.bias_grad == [0.0, 0.0]
    for o in range(2):
        for c in range(1):
            for m in range(3):
                for n in range(3):
                    assert model.conv1.weight_grad[o][c][m][n] == 0.0


def test_convnet_rejects_wrong_input_shape():
    """ConvNet must reject any input not exactly (1,4,4)."""
    model = ConvNet()
    wrong_height = [[[0.0] * 4 for _ in range(5)]]  # (1,5,4)
    with pytest.raises(ValueError, match="expects input shape"):
        model.forward(wrong_height)


# ---------- Edge Cases ----------

def test_flatten_single_element():
    """Flatten works with a (1,1,1) tensor."""
    flatten = FlattenLayer()
    X = [[[42.0]]]
    flat, backward = flatten.forward(X)
    assert flat == [42.0]
    dX = backward([3.14])
    assert dX[0][0][0] == 3.14


def test_conv2d_single_batch():
    """Conv2D with 1×1 kernel and 1×1 input works correctly."""
    conv = Conv2DLayer(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
    X = [[[5.0]]]
    out, backward = conv.forward(X)
    # output = weight * X + bias
    assert out[0][0][0] == 5.0 * conv.weight[0][0][0][0] + conv.bias[0]
    dY = [[[2.0]]]
    dX = backward(dY)
    # dX = dY * weight
    assert dX[0][0][0] == 2.0 * conv.weight[0][0][0][0]


def test_conv2d_zero_input():
    """If input is all zeros, weight gradients are zero, but input gradient is not."""
    conv = Conv2DLayer(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
    X = [[[0.0, 0.0], [0.0, 0.0]]]
    out, backward = conv.forward(X)
    assert out[0][0][0] == conv.bias[0]
    dY = [[[1.0]]]
    dX = backward(dY)

    # Weight gradients are zero because X is zero
    assert all(v == 0.0 for v in conv.weight_grad[0][0][0])
    assert conv.bias_grad[0] == 1.0
    # Input gradient = dY * weight (since output = weight*X + bias)
    expected_dx = dY[0][0][0] * conv.weight[0][0][0][0]
    assert dX[0][0][0] == expected_dx


def test_dense_invalid_input_length():
    """Dense forward raises error if input length doesn't match in_features."""
    dense = DenseLayer(in_features=3, out_features=2)
    with pytest.raises(ValueError, match="length"):
        dense.forward([1.0, 2.0])


def test_maxpool_overlapping_windows():
    """
    When pooling windows overlap, gradients accumulate at positions that receive
    multiple contributions.
    """
    pool = MaxPool2DLayer(kernel_size=2, stride=1)
    X = [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]]]  # (1,3,3)
    _, backward = pool.forward(X)  # output shape (1,2,2)
    dY = [[[1.0, 2.0],
           [3.0, 4.0]]]
    dX = backward(dY)
    # The argmax positions are:
    # (0,0) → (1,1)  gets 1.0
    # (0,1) → (1,2)  gets 2.0
    # (1,0) → (2,1)  gets 3.0
    # (1,1) → (2,2)  gets 4.0
    expected = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [0.0, 3.0, 4.0]
    ]
    assert dX[0] == expected


# ---------- Numerical Gradient Checks ----------

def test_convnet_input_gradient_finite_difference():
    """Compare autograd dX with finite‑difference approximation."""
    model = ConvNet()
    X = [[[1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [9.0, 10.0, 11.0, 12.0],
          [13.0, 14.0, 15.0, 16.0]]]
    logits, closures = model.forward(X)
    # Loss = sum of squared logits
    dLoss = [2.0 * logits[0], 2.0 * logits[1]]
    dX_autograd = model.backward(dLoss, closures)

    eps = 1e-5
    dX_fd = [[[0.0 for _ in range(4)] for _ in range(4)] for _ in range(1)]
    for c in range(1):
        for i in range(4):
            for j in range(4):
                # Forward with X + eps
                X_plus = [[[v for v in row] for row in X[0]]]
                X_plus[c][i][j] += eps
                logits_plus, _ = model.forward(X_plus)
                loss_plus = logits_plus[0] ** 2 + logits_plus[1] ** 2
                # Forward with X - eps
                X_minus = [[[v for v in row] for row in X[0]]]
                X_minus[c][i][j] -= eps
                logits_minus, _ = model.forward(X_minus)
                loss_minus = logits_minus[0] ** 2 + logits_minus[1] ** 2
                # Central difference
                dX_fd[c][i][j] = (loss_plus - loss_minus) / (2 * eps)

    # Compare relative errors
    for c in range(1):
        for i in range(4):
            for j in range(4):
                autograd_val = dX_autograd[c][i][j]
                fd_val = dX_fd[c][i][j]
                if abs(autograd_val) < 1e-12 and abs(fd_val) < 1e-12:
                    continue
                rel_error = abs(autograd_val - fd_val) / max(abs(autograd_val), abs(fd_val), 1e-12)
                assert rel_error < 1e-4, f"Gradient mismatch at ({c},{i},{j})"


def test_conv2d_weight_gradient_finite_difference():
    """Compare autograd dW and db with finite differences for Conv2D."""
    conv = Conv2DLayer(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
    X = [[[1.0, 2.0], [3.0, 4.0]]]
    out, backward = conv.forward(X)
    # Loss = output^2, so dL/dY = 2*output
    dY = [[[2.0 * out[0][0][0]]]]  # shape (1,1,1)
    backward(dY)

    analytical_w = conv.weight_grad[0][0][0][0]
    analytical_b = conv.bias_grad[0]

    eps = 1e-6

    # ---- Check weight gradient ----
    original = conv.weight[0][0][0][0]
    conv.weight[0][0][0][0] = original + eps
    out_plus, _ = conv.forward(X)
    loss_plus = out_plus[0][0][0] ** 2

    conv.weight[0][0][0][0] = original - eps
    out_minus, _ = conv.forward(X)
    loss_minus = out_minus[0][0][0] ** 2
    conv.weight[0][0][0][0] = original
    numerical_w = (loss_plus - loss_minus) / (2 * eps)
    assert math.isclose(analytical_w, numerical_w, rel_tol=1e-5, abs_tol=1e-7)

    # ---- Check bias gradient ----
    original_b = conv.bias[0]
    conv.bias[0] = original_b + eps
    out_plus, _ = conv.forward(X)
    loss_plus = out_plus[0][0][0] ** 2

    conv.bias[0] = original_b - eps
    out_minus, _ = conv.forward(X)
    loss_minus = out_minus[0][0][0] ** 2
    conv.bias[0] = original_b
    numerical_b = (loss_plus - loss_minus) / (2 * eps)
    assert math.isclose(analytical_b, numerical_b, rel_tol=1e-5, abs_tol=1e-7)


def test_dense_weight_gradient_finite_difference():
    """Compare autograd dW and db with finite differences for Dense."""
    dense = DenseLayer(in_features=2, out_features=1)
    X = [2.0, 3.0]
    out, backward = dense.forward(X)
    dY = [2.0 * out[0]]  # Loss = output^2
    backward(dY)

    analytical_w = dense.weight_grad[0][0]
    analytical_b = dense.bias_grad[0]

    eps = 1e-6

    # ---- Check weight gradient ----
    original = dense.weight[0][0]
    dense.weight[0][0] = original + eps
    out_plus, _ = dense.forward(X)
    loss_plus = out_plus[0] ** 2

    dense.weight[0][0] = original - eps
    out_minus, _ = dense.forward(X)
    loss_minus = out_minus[0] ** 2
    dense.weight[0][0] = original
    numerical_w = (loss_plus - loss_minus) / (2 * eps)
    assert math.isclose(analytical_w, numerical_w, rel_tol=1e-5, abs_tol=1e-7)

    # ---- Check bias gradient ----
    original_b = dense.bias[0]
    dense.bias[0] = original_b + eps
    out_plus, _ = dense.forward(X)
    loss_plus = out_plus[0] ** 2

    dense.bias[0] = original_b - eps
    out_minus, _ = dense.forward(X)
    loss_minus = out_minus[0] ** 2
    dense.bias[0] = original_b
    numerical_b = (loss_plus - loss_minus) / (2 * eps)
    assert math.isclose(analytical_b, numerical_b, rel_tol=1e-5, abs_tol=1e-7)


def test_convnet_full_lifecycle(gradient_input_4x4):
    """
    End‑to‑end test: forward, backward, verify gradients, then zero and verify.
    """
    model = ConvNet()
    X = gradient_input_4x4
    _, closures = model.forward(X)
    dLoss = [1.0, 0.0]
    _ = model.backward(dLoss, closures)

    # ---- Check that gradients are non‑zero ----
    assert any(v != 0.0 for row in model.fc1.weight_grad for v in row)

    total_abs_conv_w = sum(
        abs(model.conv1.weight_grad[o][c][m][n])
        for o in range(2)
        for c in range(1)
        for m in range(3)
        for n in range(3)
    )
    assert total_abs_conv_w > 1e-10, "Conv weight gradients should be non‑zero"

    total_abs_conv_b = sum(abs(model.conv1.bias_grad[o]) for o in range(2))
    assert total_abs_conv_b > 1e-10, "Conv bias gradients should be non‑zero"

    # ---- Zero gradients ----
    model.zero_grad()
    assert all(v == 0.0 for row in model.fc1.weight_grad for v in row)
    all_conv_zero = all(
        model.conv1.weight_grad[o][c][m][n] == 0.0
        for o in range(2)
        for c in range(1)
        for m in range(3)
        for n in range(3)
    )
    assert all_conv_zero, "Conv weight gradients should be zero after zero_grad"