"""
Pytest suite for the ResNet implementation.

Each test is documented to explain what it verifies and why it matters.
We use numerical gradient checking (finite differences) to validate
the analytical gradients – a strong form of correctness testing.
"""

import numpy as np
import pytest
from resnet_arch import (
    Conv2DLayer,
    ElementwiseAdd,
    ResidualBlock,
    ResNetClassifier,
    relu,
)

np.random.seed(42)  # deterministic tests


# ----------------------------------------------------------------------
# Helper for numerical gradient (finite differences)
# ----------------------------------------------------------------------
def numerical_gradient(func, x, eps=1e-5):
    """
    Compute the gradient of a scalar function func(x) w.r.t. x using
    central finite differences.

    This is used to verify that our analytical backward passes are correct.
    """
    x = x.astype(float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x_plus = x.copy()
        x_plus[idx] += eps
        x_minus = x.copy()
        x_minus[idx] -= eps
        grad[idx] = (func(x_plus) - func(x_minus)) / (2 * eps)
        it.iternext()
    return grad


# ======================================================================
# Tests for ReLU
# ======================================================================

def test_relu_forward():
    """Check that ReLU correctly zeroes negative values and keeps positives."""
    X = np.array([[[-1.0, 0.0, 2.0], [3.0, -0.5, 1.0]]])
    out, _ = relu(X)
    expected = np.array([[[0.0, 0.0, 2.0], [3.0, 0.0, 1.0]]])
    np.testing.assert_array_equal(out, expected)


def test_relu_backward():
    """
    Verify ReLU gradient: 1 for positive inputs, 0 for non‑positive.
    This includes the convention at zero (derivative = 0).
    """
    X = np.array([[[-1.0, 0.0, 2.0], [3.0, -0.5, 1.0]]])
    _, backward = relu(X)
    dY = np.ones_like(X)
    dX = backward(dY)
    expected = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    np.testing.assert_array_equal(dX, expected)


def test_relu_derivative_at_zero_is_zero():
    """Document the zero convention explicitly."""
    X = np.array([[-1.0, 0.0, 1.0]])[None, :]  # shape (1,3)
    _, backward = relu(X)
    dY = np.ones_like(X)
    dX = backward(dY)
    expected = np.array([[0.0, 0.0, 1.0]])[None, :]
    np.testing.assert_array_equal(dX, expected)


# ======================================================================
# Tests for ElementwiseAdd
# ======================================================================

def test_elementwise_add_forward():
    """Addition should work element‑wise."""
    adder = ElementwiseAdd()
    A = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    B = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    out, _ = adder.forward(A, B)
    expected = np.array([[[1.5, 2.5], [3.5, 4.5]]])
    np.testing.assert_array_equal(out, expected)


def test_elementwise_add_backward():
    """
    Backward must split gradient equally to both branches.
    This is the core of the residual gradient highway.
    """
    adder = ElementwiseAdd()
    A = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    B = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    _, backward = adder.forward(A, B)
    dZ = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    dA, dB = backward(dZ)
    np.testing.assert_array_equal(dA, dZ)
    np.testing.assert_array_equal(dB, dZ)


def test_elementwise_add_shape_mismatch():
    """Addition should reject incompatible shapes."""
    adder = ElementwiseAdd()
    A = np.random.randn(2, 3, 3)
    B = np.random.randn(2, 4, 4)
    with pytest.raises(ValueError, match="Shape mismatch"):
        adder.forward(A, B)


# ======================================================================
# Tests for Conv2D (educational limitations)
# ======================================================================

def test_conv_invalid_stride():
    """Stride must be positive and equal to 1 in this implementation."""
    with pytest.raises(ValueError, match="stride must be positive"):
        Conv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=0)
    with pytest.raises(ValueError, match="stride=1 only"):
        Conv2DLayer(in_channels=1, out_channels=1, kernel_size=3, stride=2)


# ======================================================================
# Tests for ResidualBlock
# ======================================================================

def test_residual_block_shape_preservation():
    """The residual block should keep spatial dimensions unchanged."""
    block = ResidualBlock(channels=1, kernel_size=3)
    X = np.random.randn(1, 4, 4)
    out, _ = block.forward(X)
    assert out.shape == X.shape


def test_residual_block_gradient_highway():
    """
    **Most important test for Day 40.**

    We zero out both convolution weights so that F(X) = 0.
    Then the block reduces to Y = ReLU(X).
    The gradient should be the ReLU gradient, proving that the identity
    branch is active and carries gradient.
    """
    block = ResidualBlock(channels=1)
    X = np.array([[[1.0, -1.0], [2.0, -2.0]]])  # 1x2x2

    # Make F(X) = 0
    block.conv1.W.fill(0.0)
    block.conv1.b.fill(0.0)
    block.conv2.W.fill(0.0)
    block.conv2.b.fill(0.0)

    out, backward = block.forward(X)
    dY = np.ones_like(out)
    dX = backward(dY)

    # Since F=0, Y = ReLU(X), derivative = 1 where X>0 else 0.
    expected = (X > 0).astype(float)
    np.testing.assert_array_equal(dX, expected)


def test_residual_block_numerical_gradient():
    """
    Compare analytical gradient with numerical gradient for a small random input.
    This verifies that the entire backward chain of the residual block is correct.
    """
    block = ResidualBlock(channels=1)
    X = np.array([[[0.5, -0.2], [0.1, 0.8]]])  # 1x2x2

    def loss_func(X_flat):
        X_reshaped = X_flat.reshape(1, 2, 2)
        out, _ = block.forward(X_reshaped)
        return out.sum()  # scalar loss = sum of outputs

    X_flat = X.flatten()
    grad_num = numerical_gradient(loss_func, X_flat, eps=1e-6)

    out, backward = block.forward(X)
    dY = np.ones_like(out)          # derivative of sum(outputs) w.r.t. outputs
    grad_ana = backward(dY).flatten()

    np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-5, atol=1e-5)


def test_residual_block_gradient_accumulation():
    """
    Verify that calling backward twice without zero_grad accumulates gradients.
    This is essential for mini‑batch training.
    """
    block = ResidualBlock(channels=1)
    X = np.random.randn(1, 4, 4)
    block.zero_grad()

    # First backward
    out, b1 = block.forward(X)
    dY = np.ones_like(out)
    b1(dY)
    dW1_conv1 = block.conv1.dW.copy()
    dW1_conv2 = block.conv2.dW.copy()

    # Second backward (without zero_grad)
    _, b2 = block.forward(X)
    b2(dY)
    dW2_conv1 = block.conv1.dW.copy()
    dW2_conv2 = block.conv2.dW.copy()

    # Single backward from fresh state
    block.zero_grad()
    _, b3 = block.forward(X)
    b3(dY)
    dW_single_conv1 = block.conv1.dW.copy()
    dW_single_conv2 = block.conv2.dW.copy()

    # Accumulation should be additive
    np.testing.assert_allclose(dW2_conv1, dW1_conv1 + dW_single_conv1)
    np.testing.assert_allclose(dW2_conv2, dW1_conv2 + dW_single_conv2)


def test_residual_block_zero_grad():
    """zero_grad() should reset all gradients to zero."""
    block = ResidualBlock(channels=1)
    X = np.random.randn(1, 4, 4)
    block.forward(X)[1](np.ones((1, 4, 4)))
    assert np.any(block.conv1.dW != 0)
    block.zero_grad()
    assert np.all(block.conv1.dW == 0)
    assert np.all(block.conv2.dW == 0)


# ======================================================================
# Tests for full ResNetClassifier
# ======================================================================

def test_resnet_classifier_forward_shape():
    """Forward should produce logits of shape (num_classes,)."""
    model = ResNetClassifier(in_channels=1, hidden_channels=2, num_classes=3)
    X = np.random.randn(1, 4, 4)
    logits, _ = model.forward(X)
    assert logits.shape == (3,)


def test_resnet_classifier_backward_shape():
    """Backward should produce gradient of same shape as input."""
    model = ResNetClassifier(in_channels=1, hidden_channels=2, num_classes=3)
    X = np.random.randn(1, 4, 4)
    logits, closures = model.forward(X)  # noqa: RUF059
    dL = np.random.randn(3)
    dX = model.backward(dL, closures)
    assert dX.shape == X.shape


def test_resnet_classifier_gradient_highway():
    """
    **Full‑network gradient‑highway test.**

    We zero only the residual block convolutions (not stem, not FC).
    Then the only way gradients can reach the input is through the
    identity path inside the residual block. We compare with numerical
    gradient to prove the identity path is functional.
    """
    model = ResNetClassifier(in_channels=1, hidden_channels=2, num_classes=2,
                             input_height=2, input_width=2)

    # Zero only the residual block weights
    model.res_block.conv1.W.fill(0.0)
    model.res_block.conv1.b.fill(0.0)
    model.res_block.conv2.W.fill(0.0)
    model.res_block.conv2.b.fill(0.0)

    X = np.random.randn(1, 2, 2)

    def loss_func(X_flat):
        X_reshaped = X_flat.reshape(1, 2, 2)
        logits, _ = model.forward(X_reshaped)
        return logits.sum()  # scalar loss

    X_flat = X.flatten()
    grad_num = numerical_gradient(loss_func, X_flat, eps=1e-6)

    logits, closures = model.forward(X)
    dL = np.ones_like(logits)  # derivative of sum(logits)
    grad_ana = model.backward(dL, closures).flatten()

    # If the identity path were broken, grad_ana would be wrong.
    np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-5, atol=1e-5)


def test_resnet_classifier_parameter_gradients():
    """All trainable parameters should receive non‑zero gradients (random weights)."""
    model = ResNetClassifier(in_channels=1, hidden_channels=2, num_classes=2)
    X = np.random.randn(1, 4, 4)
    _, closures = model.forward(X)
    dL = np.random.randn(2)
    model.backward(dL, closures)
    assert np.any(model.stem_conv.dW != 0)
    assert np.any(model.res_block.conv1.dW != 0)
    assert np.any(model.res_block.conv2.dW != 0)
    assert np.any(model.fc.dW != 0)


def test_resnet_classifier_invalid_input_channels():
    """Invalid channel count should raise an error."""
    model = ResNetClassifier(in_channels=1, hidden_channels=2)
    X = np.random.randn(2, 4, 4)
    with pytest.raises(ValueError, match="Input channels"):
        model.forward(X)


def test_resnet_classifier_invalid_spatial_dimensions():
    """Invalid spatial size should raise an error (no silent warnings)."""
    model = ResNetClassifier(input_height=4, input_width=4)
    X = np.random.randn(1, 5, 4)
    with pytest.raises(ValueError, match="spatial size"):
        model.forward(X)


def test_resnet_classifier_zero_grad():
    """zero_grad() should reset all parameter gradients."""
    model = ResNetClassifier()
    X = np.random.randn(1, 4, 4)
    _, closures = model.forward(X)
    dL = np.random.randn(2)
    model.backward(dL, closures)
    assert np.any(model.stem_conv.dW != 0)
    model.zero_grad()
    assert np.all(model.stem_conv.dW == 0)
    assert np.all(model.res_block.conv1.dW == 0)
    assert np.all(model.res_block.conv2.dW == 0)
    assert np.all(model.fc.dW == 0)


def test_resnet_classifier_repeated_backward_accumulation():
    """Gradients should accumulate over multiple backward calls."""
    model = ResNetClassifier()
    X = np.random.randn(1, 4, 4)
    model.zero_grad()

    # First backward
    _, closures = model.forward(X)
    dL = np.random.randn(2)
    model.backward(dL, closures)
    grad1 = model.stem_conv.dW.copy()

    # Second backward (without zero_grad)
    _, closures2 = model.forward(X)
    model.backward(dL, closures2)
    grad2 = model.stem_conv.dW.copy()

    # Single backward from zero
    model.zero_grad()
    _, closures3 = model.forward(X)
    model.backward(dL, closures3)
    grad_single = model.stem_conv.dW.copy()

    # Accumulation should be additive
    np.testing.assert_allclose(grad2, grad1 + grad_single)


def test_resnet_classifier_numerical_gradient():
    """
    Full‑network numerical gradient check.
    This is a strong overall correctness test for the whole pipeline.
    """
    model = ResNetClassifier(in_channels=1, hidden_channels=2, num_classes=2,
                             input_height=2, input_width=2)
    X = np.random.randn(1, 2, 2)

    def loss_func(X_flat):
        X_reshaped = X_flat.reshape(1, 2, 2)
        logits, _ = model.forward(X_reshaped)
        return logits.sum()

    X_flat = X.flatten()
    grad_num = numerical_gradient(loss_func, X_flat, eps=1e-6)

    logits, closures = model.forward(X)
    dL = np.ones_like(logits)
    grad_ana = model.backward(dL, closures).flatten()

    np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-5, atol=1e-5)


def test_resnet_gradients_are_finite():
    """Check that all gradients are finite (no NaN/Inf)."""
    model = ResNetClassifier()
    X = np.random.randn(1, 4, 4)
    _, closures = model.forward(X)
    dL = np.random.randn(2)
    dX = model.backward(dL, closures)

    assert np.isfinite(dX).all()
    assert np.isfinite(model.stem_conv.dW).all()
    assert np.isfinite(model.res_block.conv1.dW).all()
    assert np.isfinite(model.res_block.conv2.dW).all()
    assert np.isfinite(model.fc.dW).all()