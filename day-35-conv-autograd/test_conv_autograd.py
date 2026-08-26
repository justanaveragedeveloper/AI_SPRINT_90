"""
test_conv_autograd.py
Comprehensive tests for Day 35 autograd integration.

These tests verify:
  - Conv2D forward shapes and values (single/multi‑channel, multiple filters, stride, padding).
  - Conv2D backward gradients (dX, dW, db) against analytical expectations.
  - Gradient accumulation and zero_grad.
  - Finite‑difference verification for dX, dW, db (both basic and with stride+padding).
  - MaxPool forward, argmax routing, overlapping windows, ties, and accumulation.
  - Autograd graph integration (full backward from scalar loss).
  - Defensive input validation (invalid shapes, channel mismatches, etc.).
"""

import random

import pytest
from conv_autograd import (
    Tensor,
    conv2d,
    maxpool2d,
    shape_list,
)

# ----------------------------------------------------------------------
# Helper: finite‑difference gradient checking
# ----------------------------------------------------------------------

def finite_diff_grad(func, tensor, eps=1e-5):
    """
    Approximate gradient of a scalar function w.r.t. tensor.data.

    This is a robust implementation that:
      1. Flattens the tensor's data and records the shape.
      2. For each element, perturbs it by +eps and -eps,
         computes the loss, and uses the centered difference formula:
             dL/dx ≈ (f(x+eps) - f(x-eps)) / (2*eps)
      3. Reconstructs the gradient in the original nested structure.

    The function does NOT mutate the original tensor; it creates copies.
    """
    original = tensor.data

    def deep_copy(x):
        if not isinstance(x, list):
            return x
        return [deep_copy(sub) for sub in x]

    def set_element(x, path, val):
        node = x
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = val

    # Collect all elements and their index paths.
    elements = []
    paths = []

    def traverse(x, path):
        if not isinstance(x, list):
            elements.append(x)
            paths.append(path)
        else:
            for i, sub in enumerate(x):
                traverse(sub, path + [i])

    traverse(original, [])

    grad_flat = [0.0] * len(elements)

    for idx, (val, path) in enumerate(zip(elements, paths)):
        # Forward perturb
        data_plus = deep_copy(original)
        set_element(data_plus, path, val + eps)
        loss_plus = func(Tensor(data_plus, requires_grad=False))

        # Backward perturb
        data_minus = deep_copy(original)
        set_element(data_minus, path, val - eps)
        loss_minus = func(Tensor(data_minus, requires_grad=False))

        grad_flat[idx] = (loss_plus - loss_minus) / (2 * eps)

    # Reconstruct the gradient structure using the original shape.
    def get_shape(x):
        if not isinstance(x, list):
            return ()
        return (len(x),) + get_shape(x[0]) if x else (0,)

    shape = get_shape(original)

    def build_grad(shape, flat_iter):
        if not shape:
            return next(flat_iter)
        return [build_grad(shape[1:], flat_iter) for _ in range(shape[0])]

    it = iter(grad_flat)
    return build_grad(shape, it)


def finite_diff_scalar_from_tensor(t):
    """
    Sum all elements of a nested list or Tensor to produce a scalar loss.
    This is used as the loss function for finite‑difference tests.
    """
    if isinstance(t, Tensor):
        data = t.data
    else:
        data = t
    if not isinstance(data, list):
        return data
    total = 0.0
    for sub in data:
        total += finite_diff_scalar_from_tensor(sub)
    return total


# ----------------------------------------------------------------------
# Helper: scalar sum operation for autograd graph test
# ----------------------------------------------------------------------

def sum_tensor(t: Tensor) -> Tensor:
    """
    Sum all elements of a tensor and return a scalar Tensor with autograd.
    This is used to create a scalar loss that can call .backward().
    The backward of sum is simply ones (each element gets gradient = 1).
    """
    total = finite_diff_scalar_from_tensor(t)
    out = Tensor(total, requires_grad=True, _children=(t,))

    def _backward():
        if t.requires_grad:
            if t.grad is None:
                t.grad = zeros_like(t.data)
            # Propagate gradient of 1.0 to all elements.
            def add_ones(x):
                if not isinstance(x, list):
                    return 1.0
                return [add_ones(sub) for sub in x]
            grad_ones = add_ones(t.data)

            def add_grad(a, b):
                if isinstance(a, list):
                    for i, sub in enumerate(a):
                        add_grad(sub, b[i])
                else:
                    a += b
            add_grad(t.grad, grad_ones)

    out._backward = _backward
    return out


def zeros_like(x):
    """Create a nested list of zeros with the same structure as x."""
    if not isinstance(x, list):
        return 0.0
    return [zeros_like(sub) for sub in x]


# ----------------------------------------------------------------------
# Conv2D Forward Tests
# ----------------------------------------------------------------------

def test_conv2d_forward_shape_single():
    """Single channel, single filter, stride=1, no padding."""
    X = Tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]]])
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]])
    out = conv2d(X, W, None)
    assert shape_list(out.data) == (1, 2, 2)

def test_conv2d_forward_multi_channel():
    """Two input channels, one output filter."""
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]])
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]],
                 [[0.0, 1.0], [1.0, 0.0]]]])
    out = conv2d(X, W, None)
    assert shape_list(out.data) == (1, 1, 1)
    assert abs(out.data[0][0][0] - 18.0) < 1e-6

def test_conv2d_forward_multiple_output():
    """Two output filters."""
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]])
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]]])
    out = conv2d(X, W, None)
    assert shape_list(out.data) == (2, 1, 1)
    assert abs(out.data[0][0][0] - 5.0) < 1e-6
    assert abs(out.data[1][0][0] - 5.0) < 1e-6

def test_conv2d_forward_padding_stride():
    """Stride=2, padding=1."""
    X = Tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]]])
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]])
    out = conv2d(X, W, None, stride=2, padding=1)
    assert shape_list(out.data) == (1, 2, 2)
    assert abs(out.data[0][0][0] - 1.0) < 1e-6
    assert abs(out.data[0][0][1] - 3.0) < 1e-6
    assert abs(out.data[0][1][0] - 7.0) < 1e-6
    assert abs(out.data[0][1][1] - 14.0) < 1e-6


# ----------------------------------------------------------------------
# Conv2D Backward Tests (analytical)
# ----------------------------------------------------------------------

def test_conv2d_backward_dx_simple():
    """Verify dX for one output, one filter, no padding."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]], requires_grad=False)
    out = conv2d(X, W, None)
    out.grad = [[[1.0]]]
    out._backward()
    expected = [[[1.0, 0.0], [0.0, 1.0]]]
    assert X.grad == expected

def test_conv2d_backward_dw_simple():
    """Verify dW."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=False)
    W = Tensor([[[[0.0, 0.0],
                  [0.0, 0.0]]]], requires_grad=True)
    out = conv2d(X, W, None)
    out.grad = [[[1.0]]]
    out._backward()
    expected = [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert W.grad == expected

def test_conv2d_backward_db():
    """Verify bias gradient."""
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=False)
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=False)
    b = Tensor([0.0], requires_grad=True)
    out = conv2d(X, W, b)
    out.grad = [[[1.0]]]
    out._backward()
    assert b.grad == [1.0]

def test_conv2d_gradient_accumulation():
    """Call backward twice; gradients should accumulate (add)."""
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)
    b = Tensor([0.0], requires_grad=True)
    out = conv2d(X, W, b)
    out.grad = [[[1.0]]]
    out._backward()
    out.grad = [[[1.0]]]
    out._backward()
    expected_dX = [[[2.0, 0.0], [0.0, 2.0]]]
    assert X.grad == expected_dX
    expected_dW = [[[[2.0, 4.0], [6.0, 8.0]]]]
    assert W.grad == expected_dW
    assert b.grad == [2.0]

def test_conv2d_zero_grad():
    """After zero_grad, gradients should be None."""
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]], requires_grad=True)
    b = Tensor([0.0], requires_grad=True)
    out = conv2d(X, W, b)
    out.grad = [[[1.0]]]
    out._backward()
    X.zero_grad()
    W.zero_grad()
    b.zero_grad()
    assert X.grad is None
    assert W.grad is None
    assert b.grad is None

def test_conv2d_overlapping_receptive_fields():
    """
    Check that dX accumulates correctly when receptive fields overlap.
    With kernel [[1,0],[0,1]] and all dY=1, we manually compute expected dX.
    """
    X = Tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]], requires_grad=False)
    out = conv2d(X, W, None)
    out.grad = [[[1.0, 1.0],
                 [1.0, 1.0]]]
    out._backward()
    # Expected dX (see comments in the test for derivation)
    expected = [[[1.0, 1.0, 0.0],
                 [1.0, 2.0, 1.0],
                 [0.0, 1.0, 1.0]]]
    assert X.grad == expected

def test_conv2d_stride_greater_than_one():
    """Stride=2: check that only certain input positions receive gradients."""
    X = Tensor([[[1.0, 2.0, 3.0, 4.0],
                 [5.0, 6.0, 7.0, 8.0],
                 [9.0,10.0,11.0,12.0],
                 [13.0,14.0,15.0,16.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]], requires_grad=False)
    out = conv2d(X, W, None, stride=2)
    out.grad = [[[1.0, 1.0],
                 [1.0, 1.0]]]
    out._backward()
    expected = [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ]
    assert X.grad[0] == expected

def test_conv2d_padding_affects_dx():
    """With padding, the gradient shape must match the original input."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]], requires_grad=False)
    out = conv2d(X, W, None, padding=1)
    out.grad = [[[1.0]*3 for _ in range(3)]]
    out._backward()
    # We only check shape; exact values are verified by finite‑difference.
    assert shape_list(X.grad) == (1, 2, 2)


# ----------------------------------------------------------------------
# Finite‑difference tests (using manual out.grad to isolate conv2d backward)
# ----------------------------------------------------------------------

def test_conv2d_finite_difference_all_grads_basic():
    """
    Compare analytical gradients (dX, dW, db) with finite differences for a
    simple convolution (1 channel, 1 filter, stride=1, no padding).
    """
    random.seed(42)
    X_data = [[[random.random() for _ in range(3)] for _ in range(3)]]
    W_data = [[[[random.random() for _ in range(2)] for _ in range(2)]]]
    b_data = [random.random()]

    X = Tensor(X_data, requires_grad=True)
    W = Tensor(W_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    def loss_fn(Xt, Wt, bt):
        out = conv2d(Xt, Wt, bt)
        return finite_diff_scalar_from_tensor(out)

    # Analytical gradients using manual out.grad (proven correct)
    out = conv2d(X, W, b)
    out.grad = [[[1.0] * shape_list(out.data)[2] for _ in range(shape_list(out.data)[1])]]
    out._backward()
    grad_ana_X = X.grad
    grad_ana_W = W.grad
    grad_ana_b = b.grad

    eps = 1e-5
    grad_num_X = finite_diff_grad(lambda Xt: loss_fn(Xt, W, b), X, eps)
    grad_num_W = finite_diff_grad(lambda Wt: loss_fn(X, Wt, b), W, eps)
    grad_num_b = finite_diff_grad(lambda bt: loss_fn(X, W, bt), b, eps)

    def flatten_grad(grad):
        res = []
        def rec(x):
            if not isinstance(x, list):
                res.append(x)
            else:
                for sub in x:
                    rec(sub)
        rec(grad)
        return res

    flat_ana_X = flatten_grad(grad_ana_X)
    flat_num_X = flatten_grad(grad_num_X)
    flat_ana_W = flatten_grad(grad_ana_W)
    flat_num_W = flatten_grad(grad_num_W)

    for a, n in zip(flat_ana_X, flat_num_X):
        assert abs(a - n) < 1e-4
    for a, n in zip(flat_ana_W, flat_num_W):
        assert abs(a - n) < 1e-4
    for a, n in zip(grad_ana_b, grad_num_b):
        assert abs(a - n) < 1e-4

def test_conv2d_finite_difference_stride_padding():
    """
    Finite‑difference test with stride=2, padding=1, multiple channels and filters.
    This catches indexing errors that might not appear in the basic case.
    """
    random.seed(123)
    X_data = [[[random.random() for _ in range(5)] for _ in range(5)] for _ in range(2)]
    W_data = [[[[random.random() for _ in range(3)] for _ in range(3)] for _ in range(2)] for _ in range(2)]
    b_data = [random.random(), random.random()]

    X = Tensor(X_data, requires_grad=True)
    W = Tensor(W_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    def loss_fn(Xt, Wt, bt):
        out = conv2d(Xt, Wt, bt, stride=2, padding=1)
        return finite_diff_scalar_from_tensor(out)

    out = conv2d(X, W, b, stride=2, padding=1)
    out.grad = [[[1.0]*shape_list(out.data)[2] for _ in range(shape_list(out.data)[1])] for _ in range(shape_list(out.data)[0])]
    out._backward()
    grad_ana_X = X.grad
    grad_ana_W = W.grad
    grad_ana_b = b.grad

    eps = 1e-5
    grad_num_X = finite_diff_grad(lambda Xt: loss_fn(Xt, W, b), X, eps)
    grad_num_W = finite_diff_grad(lambda Wt: loss_fn(X, Wt, b), W, eps)
    grad_num_b = finite_diff_grad(lambda bt: loss_fn(X, W, bt), b, eps)

    def flatten_grad(grad):
        res = []
        def rec(x):
            if not isinstance(x, list):
                res.append(x)
            else:
                for sub in x:
                    rec(sub)
        rec(grad)
        return res

    flat_ana_X = flatten_grad(grad_ana_X)
    flat_num_X = flatten_grad(grad_num_X)
    flat_ana_W = flatten_grad(grad_ana_W)
    flat_num_W = flatten_grad(grad_num_W)

    for a, n in zip(flat_ana_X, flat_num_X):
        assert abs(a - n) < 1e-4
    for a, n in zip(flat_ana_W, flat_num_W):
        assert abs(a - n) < 1e-4
    for a, n in zip(grad_ana_b, grad_num_b):
        assert abs(a - n) < 1e-4


# ----------------------------------------------------------------------
# Autograd Graph Integration Test (true graph-level using sum_tensor)
# ----------------------------------------------------------------------

def test_conv2d_graph_integration():
    """
    This test exercises the full autograd graph:
        X, W, b -> Conv2D -> sum -> loss -> backward()
    It verifies that the public backward() method traverses the graph and
    computes gradients for all parameters.
    """
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    W = Tensor([[[[1.0, 0.0],
                  [0.0, 1.0]]]], requires_grad=True)
    b = Tensor([0.0], requires_grad=True)
    out = conv2d(X, W, b)
    loss = sum_tensor(out)
    loss.backward()
    # Verify gradients exist and have correct shapes.
    assert X.grad is not None
    assert W.grad is not None
    assert b.grad is not None
    assert shape_list(X.grad) == (1, 2, 2)
    assert shape_list(W.grad) == (1, 1, 2, 2)
    assert shape_list(b.grad) == (1,)


# ----------------------------------------------------------------------
# MaxPool Tests
# ----------------------------------------------------------------------

def test_maxpool_forward_shape():
    X = Tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]]])
    out = maxpool2d(X, kernel_size=2, stride=2)
    assert shape_list(out.data) == (1, 1, 1)

def test_maxpool_backward_argmax_routing():
    """Gradient goes only to the max position (4 at (1,1))."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    out = maxpool2d(X, kernel_size=2, stride=2)
    out.grad = [[[1.0]]]
    out._backward()
    expected = [[[0.0, 0.0],
                 [0.0, 1.0]]]
    assert X.grad == expected

def test_maxpool_overlapping_windows():
    """With stride=1, each input may win multiple windows; gradients accumulate."""
    X = Tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]]], requires_grad=True)
    out = maxpool2d(X, kernel_size=2, stride=1)
    out.grad = [[[1.0, 1.0],
                 [1.0, 1.0]]]
    out._backward()
    expected = [[[0.0, 0.0, 0.0],
                 [0.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]]]
    assert X.grad == expected

def test_maxpool_tie_behavior():
    """All values equal: tie‑break chooses top‑left (0,0)."""
    X = Tensor([[[1.0, 1.0],
                 [1.0, 1.0]]], requires_grad=True)
    out = maxpool2d(X, kernel_size=2, stride=2)
    out.grad = [[[1.0]]]
    out._backward()
    expected = [[[1.0, 0.0],
                 [0.0, 0.0]]]
    assert X.grad == expected

def test_maxpool_gradient_accumulation():
    """Call backward twice -> gradients double."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    out = maxpool2d(X, kernel_size=2, stride=2)
    out.grad = [[[1.0]]]
    out._backward()
    out.grad = [[[1.0]]]
    out._backward()
    expected = [[[0.0, 0.0],
                 [0.0, 2.0]]]
    assert X.grad == expected

def test_maxpool_padding():
    """Padding adds border; check shape and that backward doesn't crash."""
    X = Tensor([[[1.0, 2.0],
                 [3.0, 4.0]]], requires_grad=True)
    out = maxpool2d(X, kernel_size=2, stride=1, padding=1)
    assert shape_list(out.data) == (1, 3, 3)
    out.grad = [[[1.0]*3 for _ in range(3)]]
    out._backward()
    assert shape_list(X.grad) == (1, 2, 2)


# ----------------------------------------------------------------------
# Defensive / Edge Cases
# ----------------------------------------------------------------------

def test_conv2d_invalid_stride():
    X = Tensor([[[1.0]]])
    W = Tensor([[[[1.0]]]])
    with pytest.raises(ValueError, match="stride"):
        conv2d(X, W, None, stride=0)

def test_conv2d_invalid_padding():
    X = Tensor([[[1.0]]])
    W = Tensor([[[[1.0]]]])
    with pytest.raises(ValueError, match="padding"):
        conv2d(X, W, None, padding=-1)

def test_conv2d_output_dimension_zero():
    X = Tensor([[[1.0, 2.0]]])  # width=2, kernel=3 → output width = (2-3)//1+1 = 0
    W = Tensor([[[[1.0, 2.0, 3.0]]]])
    with pytest.raises(ValueError, match="Output dimension"):
        conv2d(X, W, None, stride=1, padding=0)

def test_conv2d_invalid_shape_input():
    X_bad = Tensor([[1.0, 2.0]])  # 2D instead of 3D
    W = Tensor([[[[1.0]]]])
    with pytest.raises(ValueError, match="Input must be 3D"):
        conv2d(X_bad, W, None)

def test_conv2d_channel_mismatch():
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # C=1
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]],
                 [[0.0, 1.0], [1.0, 0.0]]]])  # Cin=2
    with pytest.raises(ValueError, match="Weight input channels"):
        conv2d(X, W, None)

def test_conv2d_bias_shape_mismatch():
    X = Tensor([[[1.0, 2.0], [3.0, 4.0]]])
    W = Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # Cout=1
    b = Tensor([0.0, 1.0])  # length 2
    with pytest.raises(ValueError, match="Bias length"):
        conv2d(X, W, b)

def test_maxpool_invalid_padding():
    X = Tensor([[[1.0]]])
    with pytest.raises(ValueError):
        maxpool2d(X, kernel_size=1, padding=-1)

def test_maxpool_invalid_stride():
    X = Tensor([[[1.0]]])
    with pytest.raises(ValueError):
        maxpool2d(X, kernel_size=1, stride=0)

def test_maxpool_output_dimension_zero():
    X = Tensor([[[1.0, 2.0]]])  # width=2, kernel=3 → output width = 0
    with pytest.raises(ValueError):
        maxpool2d(X, kernel_size=3, stride=1, padding=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])