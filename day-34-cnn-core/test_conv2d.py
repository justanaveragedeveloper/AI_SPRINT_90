import math

import pytest
from conv2d import (
    AvgPool2D,
    Conv2D,
    MaxPool2D,
    _validate_tensor3d,
    pad2d,
    unpad,
)


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def assert_close(actual, expected, rtol=1e-4, atol=1e-6):
    for a, e in zip(flatten(actual), flatten(expected)):
        assert math.isclose(a, e, rel_tol=rtol, abs_tol=atol), f"{a} != {e}"


def numerical_loss(conv, X):
    output = conv.forward(X)
    return sum(flatten(output))


def finite_difference_central(conv, X, eps=1e-6):
    """Compute numerical gradients using central differences (more accurate)."""
    base_loss = numerical_loss(conv, X)  # noqa: F841

    dX = [[[0.0 for _ in row] for row in channel] for channel in X]
    for c in range(len(X)):
        for i in range(len(X[0])):
            for j in range(len(X[0][0])):
                X_plus = [[row[:] for row in channel] for channel in X]
                X_minus = [[row[:] for row in channel] for channel in X]
                X_plus[c][i][j] += eps
                X_minus[c][i][j] -= eps
                loss_plus = numerical_loss(conv, X_plus)
                loss_minus = numerical_loss(conv, X_minus)
                dX[c][i][j] = (loss_plus - loss_minus) / (2 * eps)

    dW = [
        [[[0.0 for _ in row] for row in channel] for channel in output_channel]
        for output_channel in conv.weight
    ]
    for o in range(conv.out_channels):
        for c in range(conv.in_channels):
            for m in range(conv.kernel_size[0]):
                for n in range(conv.kernel_size[1]):
                    orig = conv.weight[o][c][m][n]
                    conv.weight[o][c][m][n] = orig + eps
                    loss_plus = numerical_loss(conv, X)
                    conv.weight[o][c][m][n] = orig - eps
                    loss_minus = numerical_loss(conv, X)
                    dW[o][c][m][n] = (loss_plus - loss_minus) / (2 * eps)
                    conv.weight[o][c][m][n] = orig

    db = []
    for o in range(conv.out_channels):
        orig = conv.bias[o]
        conv.bias[o] = orig + eps
        loss_plus = numerical_loss(conv, X)
        conv.bias[o] = orig - eps
        loss_minus = numerical_loss(conv, X)
        db.append((loss_plus - loss_minus) / (2 * eps))
        conv.bias[o] = orig

    return dX, dW, db


def test_pad2d_and_unpad():
    X = [[[1.0, 2.0], [3.0, 4.0]]]
    padded = pad2d(X, 1)
    assert padded == [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ]
    assert unpad(padded, 1) == X


def test_conv2d_output_shape():
    conv = Conv2D(1, 2, kernel_size=3, stride=2, padding=1)
    X = [[[1.0] * 5 for _ in range(5)]]
    output = conv.forward(X)
    H_out = (5 + 2 * 1 - 3) // 2 + 1
    assert H_out == 3
    assert len(output) == 2
    assert len(output[0]) == 3
    assert len(output[0][0]) == 3


def test_conv2d_rectangular_kernel():
    """Test a non-square kernel (height=3, width=2)."""
    conv = Conv2D(1, 1, kernel_size=(3, 2), stride=1, padding=0)
    conv.weight = [[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]]  # 3x2 kernel
    conv.bias = [0.0]
    X = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
    # Manually compute expected:
    # Output shape: (4 - 3)//1+1 = 2, (3 - 2)//1+1 = 2
    # Y[0,0] = X[0,0]*1 + X[0,1]*0 + X[1,0]*0 + X[1,1]*1 + X[2,0]*1 + X[2,1]*0
    #        = 1 + 5 + 7 = 13
    # Y[0,1] = X[0,1]*1 + X[0,2]*0 + X[1,1]*0 + X[1,2]*1 + X[2,1]*1 + X[2,2]*0
    #        = 2 + 6 + 8 = 16
    # Y[1,0] = X[1,0]*1 + X[1,1]*0 + X[2,0]*0 + X[2,1]*1 + X[3,0]*1 + X[3,1]*0
    #        = 4 + 8 + 10 = 22
    # Y[1,1] = X[1,1]*1 + X[1,2]*0 + X[2,1]*0 + X[2,2]*1 + X[3,1]*1 + X[3,2]*0
    #        = 5 + 9 + 11 = 25
    expected = [[[13.0, 16.0], [22.0, 25.0]]]
    assert conv.forward(X) == expected


def test_conv2d_known_cross_correlation():
    conv = Conv2D(1, 1, kernel_size=2)
    conv.weight = [[[[1.0, 0.0], [0.0, 1.0]]]]
    conv.bias = [0.0]
    X = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    assert conv.forward(X) == [[[6.0, 8.0], [12.0, 14.0]]]


def test_conv2d_bias():
    conv = Conv2D(1, 1, 2)
    conv.weight = [[[[1.0, 0.0], [0.0, 1.0]]]]
    conv.bias = [1.0]
    X = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    assert conv.forward(X) == [[[7.0, 9.0], [13.0, 15.0]]]


def test_conv2d_multichannel():
    conv = Conv2D(2, 1, 2)
    conv.weight = [
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    ]
    conv.bias = [0.0]
    X = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    ]
    assert conv.forward(X) == [[[1.0, 3.0], [5.0, 5.0]]]


def test_conv2d_padding_and_stride():
    conv = Conv2D(1, 1, 2, stride=2, padding=1)
    conv.weight = [[[[1.0, 1.0], [1.0, 1.0]]]]
    conv.bias = [0.0]
    X = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
    assert conv.forward(X) == [[[1.0, 5.0], [11.0, 28.0]]]


def test_conv2d_backward_hand_calculation():
    conv = Conv2D(1, 1, 2)
    conv.weight = [[[[0.5, 0.2], [0.1, 0.3]]]]
    conv.bias = [0.1]
    X = [[[1.0, 2.0], [3.0, 4.0]]]
    conv.forward(X)
    dX, dW, db = conv.backward([[[1.0]]])
    assert dX == [[[0.5, 0.2], [0.1, 0.3]]]
    assert dW == [[[[1.0, 2.0], [3.0, 4.0]]]]
    assert db == [1.0]


def test_conv2d_backward_finite_difference_central():
    conv = Conv2D(2, 2, 2, stride=2, padding=1, seed=7)
    X = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[-1.0, 0.5, 2.0], [1.5, -2.0, 0.0], [3.0, 1.0, -0.5]],
    ]
    Y = conv.forward(X)
    dY = [[[1.0 for _ in row] for row in channel] for channel in Y]
    analytical = conv.backward(dY)
    numerical = finite_difference_central(conv, X)
    for a, n in zip(analytical, numerical):
        assert_close(a, n)


def test_conv2d_backward_invalid_dY():
    conv = Conv2D(1, 1, 2)
    X = [[[1.0, 2.0], [3.0, 4.0]]]
    conv.forward(X)
    # Wrong shape: should be (1,1,1) but we pass (1,2,1)
    with pytest.raises(ValueError, match="dY shape must be"):
        conv.backward([[[1.0], [2.0]]])


def test_maxpool_forward_backward():
    X = [
        [
            [1.0, 3.0, 2.0, 4.0],
            [5.0, 6.0, 1.0, 2.0],
            [0.0, 2.0, 8.0, 3.0],
            [1.0, 4.0, 2.0, 1.0],
        ]
    ]
    pool = MaxPool2D(kernel_size=2, stride=2)
    assert pool.forward(X) == [[[6.0, 4.0], [4.0, 8.0]]]
    assert pool.backward([[[1.0, 2.0], [3.0, 4.0]]]) == [
        [
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
        ]
    ]


def test_maxpool_tie_behavior():
    pool = MaxPool2D(kernel_size=2, stride=1)
    pool.forward([[[1.0, 1.0], [1.0, 0.0]]])
    assert pool.backward([[[5.0]]]) == [[[5.0, 0.0], [0.0, 0.0]]]


def test_pool_backward_invalid_dY():
    pool = MaxPool2D(kernel_size=2, stride=2)
    X = [
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    ]
    pool.forward(X)
    with pytest.raises(ValueError, match="dY shape does not match"):
        pool.backward([[[1.0, 2.0]]])  # wrong width


def test_avgpool_forward_backward():
    pool = AvgPool2D(kernel_size=2, stride=2)
    X = [[[1.0, 3.0], [5.0, 7.0]]]
    assert pool.forward(X) == [[[4.0]]]
    assert pool.backward([[[2.0]]]) == [[[0.5, 0.5], [0.5, 0.5]]]


def test_avgpool_overlap_accumulates():
    pool = AvgPool2D(kernel_size=2, stride=1)
    X = [[[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]]]
    pool.forward(X)
    assert pool.backward([[[1.0, 1.0], [1.0, 1.0]]]) == [
        [
            [0.25, 0.5, 0.25],
            [0.5, 1.0, 0.5],
            [0.25, 0.5, 0.25],
        ]
    ]


def test_malformed_tensors():
    # Empty tensor
    with pytest.raises(ValueError, match="non-empty"):
        _validate_tensor3d([], "X")
    # Channel not a list
    with pytest.raises(ValueError, match="must contain non-empty"):
        _validate_tensor3d(["foo"], "X")
    # Empty channel
    with pytest.raises(ValueError, match="non-empty"):
        _validate_tensor3d([[]], "X")
    # Mismatched heights
    with pytest.raises(ValueError, match="same height"):
        _validate_tensor3d([[[1.0], [2.0]], [[3.0]]], "X")
    # Zero width
    with pytest.raises(ValueError, match="width must be positive"):
        _validate_tensor3d([[[], []]], "X")
    # Non-rectangular rows
    with pytest.raises(ValueError, match="same width"):
        _validate_tensor3d([[[1.0, 2.0], [3.0]]], "X")
    # NaN
    with pytest.raises(ValueError, match="NaN"):
        _validate_tensor3d([[[float("nan")]]], "X")
    # Inf
    with pytest.raises(ValueError, match="Inf"):
        _validate_tensor3d([[[float("inf")]]], "X")


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: Conv2D(0, 1, 3),
        lambda: Conv2D(1, 0, 3),
        lambda: Conv2D(1, 1, 0),
        lambda: Conv2D(1, 1, (2, 0)),
        lambda: Conv2D(1, 1, 3, stride=0),
        lambda: Conv2D(1, 1, 3, padding=-1),
    ],
)
def test_invalid_conv_parameters(constructor):
    with pytest.raises(ValueError):
        constructor()


def test_invalid_pool_parameters():
    with pytest.raises(ValueError):
        MaxPool2D(0)
    with pytest.raises(ValueError):
        MaxPool2D(kernel_size=2, stride=0)
    with pytest.raises(ValueError):
        AvgPool2D(0)


def test_pool_backward_requires_forward():
    pool = MaxPool2D()
    with pytest.raises(RuntimeError):
        pool.backward([[[1.0]]])


def test_reproducible_initialization():
    a = Conv2D(1, 1, 3, seed=123)
    b = Conv2D(1, 1, 3, seed=123)
    assert a.weight == b.weight
    assert a.bias == b.bias
