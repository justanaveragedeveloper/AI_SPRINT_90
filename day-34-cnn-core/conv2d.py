"""
Day 34: 2D convolution and pooling operations from scratch.

Tensor layout:
    Input X:  (C, H, W)
    Conv W:  (C_out, C_in, K_H, K_W)
    Conv Y:  (C_out, H_out, W_out)

Conv2D uses the deep-learning cross-correlation convention:
kernels are not spatially flipped.
"""

import math
import random
from typing import TypeAlias

# Type aliases using built-in generics
Tensor3D: TypeAlias = list[list[list[float]]]
Kernel4D: TypeAlias = list[list[list[list[float]]]]


def _assert_finite_tensor(X: Tensor3D, name: str) -> None:
    """Raise ValueError if any element is NaN or infinite."""
    if not isinstance(X, list):
        raise TypeError(f"{name} must be a list")
    for c, channel in enumerate(X):
        if not isinstance(channel, list):
            raise TypeError(f"{name}[{c}] must be a list")
        for i, row in enumerate(channel):
            if not isinstance(row, list):
                raise TypeError(f"{name}[{c}][{i}] must be a list")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"{name}[{c}][{i}][{j}] must be a number")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"{name} contains NaN or Inf at [{c}][{i}][{j}]")


def _validate_tensor3d(X: Tensor3D, name: str) -> tuple[int, int, int]:
    """Validate a non-empty rectangular (C, H, W) tensor and check finite values."""
    if not isinstance(X, list) or not X:
        raise ValueError(f"{name} must be a non-empty 3D list")

    if any(not isinstance(channel, list) or not channel for channel in X):
        raise ValueError(f"{name} must contain non-empty channels")

    height = len(X[0])
    if any(len(channel) != height for channel in X):
        raise ValueError(f"{name} channels must have the same height")

    width = len(X[0][0])
    if width == 0:
        raise ValueError(f"{name} width must be positive")

    if any(any(len(row) != width for row in channel) for channel in X):
        raise ValueError(f"{name} rows must have the same width")

    _assert_finite_tensor(X, name)
    return len(X), height, width


def pad2d(X: Tensor3D, padding: int) -> Tensor3D:
    """Return X zero-padded on both spatial dimensions."""
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    channels, height, width = _validate_tensor3d(X, "X")
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding

    padded = [
        [[0.0 for _ in range(padded_width)] for _ in range(padded_height)]
        for _ in range(channels)
    ]

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                padded[c][i + padding][j + padding] = X[c][i][j]
    return padded


def unpad(X_pad: Tensor3D, padding: int) -> Tensor3D:
    """Remove spatial zero-padding from a tensor."""
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    channels, padded_height, padded_width = _validate_tensor3d(X_pad, "X_pad")

    if padding == 0:
        return [
            [
                [X_pad[c][i][j] for j in range(padded_width)]
                for i in range(padded_height)
            ]
            for c in range(channels)
        ]

    if 2 * padding >= padded_height or 2 * padding >= padded_width:
        raise ValueError("padding is too large for tensor")

    height = padded_height - 2 * padding
    width = padded_width - 2 * padding
    return [
        [
            [X_pad[c][i + padding][j + padding] for j in range(width)]
            for i in range(height)
        ]
        for c in range(channels)
    ]


class Conv2D:
    """First-principles 2D cross-correlation layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        seed: int = 42,
    ):
        for value, name in (
            (in_channels, "in_channels"),
            (out_channels, "out_channels"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if isinstance(kernel_size, int) and not isinstance(kernel_size, bool):
            if kernel_size <= 0:
                raise ValueError("kernel_size must be positive")
            self.kernel_size = (kernel_size, kernel_size)
        elif (
            isinstance(kernel_size, tuple)
            and len(kernel_size) == 2
            and all(
                isinstance(k, int) and not isinstance(k, bool) and k > 0
                for k in kernel_size
            )
        ):
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                "kernel_size must be a positive int or a positive (height, width) tuple"
            )

        if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
            raise ValueError("stride must be a positive integer")
        if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
            raise ValueError("padding must be a non-negative integer")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        kernel_height, kernel_width = self.kernel_size
        fan_in = in_channels * kernel_height * kernel_width
        scale = math.sqrt(2.0 / fan_in)

        # Local RNG avoids global state pollution
        rng = random.Random(seed)
        self.weight = [
            [
                [
                    [rng.uniform(-scale, scale) for _ in range(kernel_width)]
                    for _ in range(kernel_height)
                ]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias = [0.0 for _ in range(out_channels)]

        self._cache_X: Tensor3D | None = None
        self._cache_X_pad: Tensor3D | None = None
        self._cache_output_shape: tuple[int, int] | None = None

    def _validate_input(self, X: Tensor3D) -> tuple[int, int, int]:
        channels, height, width = _validate_tensor3d(X, "X")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")
        return channels, height, width

    def forward(self, X: Tensor3D) -> Tensor3D:
        _, height_in, width_in = self._validate_input(X)
        kernel_height, kernel_width = self.kernel_size
        stride = self.stride
        padding = self.padding

        height_out = (height_in + 2 * padding - kernel_height) // stride + 1
        width_out = (width_in + 2 * padding - kernel_width) // stride + 1
        if height_out <= 0 or width_out <= 0:
            raise ValueError("Kernel larger than padded input")

        X_pad = pad2d(X, padding)
        Y = [
            [[0.0 for _ in range(width_out)] for _ in range(height_out)]
            for _ in range(self.out_channels)
        ]

        for o in range(self.out_channels):
            for i in range(height_out):
                for j in range(width_out):
                    value = self.bias[o]
                    h_start = i * stride
                    w_start = j * stride
                    for c in range(self.in_channels):
                        for m in range(kernel_height):
                            for n in range(kernel_width):
                                value += (
                                    X_pad[c][h_start + m][w_start + n]
                                    * self.weight[o][c][m][n]
                                )
                    Y[o][i][j] = value

        self._cache_X = [
            [[X[c][i][j] for j in range(width_in)] for i in range(height_in)]
            for c in range(self.in_channels)
        ]
        self._cache_X_pad = X_pad
        self._cache_output_shape = (height_out, width_out)
        return Y

    def backward(self, dY: Tensor3D) -> tuple[Tensor3D, Kernel4D, list[float]]:
        if self._cache_X is None:
            raise RuntimeError("Call forward() before backward().")

        height_out, width_out = self._cache_output_shape
        _, height_in, width_in = self._validate_input(self._cache_X)

        dY_shape = _validate_tensor3d(dY, "dY")
        expected_shape = (self.out_channels, height_out, width_out)
        if dY_shape != expected_shape:
            raise ValueError(f"dY shape must be {expected_shape}")

        kernel_height, kernel_width = self.kernel_size

        # db
        db = [0.0 for _ in range(self.out_channels)]
        for o in range(self.out_channels):
            for i in range(height_out):
                for j in range(width_out):
                    db[o] += dY[o][i][j]

        # dW
        dW = [
            [
                [[0.0 for _ in range(kernel_width)] for _ in range(kernel_height)]
                for _ in range(self.in_channels)
            ]
            for _ in range(self.out_channels)
        ]
        for o in range(self.out_channels):
            for c in range(self.in_channels):
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        total = 0.0
                        for i in range(height_out):
                            for j in range(width_out):
                                total += (
                                    dY[o][i][j]
                                    * self._cache_X_pad[c][i * self.stride + m][
                                        j * self.stride + n
                                    ]
                                )
                        dW[o][c][m][n] = total

        # dX
        height_pad = height_in + 2 * self.padding
        width_pad = width_in + 2 * self.padding
        dX_pad = [
            [[0.0 for _ in range(width_pad)] for _ in range(height_pad)]
            for _ in range(self.in_channels)
        ]
        for o in range(self.out_channels):
            for i in range(height_out):
                for j in range(width_out):
                    grad = dY[o][i][j]
                    if grad == 0.0:
                        continue
                    h_start = i * self.stride
                    w_start = j * self.stride
                    for c in range(self.in_channels):
                        for m in range(kernel_height):
                            for n in range(kernel_width):
                                dX_pad[c][h_start + m][w_start + n] += (
                                    grad * self.weight[o][c][m][n]
                                )

        dX = unpad(dX_pad, self.padding)
        return dX, dW, db


class MaxPool2D:
    """
    2D max pooling.

    Ties are resolved deterministically:
    the first encountered maximum receives the gradient.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        for value, name in ((kernel_size, "kernel_size"), (stride, "stride")):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.kernel_size = kernel_size
        self.stride = stride

        self._cache_indices: list[tuple[int, int, int, int, int]] | None = None
        self._cache_input_shape: tuple[int, int, int] | None = None

    def forward(self, X: Tensor3D) -> Tensor3D:
        channels, height, width = _validate_tensor3d(X, "X")
        kernel = self.kernel_size
        stride = self.stride

        height_out = (height - kernel) // stride + 1
        width_out = (width - kernel) // stride + 1
        if height_out <= 0 or width_out <= 0:
            raise ValueError("Kernel larger than input")

        Y = [
            [[0.0 for _ in range(width_out)] for _ in range(height_out)]
            for _ in range(channels)
        ]
        indices = []

        for c in range(channels):
            for i in range(height_out):
                for j in range(width_out):
                    best_value = float("-inf")
                    best_pos = (0, 0)
                    for m in range(kernel):
                        for n in range(kernel):
                            val = X[c][i * stride + m][j * stride + n]
                            if val > best_value:  # strict > gives first‑encounter tie
                                best_value = val
                                best_pos = (m, n)
                    Y[c][i][j] = best_value
                    indices.append(
                        (
                            c,
                            i,
                            j,
                            i * stride + best_pos[0],
                            j * stride + best_pos[1],
                        )
                    )

        self._cache_indices = indices
        self._cache_input_shape = (channels, height, width)
        return Y

    def backward(self, dY: Tensor3D) -> Tensor3D:
        if self._cache_indices is None:
            raise RuntimeError("Call forward() before backward().")

        channels, height, width = self._cache_input_shape
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        if _validate_tensor3d(dY, "dY") != (channels, height_out, width_out):
            raise ValueError("dY shape does not match pooling output")

        dX = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(channels)
        ]
        for c, i, j, h_pos, w_pos in self._cache_indices:
            dX[c][h_pos][w_pos] += dY[c][i][j]
        return dX


class AvgPool2D:
    """2D average-pooling layer."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        for value, name in ((kernel_size, "kernel_size"), (stride, "stride")):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.kernel_size = kernel_size
        self.stride = stride
        self._cache_input_shape: tuple[int, int, int] | None = None

    def forward(self, X: Tensor3D) -> Tensor3D:
        channels, height, width = _validate_tensor3d(X, "X")
        kernel = self.kernel_size
        stride = self.stride

        height_out = (height - kernel) // stride + 1
        width_out = (width - kernel) // stride + 1
        if height_out <= 0 or width_out <= 0:
            raise ValueError("Kernel larger than input")

        Y = [
            [[0.0 for _ in range(width_out)] for _ in range(height_out)]
            for _ in range(channels)
        ]
        for c in range(channels):
            for i in range(height_out):
                for j in range(width_out):
                    total = 0.0
                    for m in range(kernel):
                        for n in range(kernel):
                            total += X[c][i * stride + m][j * stride + n]
                    Y[c][i][j] = total / (kernel * kernel)

        self._cache_input_shape = (channels, height, width)
        return Y

    def backward(self, dY: Tensor3D) -> Tensor3D:
        if self._cache_input_shape is None:
            raise RuntimeError("Call forward() before backward().")

        channels, height, width = self._cache_input_shape
        kernel = self.kernel_size
        stride = self.stride
        height_out = (height - kernel) // stride + 1
        width_out = (width - kernel) // stride + 1

        if _validate_tensor3d(dY, "dY") != (channels, height_out, width_out):
            raise ValueError("dY shape does not match pooling output")

        dX = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(channels)
        ]
        scale = 1.0 / (kernel * kernel)
        for c in range(channels):
            for i in range(height_out):
                for j in range(width_out):
                    grad = dY[c][i][j] * scale
                    for m in range(kernel):
                        for n in range(kernel):
                            dX[c][i * stride + m][j * stride + n] += grad
        return dX
