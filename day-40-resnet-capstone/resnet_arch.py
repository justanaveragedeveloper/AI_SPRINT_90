"""
Day 40: ResNet Architecture with First‑Principles Residual Connections

This module implements a small ResNet‑style classifier from scratch using NumPy.
It is designed for **educational purposes** – every mathematical operation is
explicit, so you can see exactly how forward propagation, backpropagation,
and the residual gradient highway work.

Key concepts demonstrated:
  - Identity skip connections (residual learning).
  - Element‑wise addition and its backward gradient splitting.
  - Manual autograd via closures (each forward pass returns a backward function).
  - Gradient accumulation for mini‑batch training (simulated).

All layers operate on single examples (no batch dimension) to keep the code clear.
"""

import logging
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# Logging setup (better than print() for production awareness)
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set a fixed seed for reproducible weight initialisation and tests.
np.random.seed(42)


# ----------------------------------------------------------------------
# Helper: ensure finite numbers (no NaN/Inf)
# ----------------------------------------------------------------------
def _assert_finite(X: NDArray, name: str) -> None:
    """
    Raise a ValueError if X contains any NaN or Inf.
    This is a defensive programming measure.
    """
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or Inf values")


# ======================================================================
# 1.  CORE LAYERS (Conv2D, Flatten, Dense)
# ======================================================================

class Conv2DLayer:
    """
    2D Convolutional layer – single example, no batch.

    Educational limitation: supports only stride=1.
    Padding is applied as 'same' (padding = kernel_size // 2) when requested.

    Forward pass:
        - Pads input if needed.
        - Extracts sliding windows (patches) using NumPy's as_strided.
        - Computes output via tensordot (vectorised per patch).

    Backward pass:
        - Computes gradients w.r.t. weights (dW) and bias (db).
        - Computes gradient w.r.t. input (dX) using convolution of dY with kernel.
        - Accumulates gradients in self.dW / self.db.

    The backward closure captures all forward‑specific values so that
    multiple forwards can happen before backwards (no stale cache).

    This is the most complex layer; we keep loops explicit for clarity.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        """Initialise weights (He), biases, and gradient accumulators."""
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError("Channels and kernel_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride != 1:
            raise ValueError("This educational Conv2DLayer supports stride=1 only.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialisation (good for ReLU)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        # Gradient accumulators (zeroed after each batch)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero (call before a new batch)."""
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, X: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
        """
        Forward pass.

        Args:
            X: input tensor of shape (C_in, H, W)

        Returns:
            out: output tensor of shape (out_channels, H_out, W_out)
            backward: a closure that computes gradients when called with dY.
        """
        _assert_finite(X, "Conv2D input")

        C_in, H, W = X.shape  # noqa: RUF059
        if C_in != self.in_channels:
            raise ValueError(f"Input channels {C_in} != expected {self.in_channels}")

        # 1. Pad input if needed
        pad = self.padding
        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            X_pad = X
        H_pad, W_pad = X_pad.shape[1], X_pad.shape[2]

        k = self.kernel_size
        H_out = (H_pad - k) // self.stride + 1
        W_out = (W_pad - k) // self.stride + 1

        # 2. Extract sliding windows (patches) using as_strided.
        #    This gives a view of shape (C_in, H_out, W_out, k, k).
        from numpy.lib.stride_tricks import as_strided
        shape = (C_in, H_out, W_out, k, k)
        strides = (X_pad.strides[0],
                   X_pad.strides[1] * self.stride,
                   X_pad.strides[2] * self.stride,
                   X_pad.strides[1],
                   X_pad.strides[2])
        patches = as_strided(X_pad, shape=shape, strides=strides)
        # Reshape to (C_in, N, k^2) where N = H_out * W_out
        patches_flat = patches.reshape(C_in, H_out * W_out, -1)

        # 3. Reshape weights to (out_channels, C_in, k^2)
        W_flat = self.W.reshape(self.out_channels, C_in, -1)

        # 4. Compute output: out[o, n] = sum_c W_flat[o,c,:] * patches_flat[c,n,:]
        #    This is a tensor contraction (einsum) – vectorised over patches.
        out_flat = np.einsum('ocp,cnp->on', W_flat, patches_flat)  # (out, N)
        out = out_flat.reshape(self.out_channels, H_out, W_out) + self.b[:, None, None]

        # ---- Capture forward‑specific values for the backward pass ----
        # This prevents stale‑cache bugs if multiple forwards are called.
        X_pad_captured = X_pad
        patches_flat_captured = patches_flat
        k_captured = k
        H_pad_captured, W_pad_captured = H_pad, W_pad
        C_in_captured = C_in
        W_captured = self.W

        # ---- Backward closure ----
        def backward(dY: NDArray) -> NDArray:
            """
            Backward pass for this specific forward call.

            Args:
                dY: upstream gradient of shape (out_channels, H_out, W_out)

            Returns:
                dX: gradient w.r.t. input X, shape (C_in, H, W)
            """
            _assert_finite(dY, "Conv2D dY")

            # a) Gradient w.r.t. bias: sum over spatial dimensions
            self.db += dY.sum(axis=(1, 2))

            # b) Gradient w.r.t. weights: correlation of input patches with dY
            dY_flat = dY.reshape(self.out_channels, -1)  # (out, N)
            dW_flat = np.einsum('on, cnp -> ocp', dY_flat, patches_flat_captured)
            self.dW += dW_flat.reshape(self.out_channels, C_in_captured, k_captured, k_captured)

            # c) Gradient w.r.t. input (dX_padded): convolution of dY with the *same* kernel.
            #    Because forward is cross‑correlation, backward is convolution with the kernel
            #    (no rotation) – this is the correct gradient for a cross‑correlation layer.
            dX_pad = np.zeros_like(X_pad_captured)
            H_dY, W_dY = dY.shape[1], dY.shape[2]
            for c in range(C_in_captured):
                acc = np.zeros((H_pad_captured, W_pad_captured))
                for o in range(self.out_channels):
                    kernel = W_captured[o, c]  # (k, k)
                    # For each output position (i,j) in dX_pad, sum over kernel positions
                    # that overlap with dY. This is the convolution operation.
                    for i in range(H_pad_captured):
                        for j in range(W_pad_captured):
                            val = 0.0
                            for di in range(k_captured):
                                for dj in range(k_captured):
                                    i_dy = i - di
                                    j_dy = j - dj
                                    if 0 <= i_dy < H_dY and 0 <= j_dy < W_dY:
                                        val += dY[o, i_dy, j_dy] * kernel[di, dj]
                            acc[i, j] += val
                dX_pad[c] = acc

            # Remove padding to get dX of original shape
            if self.padding > 0:
                dX = dX_pad[:, self.padding:-self.padding, self.padding:-self.padding]
            else:
                dX = dX_pad
            return dX

        return out, backward


class FlattenLayer:
    """Flatten a 3D tensor (C, H, W) to 1D (C*H*W)."""

    def forward(self, X: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
        _assert_finite(X, "Flatten input")
        input_shape = X.shape
        out = X.flatten()

        def backward(dY: NDArray) -> NDArray:
            # Reshape gradient back to original shape
            return dY.reshape(input_shape)

        return out, backward

    def zero_grad(self) -> None:
        """No parameters to zero."""
        pass  # noqa: PIE790


class DenseLayer:
    """Fully connected (linear) layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Features must be positive.")
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)  # He initialisation
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def forward(self, X: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
        _assert_finite(X, "Dense input")
        if X.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {X.shape}")
        if X.shape[0] != self.in_features:
            raise ValueError(f"Input size {X.shape[0]} != {self.in_features}")

        # Capture X for backward
        X_captured = X
        out = X @ self.W + self.b

        def backward(dY: NDArray) -> NDArray:
            self.dW += np.outer(X_captured, dY)  # outer product = X * dY^T
            self.db += dY
            return dY @ self.W.T  # gradient w.r.t. X

        return out, backward


# ======================================================================
# 2.  RESNET SPECIFIC COMPONENTS
# ======================================================================

def relu(X: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
    """
    ReLU activation function (element‑wise).

    Forward: ReLU(x) = max(0, x)
    Backward: gradient is 1 for x > 0, else 0.
    At x == 0, we define derivative = 0 (a common convention).

    This is a function, not a class, because it has no parameters.
    """
    _assert_finite(X, "ReLU input")
    out = np.maximum(X, 0.0)
    # Mask: 1 where X > 0, 0 otherwise (including exactly 0)
    mask = (X > 0).astype(X.dtype)

    def backward(dY: NDArray) -> NDArray:
        _assert_finite(dY, "ReLU dY")
        return dY * mask

    return out, backward


class ElementwiseAdd:
    """
    Element‑wise addition: Z = A + B.

    This is the key layer for residual connections.
    In the residual block, A = F(X) (residual branch) and B = X (identity branch).

    Backward: gradients split equally:
        dA = dZ   (gradient w.r.t. residual branch)
        dB = dZ   (gradient w.r.t. identity branch)

    This is the **gradient highway** – the identity branch always gets the full
    upstream gradient, so even if the residual branch gradient vanishes, training
    can continue.
    """

    def forward(self, A: NDArray, B: NDArray) -> tuple[NDArray, Callable[[NDArray], tuple[NDArray, NDArray]]]:
        _assert_finite(A, "ElementwiseAdd A")
        _assert_finite(B, "ElementwiseAdd B")
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: A {A.shape} vs B {B.shape}")
        out = A + B

        def backward(dZ: NDArray) -> tuple[NDArray, NDArray]:
            _assert_finite(dZ, "ElementwiseAdd dZ")
            # Both branches receive the same gradient.
            return dZ, dZ

        return out, backward


class ResidualBlock:
    """
    A single ResNet block:
        X → Conv1 → ReLU → Conv2 → Add(F(X), X) → ReLU → Y

    This is the core of the ResNet architecture.
    The skip connection (identity) allows gradients to flow directly.

    Mathematical invariant:
        Y = ReLU( F(X) + X )
        where F(X) = Conv2( ReLU( Conv1(X) ) )

    Backward:
        dY → ReLU → Add → (identity path) and (residual path)
        At the input, gradients from both paths are summed.

    This implementation uses the same number of channels for both paths
    (no projection shortcut) – shapes must match.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        # Both convolutions preserve spatial size (padding = kernel_size//2, stride=1)
        self.conv1 = Conv2DLayer(channels, channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = Conv2DLayer(channels, channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.add = ElementwiseAdd()

    def zero_grad(self) -> None:
        """Reset gradients of both convolution layers."""
        self.conv1.zero_grad()
        self.conv2.zero_grad()

    def forward(self, X: NDArray) -> tuple[NDArray, Callable[[NDArray], NDArray]]:
        # Forward path
        out1, b1 = self.conv1.forward(X)
        act1, b_act1 = relu(out1)

        out2, b2 = self.conv2.forward(act1)
        res_out, b_add = self.add.forward(out2, X)   # res_out = F(X) + X
        act2, b_act2 = relu(res_out)

        # ---- Backward closure for this block ----
        def backward(dY: NDArray) -> NDArray:
            # dY is gradient of loss w.r.t. act2 (output of block)
            d_res = b_act2(dY)                # gradient through final ReLU
            d_out2, d_identity = b_add(d_res) # split at addition

            # Residual path
            d_act1 = b2(d_out2)               # through Conv2
            d_out1 = b_act1(d_act1)           # through first ReLU
            d_conv_in = b1(d_out1)            # through Conv1

            # Sum of gradients: residual path + identity path
            # This is the **gradient highway**.
            return d_conv_in + d_identity

        return act2, backward


class ResNetClassifier:
    """
    Full ResNet classifier: small network for demonstration.

    Architecture:
        Input (C, H, W)
          ↓
        Stem Conv (3x3, C→hidden_channels, padding=1) → preserves spatial size
          ↓
        Residual Block (hidden_channels → hidden_channels, 3x3)
          ↓
        Flatten (spatial to 1D)
          ↓
        Dense (hidden_channels * H * W → num_classes)
          ↓
        Logits

    All shapes are fixed at construction time.
    The residual block keeps spatial dimensions unchanged, so flatten size
    is known in advance.

    This demonstrates the entire pipeline from input to logits with manual autograd.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 2, num_classes: int = 2,
                 input_height: int = 4, input_width: int = 4) -> None:
        """
        Args:
            in_channels: number of input channels (e.g., 1 for grayscale)
            hidden_channels: number of channels after stem and throughout residual block
            num_classes: number of output classes
            input_height, input_width: expected spatial dimensions of input
        """
        if in_channels <= 0 or hidden_channels <= 0 or num_classes <= 0:
            raise ValueError("All channel/class numbers must be positive.")
        if input_height <= 0 or input_width <= 0:
            raise ValueError("input_height and input_width must be positive.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width

        # Stem convolution: preserves spatial size (padding=1, stride=1)
        self.stem_conv = Conv2DLayer(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

        # Residual block
        self.res_block = ResidualBlock(hidden_channels, kernel_size=3)

        # Flatten
        self.flatten = FlattenLayer()

        # Dense layer: pre‑compute flatten size
        flat_size = hidden_channels * input_height * input_width
        self.fc = DenseLayer(in_features=flat_size, out_features=num_classes)

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero (call before a new training batch)."""
        self.stem_conv.zero_grad()
        self.res_block.zero_grad()
        self.fc.zero_grad()

    def forward(self, X: NDArray) -> tuple[NDArray, list[Callable[[NDArray], NDArray]]]:
        """
        Forward pass.

        Args:
            X: input tensor (C, H, W) – must match expected dimensions

        Returns:
            logits: (num_classes,) array
            closures: list of backward closures in forward order.
                      To backpropagate, call them in reverse order.
        """
        _assert_finite(X, "ResNet input")

        # Validate shape
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (C,H,W), got shape {X.shape}")
        C, H, W = X.shape
        if C != self.in_channels:
            raise ValueError(f"Input channels {C} != expected {self.in_channels}")
        if H != self.input_height or W != self.input_width:
            raise ValueError(
                f"Input spatial size {H}x{W} != expected {self.input_height}x{self.input_width}"
            )

        closures: list[Callable[[NDArray], NDArray]] = []

        # Stem
        out, b = self.stem_conv.forward(X)
        closures.append(b)

        # Residual block
        out, b = self.res_block.forward(out)
        closures.append(b)

        # Flatten
        out, b = self.flatten.forward(out)
        closures.append(b)

        # Dense (FC)
        logits, b = self.fc.forward(out)
        closures.append(b)

        return logits, closures

    def backward(self, dL_dlogits: NDArray, closures: list[Callable[[NDArray], NDArray]]) -> NDArray:
        """
        Backward pass.

        Args:
            dL_dlogits: gradient of loss w.r.t. logits (shape (num_classes,))
            closures: list of backward closures (from forward, in forward order)

        Returns:
            dX: gradient w.r.t. input X (shape (C, H, W))
        """
        _assert_finite(dL_dlogits, "ResNet dL_dlogits")
        grad = dL_dlogits
        # Apply closures in reverse order to propagate from output to input
        for b in reversed(closures):
            grad = b(grad)
        return grad