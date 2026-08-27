"""
Day 36: Complete CNN from First Principles

This module builds a modular Convolutional Neural Network using only pure Python lists.
It implements:

    Conv2D → MaxPool2D → Flatten → Dense

All layers support:
    - Forward pass returning the output and a 'backward' closure.
    - Backward pass that computes gradients w.r.t. inputs and accumulates
      parameter gradients (weight & bias) using the chain rule.

The design is educational: loops are explicit to show exactly how convolution,
pooling, flattening, and dense operations work under the hood.
"""

from collections.abc import Callable

# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------

def _validate_spatial_input(X: list[list[list[float]]], name: str = "X") -> None:
    """
    Ensure X is a valid 3D tensor represented as a nested list.

    Checks:
        - Non‑empty.
        - Outer dimension = number of channels.
        - All channels have the same height.
        - All rows have the same width.
        - All values are numeric (int or float).

    Raises:
        ValueError or TypeError with a clear message.
    """
    if not isinstance(X, list) or not X:
        raise ValueError(f"{name} must be a non‑empty 3D list")
    if not all(isinstance(ch, list) for ch in X):
        raise TypeError(f"{name} must be a list of 2D lists (channels)")

    # Check height consistency
    H = len(X[0])
    if H == 0:
        raise ValueError(f"{name} height cannot be zero")
    if not all(len(ch) == H for ch in X):
        raise ValueError(f"{name} has inconsistent heights")

    # Check width consistency and numeric types
    W = len(X[0][0])
    if W == 0:
        raise ValueError(f"{name} width cannot be zero")
    for ch in X:
        for row in ch:
            if not isinstance(row, list) or len(row) != W:
                raise ValueError(f"{name} has inconsistent widths")
            if not all(isinstance(v, (int, float)) for v in row):
                raise TypeError(f"{name} contains non‑numeric values")


def _validate_vector(X: list[float], name: str = "X") -> None:
    """Check that X is a non‑empty list of numbers."""
    if not isinstance(X, list) or not X:
        raise ValueError(f"{name} must be a non‑empty list of floats")
    if not all(isinstance(v, (int, float)) for v in X):
        raise TypeError(f"{name} contains non‑numeric values")


# ----------------------------------------------------------------------
# Convolutional Layer
# ----------------------------------------------------------------------

class Conv2DLayer:
    """
    2D Convolution with autograd support.

    Attributes:
        in_channels, out_channels, kernel_size, stride, padding (int)
        weight: 4D list [out_channels][in_channels][kernel_size][kernel_size]
        bias: 1D list [out_channels]
        weight_grad, bias_grad: accumulated gradients from backward passes.

    Initialisation uses a modified He scaling (factor 0.1) to keep early
    activations moderate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        """Store hyper‑parameters and initialise weights/biases with gradients."""
        # Input validation
        if not all(isinstance(v, int) and v > 0 for v in (in_channels, out_channels, kernel_size, stride)):
            raise ValueError("in_channels, out_channels, kernel_size, stride must be positive integers")
        if padding < 0:
            raise ValueError("padding must be non‑negative")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Modified He initialisation (scaled down by 0.1)
        scale = (2.0 / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.weight = [
            [
                [[0.1 * scale for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias = [0.0 for _ in range(out_channels)]

        # Gradient accumulators (same shapes)
        self.weight_grad = [
            [
                [[0.0 for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias_grad = [0.0 for _ in range(out_channels)]

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero (for next mini‑batch)."""
        for o in range(self.out_channels):
            self.bias_grad[o] = 0.0
            for c in range(self.in_channels):
                for m in range(self.kernel_size):
                    for n in range(self.kernel_size):
                        self.weight_grad[o][c][m][n] = 0.0

    def forward(self, X: list[list[list[float]]]) -> tuple[list[list[list[float]]], Callable]:
        """
        Perform cross‑correlation (convolution) with padding and stride.

        Args:
            X: Input tensor shape (C_in, H_in, W_in)

        Returns:
            output: shape (C_out, H_out, W_out)
            backward_fn: a closure that computes gradients w.r.t. X, W, and b
                         when called with dY (gradient w.r.t. output).
        """
        _validate_spatial_input(X, "Conv2DLayer.forward")
        C_in, H_in, W_in = len(X), len(X[0]), len(X[0][0])
        if C_in != self.in_channels:
            raise ValueError(f"Input channels {C_in} != layer in_channels {self.in_channels}")

        K = self.kernel_size
        P = self.padding
        S = self.stride

        # ---- Compute output dimensions ----
        H_out = (H_in + 2 * P - K) // S + 1
        W_out = (W_in + 2 * P - K) // S + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Invalid output dimensions {H_out}x{W_out}; check kernel/padding/stride")

        # ---- Pad the input ----
        H_pad = H_in + 2 * P
        W_pad = W_in + 2 * P
        X_padded = [
            [[0.0 for _ in range(W_pad)] for _ in range(H_pad)]
            for _ in range(C_in)
        ]
        for c in range(C_in):
            for i in range(H_in):
                for j in range(W_in):
                    X_padded[c][i + P][j + P] = X[c][i][j]

        # ---- Allocate output tensor ----
        out = [
            [[0.0 for _ in range(W_out)] for _ in range(H_out)]
            for _ in range(self.out_channels)
        ]

        # ---- Convolution loops (explicit for educational clarity) ----
        for o in range(self.out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    # Start with bias
                    val = self.bias[o]
                    # Sum over input channels and kernel window
                    for c in range(C_in):
                        for m in range(K):
                            for n in range(K):
                                val += (
                                    X_padded[c][i * S + m][j * S + n]
                                    * self.weight[o][c][m][n]
                                )
                    out[o][i][j] = val

        # ---- Backward closure (captures all forward‑pass data) ----
        def backward(dY: list[list[list[float]]]) -> list[list[list[float]]]:
            """
            Compute gradients and accumulate them.

            Args:
                dY: upstream gradient, shape (C_out, H_out, W_out)

            Returns:
                dX: gradient w.r.t. input X, shape (C_in, H_in, W_in)
            """
            _validate_spatial_input(dY, "Conv2DLayer.backward dY")
            if len(dY) != self.out_channels or len(dY[0]) != H_out or len(dY[0][0]) != W_out:
                raise ValueError(f"dY shape mismatch: expected ({self.out_channels},{H_out},{W_out})")

            # Initialise gradient w.r.t. padded input to zeros
            dX_padded = [
                [[0.0 for _ in range(W_pad)] for _ in range(H_pad)]
                for _ in range(C_in)
            ]

            # Accumulate gradients
            for o in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        grad = dY[o][i][j]  # scalar

                        # Bias gradient: sum of dY over all spatial positions
                        self.bias_grad[o] += grad

                        # Weight and input gradients
                        for c in range(C_in):
                            for m in range(K):
                                for n in range(K):
                                    h_idx = i * S + m
                                    w_idx = j * S + n
                                    # dW = dY * X_padded (at that position)
                                    self.weight_grad[o][c][m][n] += grad * X_padded[c][h_idx][w_idx]
                                    # dX_padded = dY * weight (transposed convolution)
                                    dX_padded[c][h_idx][w_idx] += grad * self.weight[o][c][m][n]

            # Remove padding to get dX
            dX = [
                [[0.0 for _ in range(W_in)] for _ in range(H_in)]
                for _ in range(C_in)
            ]
            for c in range(C_in):
                for i in range(H_in):
                    for j in range(W_in):
                        dX[c][i][j] = dX_padded[c][i + P][j + P]
            return dX

        return out, backward


# ----------------------------------------------------------------------
# Max Pooling Layer
# ----------------------------------------------------------------------

class MaxPool2DLayer:
    """
    Max‑pooling with argmax caching for gradient routing.

    During forward, it stores the location of the maximum in each window.
    During backward, each output gradient is routed only to that location.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        if not all(isinstance(v, int) and v > 0 for v in (kernel_size, stride)):
            raise ValueError("kernel_size and stride must be positive integers")
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X: list[list[list[float]]]) -> tuple[list[list[list[float]]], Callable]:
        """
        Forward pass: down‑sample using max.

        Args:
            X: (C, H_in, W_in)

        Returns:
            output: (C, H_out, W_out)
            backward_fn: closure that routes gradients to argmax positions.
        """
        _validate_spatial_input(X, "MaxPool2DLayer.forward")
        C, H_in, W_in = len(X), len(X[0]), len(X[0][0])
        K = self.kernel_size
        S = self.stride

        # ---- Output dimensions ----
        H_out = (H_in - K) // S + 1
        W_out = (W_in - K) // S + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Pooling output dimensions {H_out}x{W_out} invalid; kernel/stride too large")

        # ---- Allocate output and argmax map ----
        output = [
            [[0.0 for _ in range(W_out)] for _ in range(H_out)]
            for _ in range(C)
        ]
        # Map from (channel, out_i, out_j) → (argmax_h, argmax_w)
        argmax_map: dict[tuple[int, int, int], tuple[int, int]] = {}

        # ---- Pool forward ----
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * S
                    w_start = j * S
                    max_val = float("-inf")
                    max_pos = (h_start, w_start)  # fallback
                    # Scan the window; tie → first encountered (row‑major)
                    for m in range(K):
                        for n in range(K):
                            val = X[c][h_start + m][w_start + n]
                            if val > max_val:
                                max_val = val
                                max_pos = (h_start + m, w_start + n)
                    output[c][i][j] = max_val
                    argmax_map[(c, i, j)] = max_pos

        # ---- Backward closure ----
        def backward(dY: list[list[list[float]]]) -> list[list[list[float]]]:
            """
            Route each gradient to the argmax location from forward.

            If windows overlap, gradients accumulate (via +=).
            """
            _validate_spatial_input(dY, "MaxPool2DLayer.backward dY")
            if len(dY) != C or len(dY[0]) != H_out or len(dY[0][0]) != W_out:
                raise ValueError(f"dY shape mismatch: expected ({C},{H_out},{W_out})")

            dX = [
                [[0.0 for _ in range(W_in)] for _ in range(H_in)]
                for _ in range(C)
            ]
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        orig_h, orig_w = argmax_map[(c, i, j)]
                        dX[c][orig_h][orig_w] += dY[c][i][j]
            return dX

        return output, backward


# ----------------------------------------------------------------------
# Flatten Layer
# ----------------------------------------------------------------------

class FlattenLayer:
    """Convert a spatial tensor (C, H, W) to a 1D vector in channel‑major order."""

    def forward(self, X: list[list[list[float]]]) -> tuple[list[float], Callable]:
        """
        Flatten.

        Args:
            X: (C, H, W)

        Returns:
            flat_out: list of length C*H*W
            backward_fn: reshapes a 1D gradient back to (C, H, W)
        """
        _validate_spatial_input(X, "FlattenLayer.forward")
        C, H, W = len(X), len(X[0]), len(X[0][0])

        # Flatten in channel‑major order: channel0, row0, col0; row0,col1; ...
        flat_out = []
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    flat_out.append(X[c][i][j])

        def backward(dY_flat: list[float]) -> list[list[list[float]]]:
            """Restore gradient to original spatial shape."""
            _validate_vector(dY_flat, "FlattenLayer.backward dY_flat")
            expected_len = C * H * W
            if len(dY_flat) != expected_len:
                raise ValueError(f"dY_flat length {len(dY_flat)} != {expected_len}")
            dX = [
                [[0.0 for _ in range(W)] for _ in range(H)]
                for _ in range(C)
            ]
            idx = 0
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        dX[c][i][j] = dY_flat[idx]
                        idx += 1
            return dX

        return flat_out, backward


# ----------------------------------------------------------------------
# Fully‑Connected (Dense) Layer
# ----------------------------------------------------------------------

class DenseLayer:
    """Linear (affine) layer: y = W * x + b, with autograd support."""

    def __init__(self, in_features: int, out_features: int):
        if not all(isinstance(v, int) and v > 0 for v in (in_features, out_features)):
            raise ValueError("in_features and out_features must be positive integers")
        self.in_features = in_features
        self.out_features = out_features

        # Scaled He initialisation (factor 0.1)
        scale = (2.0 / in_features) ** 0.5
        self.weight = [
            [0.1 * scale for _ in range(in_features)] for _ in range(out_features)
        ]
        self.bias = [0.0 for _ in range(out_features)]

        self.weight_grad = [[0.0 for _ in range(in_features)] for _ in range(out_features)]
        self.bias_grad = [0.0 for _ in range(out_features)]

    def zero_grad(self) -> None:
        for o in range(self.out_features):
            self.bias_grad[o] = 0.0
            for i in range(self.in_features):
                self.weight_grad[o][i] = 0.0

    def forward(self, X: list[float]) -> tuple[list[float], Callable]:
        """
        Compute y = W * x + b.

        Args:
            X: input vector length in_features

        Returns:
            out: output vector length out_features
            backward_fn: computes dX and accumulates dW, db
        """
        _validate_vector(X, "DenseLayer.forward")
        if len(X) != self.in_features:
            raise ValueError(f"Input length {len(X)} != in_features {self.in_features}")

        # Affine transformation
        out = [
            self.bias[o] + sum(X[i] * self.weight[o][i] for i in range(self.in_features))
            for o in range(self.out_features)
        ]

        def backward(dY: list[float]) -> list[float]:
            """
            Backward: accumulate parameter gradients and compute dX.
            """
            _validate_vector(dY, "DenseLayer.backward dY")
            if len(dY) != self.out_features:
                raise ValueError(f"dY length {len(dY)} != out_features {self.out_features}")

            dX = [0.0 for _ in range(self.in_features)]
            for o in range(self.out_features):
                grad = dY[o]

                # Bias gradient = dY (sum over batch, but batch size = 1 here)
                self.bias_grad[o] += grad

                # Weight gradient = dY * X (outer product)
                for i in range(self.in_features):
                    self.weight_grad[o][i] += grad * X[i]
                    # Input gradient = sum over outputs of dY * weight
                    dX[i] += grad * self.weight[o][i]
            return dX

        return out, backward


# ----------------------------------------------------------------------
# CNN Container (sequential model)
# ----------------------------------------------------------------------

class ConvNet:
    """
    A simple two‑stage CNN for demonstration:

        Conv2D(1→2, K=3, S=1, P=1)
        → MaxPool2D(K=2, S=2)
        → Flatten
        → Dense(8→2)

    Input shape is fixed to (1, 4, 4) for this architecture.
    """

    def __init__(self):
        self.conv1 = Conv2DLayer(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2DLayer(kernel_size=2, stride=2)
        self.flatten = FlattenLayer()
        # After conv+pool, shape = (2,2,2) → flattened size = 8
        self.fc1 = DenseLayer(in_features=8, out_features=2)

    def zero_grad(self) -> None:
        """Zero out gradients of all trainable layers."""
        self.conv1.zero_grad()
        self.fc1.zero_grad()

    def forward(self, X: list[list[list[float]]]) -> tuple[list[float], list[Callable]]:
        """
        Run the forward pass sequentially, collecting backward closures.

        Args:
            X: input image, must be (1, 4, 4)

        Returns:
            logits: length‑2 output vector
            closures: list of backward functions in forward order.
        """
        _validate_spatial_input(X, "ConvNet.forward")

        # Enforce fixed input shape
        actual_shape = (len(X), len(X[0]), len(X[0][0]))
        expected_shape = (1, 4, 4)
        if actual_shape != expected_shape:
            raise ValueError(
                f"ConvNet expects input shape {expected_shape}, "
                f"got {actual_shape}"
            )

        closures = []

        # ---- Conv2D ----
        out, b = self.conv1.forward(X)
        closures.append(b)

        # ---- MaxPool ----
        out, b = self.pool1.forward(out)
        closures.append(b)

        # ---- Flatten ----
        out, b = self.flatten.forward(out)
        closures.append(b)

        # ---- Dense ----
        logits, b = self.fc1.forward(out)
        closures.append(b)

        return logits, closures

    def backward(self, dL_dlogits: list[float], closures: list[Callable]) -> list[list[list[float]]]:
        """
        Execute backpropagation by applying closures in reverse order.

        Args:
            dL_dlogits: gradient of loss w.r.t. logits, shape (2,)
            closures: list of backward closures in forward order

        Returns:
            gradient w.r.t. the original input X, shape (1,4,4)
        """
        _validate_vector(dL_dlogits, "ConvNet.backward dL_dlogits")
        if len(dL_dlogits) != 2:
            raise ValueError("Expected 2 logits")

        grad = dL_dlogits
        # Reverse chain rule: apply closures in reverse order
        for b in reversed(closures):
            grad = b(grad)

        # After all closures, grad should be a spatial gradient
        _validate_spatial_input(grad, "ConvNet.backward output")
        return grad