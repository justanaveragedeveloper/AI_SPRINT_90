"""
conv_autograd.py
Day 35: Autograd integration for 2D convolution and max pooling.
Provides Tensor, Conv2D, and MaxPool2D with explicit backward closures.

This file builds on Days 22–24 (autograd engine) and Day 34 (spatial operations).
It turns convolution and pooling into differentiable nodes in a computation graph.

Key idea:
  - Forward: compute output and store tensors needed for backward.
  - Backward: receive dL/dY, compute dL/dX, dL/dW, dL/db (for conv) or route gradients to argmax positions (for pooling).
  - The Tensor class handles graph traversal and gradient accumulation.
"""

# ----------------------------------------------------------------------
# Tensor: the core autograd building block
# ----------------------------------------------------------------------

class Tensor:
    """
    A simple Tensor that holds nested list data and supports autograd.

    Attributes:
        data (list | float): the numerical value(s).
        shape (tuple): e.g., (C, H, W) for 3D.
        grad (list | float | None): gradient with respect to this tensor (same shape).
        requires_grad (bool): whether to compute gradients.
        _backward (callable): the local gradient function (closure).
        _prev (set): parent tensors in the computation graph.
    """

    def __init__(self,
                 data: list | float,
                 requires_grad: bool = False,
                 _children: tuple['Tensor', ...] = ()):
        """
        Create a Tensor.

        Args:
            data: nested list of numbers, or a single float.
            requires_grad: if True, gradients will be accumulated.
            _children: parent tensors that this tensor depends on (for graph).
        """
        if isinstance(data, (int, float)):
            self.data = data
            self.shape = ()
        else:
            self.data = data
            self.shape = self._infer_shape(data)
        self.grad = None               # will be filled with same shape as data
        self.requires_grad = requires_grad
        self._backward = lambda: None  # placeholder; gets replaced by operation's closure
        self._prev = set(_children)    # parents in the DAG

    @staticmethod
    def _infer_shape(data):
        """Recursively determine shape of nested list."""
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        first = data[0]
        if not isinstance(first, list):
            return (len(data),)
        return (len(data),) + Tensor._infer_shape(first)

    def zero_grad(self):
        """Reset gradient to None (or zeros) for this tensor."""
        self.grad = None

    def backward(self):
        """
        Perform topological backward pass from this tensor (assumes scalar loss).

        This is the public entry point for autograd. It builds a topological
        ordering of the graph (from this node back to the leaves), then calls
        each node's _backward closure in reverse order.
        """
        # 1. Build topological order (children -> parents)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. Seed gradient: for a scalar loss, dLoss/dLoss = 1.0
        if self.grad is None:
            self.grad = 1.0

        # 3. Propagate gradients
        for v in reversed(topo):
            v._backward()   # each node's local gradient function

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.grad is not None})"


# ----------------------------------------------------------------------
# Utility functions for nested list manipulation
# ----------------------------------------------------------------------

def shape_list(x):
    """Return the shape of a nested list (e.g., (C, H, W))."""
    if not isinstance(x, list):
        return ()
    if not x:
        return (0,)
    return (len(x),) + shape_list(x[0])

def zeros_like(x):
    """Create a nested list of zeros with the same structure as x."""
    if not isinstance(x, list):
        return 0.0
    return [zeros_like(row) for row in x]

# The following helpers are not used directly in the core operations,
# but can be useful for tests or extensions.
def add_inplace(a, b):
    """a += b elementwise (assumes same nested structure)."""
    if isinstance(a, list):
        for i, sub in enumerate(a):
            add_inplace(sub, b[i])
    else:
        a += b

def tensor_from_list(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)

def list_from_tensor(t):
    return t.data


# ----------------------------------------------------------------------
# Conv2D autograd operation
# ----------------------------------------------------------------------

def conv2d(input: Tensor,
           weight: Tensor,
           bias: Tensor | None,
           stride: int = 1,
           padding: int = 0) -> Tensor:
    """
    2D convolution (cross‑correlation) with autograd.

    Mathematical forward:
        Y[o,i,j] = b[o] + Σ_c Σ_m Σ_n X_pad[c, i*S + m, j*S + n] * W[o,c,m,n]

    Backward:
        dW[o,c,m,n] = Σ_i Σ_j dY[o,i,j] * X_pad[c, i*S + m, j*S + n]
        db[o]       = Σ_i Σ_j dY[o,i,j]
        dX_pad[...] = Σ_o Σ_i Σ_j dY[o,i,j] * W[o,c,m,n]  (then unpadded)

    Shapes:
        X:   (C_in, H_in, W_in)
        W:   (C_out, C_in, KH, KW)
        b:   (C_out,) or None
        Y:   (C_out, H_out, W_out)
    """
    # ---------- Input validation ----------
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be non‑negative, got {padding}")

    X = input.data
    W = weight.data
    b = bias.data if bias is not None else None

    # Validate tensor ranks and channel consistency
    if len(shape_list(X)) != 3:
        raise ValueError(f"Input must be 3D (C,H,W), got shape {shape_list(X)}")
    if len(shape_list(W)) != 4:
        raise ValueError(f"Weight must be 4D (Cout,Cin,KH,KW), got shape {shape_list(W)}")

    C_in, H_in, W_in = shape_list(X)
    C_out, C_in_w, KH, KW = shape_list(W)

    if C_in_w != C_in:
        raise ValueError(f"Weight input channels {C_in_w} must match input channels {C_in}")

    if bias is not None:
        if len(shape_list(b)) != 1:
            raise ValueError(f"Bias must be 1D, got shape {shape_list(b)}")
        if shape_list(b)[0] != C_out:
            raise ValueError(f"Bias length {shape_list(b)[0]} must match output channels {C_out}")

    # ---------- Compute output dimensions ----------
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Output dimension non‑positive: H_out={H_out}, W_out={W_out}. "
                         f"Check kernel size, stride, or padding.")

    # ---------- Pad input ----------
    # Create X_pad with zeros around the border
    H_pad = H_in + 2 * padding
    W_pad = W_in + 2 * padding
    X_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                X_pad[c][i + padding][j + padding] = X[c][i][j]

    # ---------- Forward pass (explicit loops) ----------
    out_data = [[[0.0] * W_out for _ in range(H_out)] for _ in range(C_out)]
    for o in range(C_out):                     # output channel
        for i in range(H_out):                 # output height
            for j in range(W_out):             # output width
                val = b[o] if b is not None else 0.0
                for c in range(C_in):          # input channel
                    for m in range(KH):        # kernel height
                        for n in range(KW):    # kernel width
                            val += (X_pad[c][i*stride + m][j*stride + n] *
                                    W[o][c][m][n])
                out_data[o][i][j] = val

    # ---------- Build output Tensor ----------
    # The output should require gradients if any input requires them.
    requires_grad = (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    # Store parents so the graph knows who created this tensor.
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=(input, weight, bias) if bias is not None else (input, weight))

    # ---------- Backward closure ----------
    # This closure will be called during the backward pass.
    def _backward():
        dY = out.grad   # upstream gradient, shape (C_out, H_out, W_out)
        if dY is None:
            return

        # Initialize gradients if they are not already set.
        if input.requires_grad and input.grad is None:
            input.grad = zeros_like(X)
        if weight.requires_grad and weight.grad is None:
            weight.grad = zeros_like(W)
        if bias is not None and bias.requires_grad and bias.grad is None:
            bias.grad = zeros_like(b)

        dX_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]

        # For each output element, propagate the gradient to weights, bias, and input.
        for o in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    grad = dY[o][i][j]

                    # Bias gradient: sum of dY over spatial positions.
                    if bias is not None and bias.requires_grad:
                        bias.grad[o] += grad

                    # Weight and input gradients
                    for c in range(C_in):
                        for m in range(KH):
                            for n in range(KW):
                                h_idx = i * stride + m
                                w_idx = j * stride + n
                                x_val = X_pad[c][h_idx][w_idx]

                                # dW: multiply dY by the corresponding input value.
                                if weight.requires_grad:
                                    weight.grad[o][c][m][n] += grad * x_val

                                # dX: multiply dY by the corresponding weight.
                                if input.requires_grad:
                                    dX_pad[c][h_idx][w_idx] += grad * W[o][c][m][n]

        # Unpad dX_pad to get dX (remove the border).
        if input.requires_grad:
            for c in range(C_in):
                for i in range(H_in):
                    for j in range(W_in):
                        input.grad[c][i][j] += dX_pad[c][i + padding][j + padding]

    out._backward = _backward
    return out


# ----------------------------------------------------------------------
# MaxPool2D autograd operation
# ----------------------------------------------------------------------

def maxpool2d(input: Tensor,
              kernel_size: int | tuple[int, int],
              stride: int | None = None,
              padding: int = 0) -> Tensor:
    """
    2D max pooling with autograd.

    Forward:
        For each output position, pick the maximum value from a KH×KW window.
        Also store the location (argmax) so backward can route gradients.

    Backward:
        dY is routed only to the argmax positions. If an input participates
        in multiple windows (overlap), gradients accumulate.

    Args:
        kernel_size: int or (KH, KW)
        stride: if None, defaults to kernel_size
        padding: zero padding around input
    """
    if padding < 0:
        raise ValueError(f"padding must be non‑negative, got {padding}")

    X = input.data
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if stride is None:
        stride = KH
    else:
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

    if len(shape_list(X)) != 3:
        raise ValueError(f"Input must be 3D (C,H,W), got shape {shape_list(X)}")

    C_in, H_in, W_in = shape_list(X)
    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Output dimension non‑positive: H_out={H_out}, W_out={W_out}. "
                         f"Check kernel size, stride, or padding.")

    # Pad input
    H_pad = H_in + 2 * padding
    W_pad = W_in + 2 * padding
    X_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                X_pad[c][i + padding][j + padding] = X[c][i][j]

    # Forward: compute max values and remember their positions.
    out_data = [[[0.0] * W_out for _ in range(H_out)] for _ in range(C_in)]
    # indices: for each output (c,i,j), store (h_pad, w_pad) of the chosen max.
    indices = [[[None] * W_out for _ in range(H_out)] for _ in range(C_in)]

    for c in range(C_in):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                max_val = -float('inf')
                max_pos = (h_start, w_start)

                # Scan the KH×KW window.
                for m in range(KH):
                    for n in range(KW):
                        val = X_pad[c][h_start + m][w_start + n]
                        # Tie‑break: if values equal, choose the smallest (m,n) offset.
                        # This makes the policy deterministic.
                        if val > max_val or (val == max_val and (m, n) < (max_pos[0]-h_start, max_pos[1]-w_start)):
                            max_val = val
                            max_pos = (h_start + m, w_start + n)

                out_data[c][i][j] = max_val
                indices[c][i][j] = max_pos

    out = Tensor(out_data, requires_grad=input.requires_grad, _children=(input,))

    # Backward closure: route dY only to the argmax positions.
    def _backward():
        dY = out.grad
        if dY is None:
            return
        if input.requires_grad:
            if input.grad is None:
                input.grad = zeros_like(X)
            for c in range(C_in):
                for i in range(H_out):
                    for j in range(W_out):
                        h_pad, w_pad = indices[c][i][j]
                        # Convert padded coordinates back to original X indices.
                        input.grad[c][h_pad - padding][w_pad - padding] += dY[c][i][j]

    out._backward = _backward
    return out