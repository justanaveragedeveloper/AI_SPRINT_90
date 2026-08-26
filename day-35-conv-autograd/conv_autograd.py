"""
conv_autograd.py
Day 35: Autograd integration for 2D convolution and max pooling.
Provides Tensor, Conv2D, and MaxPool2D with explicit backward closures.
"""

class Tensor:
    """Simple Tensor with data (nested list) and autograd support."""

    def __init__(self,
                 data: list | float,
                 requires_grad: bool = False,
                 _children: tuple['Tensor', ...] = ()):
        if isinstance(data, (int, float)):
            self.data = data
            self.shape = ()
        else:
            self.data = data
            self.shape = self._infer_shape(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)

    @staticmethod
    def _infer_shape(data):
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        first = data[0]
        if not isinstance(first, list):
            return (len(data),)
        return (len(data),) + Tensor._infer_shape(first)

    def zero_grad(self):
        self.grad = None

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        if self.grad is None:
            self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.grad is not None})"


# Helper functions
def shape_list(x):
    if not isinstance(x, list):
        return ()
    if not x:
        return (0,)
    return (len(x),) + shape_list(x[0])

def zeros_like(x):
    if not isinstance(x, list):
        return 0.0
    return [zeros_like(row) for row in x]

def add_inplace(a, b):
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
# Conv2D autograd function
# ----------------------------------------------------------------------

def conv2d(input: Tensor,
           weight: Tensor,
           bias: Tensor | None,
           stride: int = 1,
           padding: int = 0) -> Tensor:
    # Input validation
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be non‑negative, got {padding}")

    X = input.data
    W = weight.data
    b = bias.data if bias is not None else None

    # Validate shapes
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

    H_out = (H_in + 2 * padding - KH) // stride + 1
    W_out = (W_in + 2 * padding - KW) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Output dimension non‑positive: H_out={H_out}, W_out={W_out}. "
                         f"Check kernel size, stride, or padding.")

    H_pad = H_in + 2 * padding
    W_pad = W_in + 2 * padding
    X_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                X_pad[c][i + padding][j + padding] = X[c][i][j]

    out_data = [[[0.0] * W_out for _ in range(H_out)] for _ in range(C_out)]
    for o in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                val = b[o] if b is not None else 0.0
                for c in range(C_in):
                    for m in range(KH):
                        for n in range(KW):
                            val += (X_pad[c][i*stride + m][j*stride + n] *
                                    W[o][c][m][n])
                out_data[o][i][j] = val

    # Derive requires_grad from parents
    requires_grad = (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )

    # Fix bias truthiness check
    out = Tensor(out_data, requires_grad=requires_grad,
                 _children=(input, weight, bias) if bias is not None else (input, weight))

    def _backward():
        dY = out.grad
        if dY is None:
            return

        if input.requires_grad and input.grad is None:
            input.grad = zeros_like(X)
        if weight.requires_grad and weight.grad is None:
            weight.grad = zeros_like(W)
        if bias is not None and bias.requires_grad and bias.grad is None:
            bias.grad = zeros_like(b)

        dX_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]

        for o in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    grad = dY[o][i][j]
                    if bias is not None and bias.requires_grad:
                        bias.grad[o] += grad
                    for c in range(C_in):
                        for m in range(KH):
                            for n in range(KW):
                                h_idx = i * stride + m
                                w_idx = j * stride + n
                                x_val = X_pad[c][h_idx][w_idx]
                                if weight.requires_grad:
                                    weight.grad[o][c][m][n] += grad * x_val
                                if input.requires_grad:
                                    dX_pad[c][h_idx][w_idx] += grad * W[o][c][m][n]

        if input.requires_grad:
            for c in range(C_in):
                for i in range(H_in):
                    for j in range(W_in):
                        input.grad[c][i][j] += dX_pad[c][i + padding][j + padding]

    out._backward = _backward
    return out


# ----------------------------------------------------------------------
# MaxPool2D autograd function
# ----------------------------------------------------------------------

def maxpool2d(input: Tensor,
              kernel_size: int | tuple[int, int],
              stride: int | None = None,
              padding: int = 0) -> Tensor:
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

    H_pad = H_in + 2 * padding
    W_pad = W_in + 2 * padding
    X_pad = [[[0.0] * W_pad for _ in range(H_pad)] for _ in range(C_in)]
    for c in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                X_pad[c][i + padding][j + padding] = X[c][i][j]

    out_data = [[[0.0] * W_out for _ in range(H_out)] for _ in range(C_in)]
    indices = [[[None] * W_out for _ in range(H_out)] for _ in range(C_in)]

    for c in range(C_in):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                max_val = -float('inf')
                max_pos = (h_start, w_start)
                for m in range(KH):
                    for n in range(KW):
                        val = X_pad[c][h_start + m][w_start + n]
                        if val > max_val or (val == max_val and (m, n) < (max_pos[0]-h_start, max_pos[1]-w_start)):
                            max_val = val
                            max_pos = (h_start + m, w_start + n)
                out_data[c][i][j] = max_val
                indices[c][i][j] = max_pos

    out = Tensor(out_data, requires_grad=input.requires_grad, _children=(input,))

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
                        input.grad[c][h_pad - padding][w_pad - padding] += dY[c][i][j]

    out._backward = _backward
    return out