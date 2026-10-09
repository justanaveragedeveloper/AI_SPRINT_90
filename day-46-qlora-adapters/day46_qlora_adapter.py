"""A first-principles LoRA adapter with a symmetric Int8-quantized frozen base.

What it does
------------
Stores a linear layer's base weight as a numpy int8 matrix plus one float
scale, dequantizes it on each forward pass, and trains two small low-rank
matrices A (r x in_features) and B (out_features x r) on top. Only A and B
receive gradients; the base stays frozen.

What it is not
--------------
Not NF4, not double quantization, not paged optimizers, not a full QLoRA
training stack. No transformer, tokenizer, training loop, Adam, GPU, or
mixed precision. It's the math of low-rank adaptation on top of an
Int8-quantized linear layer.

Reproducibility
---------------
Deterministic after construction. Pass either `seed` or a local
`np.random.Generator`. This module never touches NumPy's global RNG.

Complexity (N=batch, k=in_features, d=out_features, r=rank)
-----------------------------------------------------------
    forward:    O(N*d*k + N*r*k + N*d*r)
    backward:   O(N*d*r + N*r*k)              (adapters only)
    update:     O(r*(d+k))
    quantize:   O(d*k)

Forward allocates a temporary float32 [d, k] dequantized base every call.
That temporary is why persistent-memory savings on the base (roughly 4x,
int8 vs float32) are not the same as peak-memory savings.

Design notes
------------
- Local Generator over global RNG: keeps layers independent of caller code.
- B initialized to zero: adapter contribution is exactly zero at step 0,
  so the layer's initial output equals the dequantized base exactly.
- Dequantization recomputed per forward rather than cached: trades a
  per-call [d, k] allocation for lower persistent memory.
- Manual gradients: educational, not a general autograd replacement.

Public mutable state
--------------------
``w_quant``, ``w_scale``, ``lora_A``, ``lora_B``, ``grad_A``, and ``grad_B``
are plain attributes. The class does not prevent external assignment. It
does run shape/dtype/finiteness checks before using them, so obviously
broken state fails loudly instead of producing garbage silently. That is
detection, not encapsulation.
"""

from __future__ import annotations

import math

import numpy as np


def _as_finite_2d(name: str, arr, cols: int) -> np.ndarray:
    """Coerce to a finite float32 2D array with the given number of columns."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be 2D with second dimension {cols}, got {a.shape}"
        )
    if a.shape[1] != cols:
        raise ValueError(f"{name} must have shape [N, {cols}], got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite values")
    return a


def _positive_int(name: str, value) -> int:
    # bool is a subclass of int, so True would silently become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive int, not {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")
    return value


def _nonneg_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative int, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value}")
    return value


def _real(name: str, value, *, nonneg: bool) -> float:
    """Reject bool, str, bytes, and non-finite values, then return a float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a real number, not a string")
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite")
    if nonneg and f < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return f


class QuantizedLinearLoRA:
    """Linear layer with a frozen Int8-quantized base and trainable rank-r adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.in_features = _positive_int("in_features", in_features)
        self.out_features = _positive_int("out_features", out_features)
        self.r = _nonneg_int("r", r)
        self.alpha = _real("alpha", alpha, nonneg=True)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

        if seed is not None and rng is not None:
            raise ValueError("provide either seed or rng, not both")
        if rng is None:
            rng = np.random.default_rng(seed)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        raw_w0 = rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 / self.in_features),
            size=(self.out_features, self.in_features),
        ).astype(np.float32)
        self.w_scale, self.w_quant = self._quantize_int8(raw_w0)

        self.lora_A: np.ndarray | None = None
        self.lora_B: np.ndarray | None = None
        self.grad_A: np.ndarray | None = None
        self.grad_B: np.ndarray | None = None

        if self.r > 0:
            self.lora_A = rng.normal(
                loc=0.0,
                scale=1.0 / math.sqrt(self.r),
                size=(self.r, self.in_features),
            ).astype(np.float32)
            self.lora_B = np.zeros((self.out_features, self.r), dtype=np.float32)
            self.grad_A = np.zeros_like(self.lora_A)
            self.grad_B = np.zeros_like(self.lora_B)

    @staticmethod
    def _quantize_int8(w: np.ndarray) -> tuple[float, np.ndarray]:
        """Symmetric Int8 quantization: scale = max|W|/127, round, clip, cast."""
        w_arr = np.asarray(w)
        if w_arr.ndim != 2:
            raise ValueError(f"weight matrix must be 2D, got shape {w_arr.shape}")
        if w_arr.size == 0:
            raise ValueError("weight matrix must not be empty")

        w_float = w_arr.astype(np.float32, copy=False)
        if not np.isfinite(w_float).all():
            raise ValueError("weight matrix must contain only finite values")

        max_abs = float(np.max(np.abs(w_float)))
        if max_abs == 0.0:
            return 1.0, np.zeros_like(w_float, dtype=np.int8)

        scale_f32 = np.float32(max_abs / 127.0)
        # A finite max_abs can still underflow to a zero float32 scale
        # (subnormal divided by 127), which would silently divide by zero.
        if scale_f32 == 0.0 or not np.isfinite(scale_f32):
            raise ValueError(f"scale {max_abs / 127.0} is not a representable float32")

        w_quant = np.clip(np.round(w_float / scale_f32), -128, 127).astype(np.int8)
        return float(scale_f32), w_quant

    def _check_base_state(self) -> None:
        """Guard against external mutation of w_quant / w_scale."""
        if not isinstance(self.w_quant, np.ndarray):
            raise TypeError(
                f"w_quant must be np.ndarray, got {type(self.w_quant).__name__}"
            )
        if self.w_quant.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"w_quant shape {self.w_quant.shape} != "
                f"{(self.out_features, self.in_features)}"
            )
        if self.w_quant.dtype != np.int8:
            raise ValueError(f"w_quant dtype {self.w_quant.dtype} != int8")

        if isinstance(self.w_scale, (bool, np.bool_)) or not isinstance(
            self.w_scale, (int, float, np.floating)
        ):
            raise TypeError(
                f"w_scale must be a real number, got {type(self.w_scale).__name__}"
            )
        ws = float(self.w_scale)
        if not math.isfinite(ws) or ws <= 0.0:
            raise ValueError(f"w_scale must be finite and > 0, got {self.w_scale}")

    def _dequantize_w0(self) -> np.ndarray:
        self._check_base_state()
        return self.w_quant.astype(np.float32) * np.float32(self.w_scale)

    def _check_lora_shapes(self) -> None:
        if self.r == 0:
            return
        if self.lora_A is None or self.lora_B is None:
            raise RuntimeError("rank > 0 but LoRA arrays are missing")
        if self.lora_A.shape != (self.r, self.in_features):
            raise ValueError(
                f"lora_A shape {self.lora_A.shape} != {(self.r, self.in_features)}"
            )
        if self.lora_B.shape != (self.out_features, self.r):
            raise ValueError(
                f"lora_B shape {self.lora_B.shape} != {(self.out_features, self.r)}"
            )

    def _check_gradient_shapes(self) -> None:
        if self.r == 0:
            return
        if self.grad_A is None or self.grad_B is None:
            raise RuntimeError("rank > 0 but gradient arrays are missing")
        if self.grad_A.shape != (self.r, self.in_features):
            raise ValueError(
                f"grad_A shape {self.grad_A.shape} != {(self.r, self.in_features)}"
            )
        if self.grad_B.shape != (self.out_features, self.r):
            raise ValueError(
                f"grad_B shape {self.grad_B.shape} != {(self.out_features, self.r)}"
            )

    def forward(self, x) -> np.ndarray:
        """Return x @ W0_dequant.T + (alpha/r) * (x @ A.T) @ B.T."""
        x_arr = _as_finite_2d("x", x, self.in_features)
        self._check_lora_shapes()

        out = x_arr @ self._dequantize_w0().T
        if self.r > 0:
            out = out + (x_arr @ self.lora_A.T) @ self.lora_B.T * np.float32(
                self.scaling
            )

        if not np.isfinite(out).all():
            raise FloatingPointError("forward produced NaN or Inf")
        return out.astype(np.float32, copy=False)

    def backward(self, x, grad_output) -> None:
        """Compute grad_A and grad_B. Overwrites them; does not accumulate."""
        x_arr = _as_finite_2d("x", x, self.in_features)
        g_arr = _as_finite_2d("grad_output", grad_output, self.out_features)
        if x_arr.shape[0] != g_arr.shape[0]:
            raise ValueError(
                f"batch mismatch: x has {x_arr.shape[0]} rows, grad_output has {g_arr.shape[0]}"
            )
        self._check_lora_shapes()

        if self.r == 0:
            return

        s = np.float32(self.scaling)
        intermediate = x_arr @ self.lora_A.T
        grad_B = (g_arr.T @ intermediate) * s
        grad_A = ((g_arr @ self.lora_B).T @ x_arr) * s

        if not (np.isfinite(grad_A).all() and np.isfinite(grad_B).all()):
            raise FloatingPointError("backward produced NaN or Inf gradients")

        self.grad_A = grad_A
        self.grad_B = grad_B

    def update_adapters(self, lr: float = 0.01) -> None:
        """Gradient descent on A and B. Never touches the frozen base."""
        lr_f = _real("lr", lr, nonneg=True)

        if self.r == 0:
            return

        self._check_lora_shapes()
        self._check_gradient_shapes()

        if not (np.isfinite(self.grad_A).all() and np.isfinite(self.grad_B).all()):
            raise FloatingPointError("cannot update with non-finite gradients")

        new_A = self.lora_A - np.float32(lr_f) * self.grad_A
        new_B = self.lora_B - np.float32(lr_f) * self.grad_B
        if not (np.isfinite(new_A).all() and np.isfinite(new_B).all()):
            raise FloatingPointError("adapter update produced NaN or Inf")

        # In-place so external references to the adapter arrays stay valid.
        self.lora_A[...] = new_A
        self.lora_B[...] = new_B
