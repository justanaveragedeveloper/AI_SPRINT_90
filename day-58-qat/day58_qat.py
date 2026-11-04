"""day58_qat.py — Auditable NumPy reference for symmetric uniform fake quantization
and the Straight-Through Estimator (STE).

Problem this solves
-------------------
Post-training quantization (PTQ) can lose accuracy because the network is trained
in floating point and only quantized afterward. Quantization-Aware Training (QAT)
simulates the quantize-then-dequantize round trip in the forward pass so the master
weights learn to be robust to grid rounding. The backward pass is not
differentiable through `round()`, so QAT substitutes the Straight-Through
Estimator (identity Jacobian) to let gradients flow.

This module provides exactly two public operations:
  * per-tensor symmetric uniform fake quantization (forward), and
  * the identity STE backward helper.

Complexity
----------
For a weight tensor with P elements:
  * Time: O(P). One vectorized pass each for np.abs, np.max (reduce),
    division, np.rint, np.clip, and multiplication. Explicit Python loops
    over tensor elements: 0.
  * Space: O(P) for the output array, plus O(P) for each intermediate
    (|W|, W/Delta, rounded, clipped). Peak temporary memory is a small
    constant multiple of P; the implementation is not zero-copy. Peak
    memory is stated asymptotically, not measured.

What this is NOT
----------------
* Not INT8 storage. Output dtype follows the documented promotion rule; no
  integer tensors, no INT8 GEMM, no memory savings.
* Not a framework. No optimizer, autograd engine, or training loop.
* Not per-channel quantization. Per-tensor baseline only.
* Not the true derivative of round(). The STE is an optimization heuristic.
* Not a QAT accuracy guarantee. QAT can reduce quantization-induced
  degradation but cannot eliminate it; we make no accuracy claims.

Reproducibility
---------------
The fake-quantization and STE math are deterministic; no seed is required.
`FakeQuantizeLinear` initializes weights with `np.random.default_rng(seed)` —
never the global `np.random` state.

Seed policy:
  * seed=None (default): uses a fixed internal seed constant (0) so that
    construction without an explicit seed is deterministic. Two layers
    constructed with the same arguments and no seed have *identical*
    weights. This default exists for educational reproducibility: a
    freshly-run script produces the same output every time.
  * seed=int: reproducible across calls. Pass a distinct seed per layer
    if you want distinct initializations in a training pipeline.
  * seed=np.random.Generator: used as-is; its state advances on use.
    Two layers built from the same Generator object will have different
    initial weights.

  In training code, always pass an explicit seed or a fresh Generator per
  layer. Relying on seed=None for training will silently give every layer
  the same initialization.

Rounding convention: `np.rint` (banker's rounding / ties-to-even). A worked
tie case is exercised in `test_tie_to_even_at_halfway_points`.

Design note — per-tensor symmetric vs alternatives
--------------------------------------------------
Chose per-tensor symmetric over per-channel or asymmetric because the spec
requires an auditable O(P) reference and per-tensor has a single scalar scale
(easy to hand-verify, easy to test for symmetry). Asymmetric (zero-point)
would remove sign symmetry and add a per-tensor offset. Per-channel would
require per-axis max-reduction and per-axis scaling, breaking the scalar-scale
story. Matters when weights within a tensor have very different magnitudes
(per-channel tends to preserve accuracy better there); this baseline pays a
small accuracy cost on such tensors.

Design note — clipping is defensive
-----------------------------------
Under this calibration, Delta = max_abs/q_max, so |W/Delta| <= q_max by
construction. The np.clip(q, q_min, q_max) step is therefore a mathematical
no-op for calibration-anchored inputs — it protects against float-rounding
overshoot by at most one integer level, not against genuine out-of-range
values. Removing the clip would change output only in rare float-rounding
edge cases; the clip is kept as a contract anchor (`levels in [q_min, q_max]`)
and because it costs one vectorized pass. Do not mistake this for
demonstrated numerical necessity.

Design note — floating-point master weights
-------------------------------------------
Chose to keep master weights in floating point (rather than materializing
integer tensors) because the training loop needs full-precision gradient
accumulation to move past quantization-grid boundaries. Chose identity STE
over masked/clipped STE because the exercise specifies identity; clipped-STE
is a different contract requiring per-element clipping masks in the API.

Design note — single quantization kernel
----------------------------------------
Chose one shared kernel (`_quantize_parts`) over two separately-written paths
(one for integer levels, one for float output) because a duplicated kernel is
a drift risk: a change to rounding, scale, or clip could update one path and
not the other, and equivalence would only be asserted by documentation. The
shared kernel computes (levels int64, delta, max_abs) once; both the test
helper and the public fake-quantization path consume it. For all accepted
bit widths (q_max <= 32767), int64 levels are exactly representable in every
accepted working float dtype, so no precision is lost by round-tripping
through int64.
"""

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt

__all__ = ["FakeQuantizeLinear", "straight_through_estimator_backward"]


_BITS_MIN = 2
_BITS_MAX = 16
_DEFAULT_INIT_SEED = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_bits(bits: object) -> int:
    if isinstance(bits, bool):
        raise TypeError(f"bits must be an integer, not bool; got {bits!r}")
    if not isinstance(bits, numbers.Integral):
        raise TypeError(f"bits must be an integer, got {type(bits).__name__}")
    b = int(bits)
    if not (_BITS_MIN <= b <= _BITS_MAX):
        raise ValueError(
            f"bits must be in [{_BITS_MIN}, {_BITS_MAX}]; got {b}. "
            f"Lower bound avoids q_max=0; upper bound keeps q_max=32767 "
            f"exactly representable as a float32 integer."
        )
    return b


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool; got {value!r}")
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be positive; got {v}")
    return v


def _validate_numeric_array(arr: np.ndarray, name: str) -> None:
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype.kind not in ("f", "i", "u"):
        raise TypeError(
            f"{name} must be real numeric (float or integer); "
            f"got dtype {arr.dtype} (kind={arr.dtype.kind!r})"
        )
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains NaN or +/-infinity; refusing to sanitize silently"
        )


def _output_float_dtype(in_dtype: np.dtype) -> np.dtype:
    """Documented output dtype contract.

    - float16 -> float32 (promoted): float16's mantissa cannot exactly
      represent the q_max grid for higher bit widths, and round-trip
      division/multiplication compounds float16 error.
    - any other floating dtype (float32, float64, longdouble/float128)
      -> preserved. We do not demote: the caller selected the precision.
    - any integer dtype -> float64 (promoted): output must be floating
      point; float64 avoids precision loss for typical int32 ranges.
      Known limitation: int64 values above 2**53 lose low-order bits in
      float64. Documented, not a bug; the alternative (raise for huge
      integers) would reject legitimate inputs.
    """
    if in_dtype.kind == "f":
        if in_dtype == np.dtype(np.float16):
            return np.dtype(np.float32)
        return in_dtype
    return np.dtype(np.float64)


# ---------------------------------------------------------------------------
# Core kernel — single source of quantization arithmetic
# ---------------------------------------------------------------------------
def _quantization_bounds(bits: int) -> tuple[int, int]:
    """Return (q_min, q_max) for symmetric signed integer quantization.

    For bit width b, the symmetric signed range is
    [-(2**(b-1) - 1), +(2**(b-1) - 1)]. This is deliberately *not* the
    asymmetric two's-complement range [-2**(b-1), 2**(b-1) - 1]: symmetric
    quantization requires |q_min| == |q_max| so that zero maps to zero and
    Q(-W) == -Q(W).

    Exposed as a private helper so tests can verify the symmetric-bound
    contract directly. Under the current calibration, the behavioral
    difference between symmetric and asymmetric bounds is invisible to
    output-only tests (because Delta anchors at q_max and the lower clip
    bound is unreachable); this helper makes the contract testable at the
    source.
    """
    q_max = (1 << (bits - 1)) - 1
    return -q_max, q_max


def _quantize_parts(
    w: np.ndarray, bits: int
) -> tuple[np.ndarray, np.floating, np.floating]:
    """Single source of quantization arithmetic.

    Returns ``(levels, delta, max_abs)``:
      * levels  : int64 array of clipped integer levels, same shape as w
      * delta   : per-tensor scale = max_abs / q_max (0 if all-zero input)
      * max_abs : per-tensor calibration rail = max(|W|)

    Assumes w is non-empty and finite with a floating dtype. Caller is
    responsible for promotion and validation.

    For all-zero input, returns ``(zeros int64, dtype-typed 0, dtype-typed 0)``;
    the caller must check ``delta == 0`` and short-circuit.

    For non-zero input whose ``max_abs / q_max`` underflows to zero in the
    working dtype, raises ValueError.
    """
    q_min, q_max = _quantization_bounds(bits)
    max_abs = np.max(np.abs(w))
    if max_abs == 0:
        return np.zeros(w.shape, dtype=np.int64), w.dtype.type(0), w.dtype.type(0)
    delta = max_abs / w.dtype.type(q_max)
    if delta == 0:
        raise ValueError(
            f"max(|W|)={max_abs} underflows to zero scale in dtype "
            f"{w.dtype}; scale weights up or use float64"
        )
    scaled = w / delta
    rounded = np.rint(scaled)  # ties-to-even
    clipped = np.clip(rounded, q_min, q_max)  # defensive; see module docstring
    return clipped.astype(np.int64), delta, max_abs


def _symmetric_quantize_levels(w: np.ndarray, bits: int) -> np.ndarray:
    """Return integer quantization levels (int64). Thin wrapper over the shared kernel.

    Exposed so tests can verify integer-level bounds, calibration anchors,
    and tie rounding directly, without reconstructing levels from dequantized
    floats. The levels returned are the exact levels the public fake-
    quantization path uses, because both call _quantize_parts.
    """
    levels, _, _ = _quantize_parts(w, bits)
    return levels


def _symmetric_fake_quantize_float(w: np.ndarray, bits: int) -> np.ndarray:
    """Per-tensor symmetric uniform fake quantization on a float array.

    Assumes w is non-empty and finite. Caller is responsible for promotion
    and validation. Uses _quantize_parts as the single source of levels,
    delta, and calibration rail; then multiplies levels by delta and clamps
    the result back to ±max_abs (see overflow note below).
    """
    levels, delta, max_abs = _quantize_parts(w, bits)
    if delta == 0:
        # All-zero input: _quantize_parts returned delta=0; return zeros
        # without dividing.
        return np.zeros_like(w)
    # At extreme magnitudes (e.g., w[i] = np.finfo(dtype).max), delta rounds
    # such that q_max * delta overflows to inf in the working dtype even
    # though max_abs is finite. Clamping to the calibration rail restores
    # the documented "finite input -> finite output" contract.
    with np.errstate(over="ignore"):
        result = levels.astype(w.dtype) * delta
    result = np.clip(result, -max_abs, max_abs)
    return result.astype(w.dtype, copy=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class FakeQuantizeLinear:
    """Symmetric per-tensor fake quantization, plus an optional linear forward.

    Parameters
    ----------
    in_features : int
        Positive integer. Used by ``__init__`` for weight init and by
        ``forward`` for shape validation. ``fake_quantize`` accepts weight
        tensors of any shape (including the starter's 1-D 100-element
        array against 10x10 metadata).
    out_features : int
        Positive integer. Used by ``__init__`` for weight shape and by
        ``forward`` for output shape.
    bits : int
        Quantization bit width in [2, 16]. Defaults to 8.
    bias : bool
        If True (default), ``forward`` adds ``self.bias`` (a float32 vector
        of shape (out_features,)). If False, ``self.bias`` is None and
        forward omits the addition.
    seed : int, np.random.Generator, or None
        RNG for weight initialization. See module docstring "Seed policy".

    Attributes
    ----------
    weight : np.ndarray, shape (out_features, in_features), float32
        Master weights; the raw float32 tensor a training loop would
        update. ``forward`` uses ``fake_quantize(weight)``.
    bias : np.ndarray of shape (out_features,) float32, or None
        Bias vector, or None if ``bias=False``. Public so tests and callers
        can set known values directly.
    use_bias : bool
        Whether ``forward`` adds a bias. Matches the constructor argument.

    Example
    -------
    >>> import numpy as np
    >>> from day58_qat import FakeQuantizeLinear, straight_through_estimator_backward
    >>> layer = FakeQuantizeLinear(in_features=10, out_features=10, bits=8)
    >>> w = np.linspace(-1, 1, 100, dtype=np.float32)
    >>> w_q = layer.fake_quantize(w)   # 1-D ok; not reshaped to (10, 10)
    >>> w_q.shape
    (100,)
    >>> grad_in = straight_through_estimator_backward(np.ones((4, 4)))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.in_features = _validate_positive_int(in_features, "in_features")
        self.out_features = _validate_positive_int(out_features, "out_features")
        self.bits = _validate_bits(bits)

        if not isinstance(bias, bool):
            raise TypeError(
                f"bias must be bool; got {type(bias).__name__}. "
                f"Use bias=True or bias=False explicitly."
            )
        self.use_bias = bias

        if isinstance(seed, np.random.Generator):
            self._rng: np.random.Generator = seed
        elif seed is None:
            self._rng = np.random.default_rng(_DEFAULT_INIT_SEED)
        elif isinstance(seed, bool):
            raise TypeError(
                "seed must be int, np.random.Generator, or None; bool is not accepted"
            )
        elif isinstance(seed, numbers.Integral):
            self._rng = np.random.default_rng(int(seed))
        else:
            raise TypeError(
                f"seed must be int, np.random.Generator, or None; "
                f"got {type(seed).__name__}"
            )

        std = float(np.sqrt(2.0 / self.in_features))
        self.weight: np.ndarray = (
            self._rng.standard_normal((self.out_features, self.in_features)) * std
        ).astype(np.float32)

        if self.use_bias:
            self.bias: np.ndarray | None = np.zeros(self.out_features, dtype=np.float32)
        else:
            self.bias = None

    # ------------------------------------------------------------------
    def fake_quantize(self, w: npt.ArrayLike) -> np.ndarray:
        """Apply symmetric per-tensor fake quantization to ``w``.

        The output is floating point and lies on the per-tensor quantization
        grid ``k * Delta`` for integer ``k`` in ``[q_min, q_max]``, where
        ``Delta = max(|W|) / q_max`` and ``q_max = 2^(bits-1) - 1``. The
        maximum-magnitude element of ``w`` maps to ``±max(|W|)`` up to one
        float ULP.

        Parameters
        ----------
        w : array-like
            Weight tensor of any shape (including 1-D). Must be non-empty,
            finite, real numeric. The ``(out_features, in_features)`` shape
            is *not* enforced here; that constraint applies only to
            ``forward``.

        Returns
        -------
        np.ndarray
            Same shape as ``w``. Dtype contract:
              * float16 -> float32 (promoted)
              * float32 / float64 / longdouble -> preserved
              * integer -> float64 (promoted)
        """
        arr = np.asarray(w)
        _validate_numeric_array(arr, "w")
        out_dtype = _output_float_dtype(arr.dtype)
        w_f = arr.astype(out_dtype, copy=False)
        return _symmetric_fake_quantize_float(w_f, self.bits)

    # ------------------------------------------------------------------
    def forward(self, x: npt.ArrayLike) -> np.ndarray:
        """Linear forward with fake-quantized weights.

        Computes ``y = x @ fake_quantize(W).T (+ self.bias)``. Activations
        are *not* quantized in this baseline. This is not a complete
        trainable layer: no autograd, no optimizer, no backward integration.
        Call ``straight_through_estimator_backward`` manually if you need
        STE gradients.

        Parameters
        ----------
        x : array-like
            Shape ``(..., in_features)``. Last axis must equal
            ``in_features``. Must be non-empty and finite.

        Returns
        -------
        np.ndarray
            Shape ``(..., out_features)``. dtype: float32 for float16/float32
            input, input dtype for float64/longdouble, float64 for integer.
        """
        arr = np.asarray(x)
        if arr.ndim == 0:
            raise ValueError("x must have at least one dimension; got 0-d")
        if arr.shape[-1] != self.in_features:
            raise ValueError(
                f"x last axis must equal in_features={self.in_features}; "
                f"got shape {arr.shape}"
            )
        _validate_numeric_array(arr, "x")

        out_dtype = _output_float_dtype(arr.dtype)
        x_f = arr.astype(out_dtype, copy=False)

        w_q = self.fake_quantize(self.weight)  # float32
        w_q_f = w_q.astype(out_dtype, copy=False)

        y = x_f @ w_q_f.T
        if self.use_bias and self.bias is not None:
            y = y + self.bias.astype(out_dtype, copy=False)
        return y


# ---------------------------------------------------------------------------
# STE
# ---------------------------------------------------------------------------
def straight_through_estimator_backward(grad_output: npt.ArrayLike) -> np.ndarray:
    """Identity Jacobian for the Straight-Through Estimator.

    The STE substitutes ``d Q(W) / dW = I`` (identity) during backprop,
    because ``round()`` has zero derivative almost everywhere and would
    otherwise kill gradient flow. This is an optimization approximation,
    not the true derivative of rounding.

    This function is the *identity* STE only. It does not clip or mask the
    gradient. Clipped / masked STE variants (used in some QAT papers) are
    deliberately out of scope.

    Parameters
    ----------
    grad_output : array-like
        Incoming gradient, any shape. Must be non-empty, finite, real
        numeric.

    Returns
    -------
    np.ndarray
        An owned copy of ``grad_output`` with identical shape and dtype.
        Returning a copy (rather than the input array or a view) prevents
        downstream weight updates from silently overwriting the caller's
        gradient buffer. Cost: one O(P) allocation per call.

    Rationale
    ---------
    Chose owned copy over aliasing because STE aliasing lets downstream
    optimizer code mutate a caller-held gradient array in place, producing
    hard-to-reproduce bugs when the same gradient tensor is reused for
    debugging or checkpointing. Matters when gradients are shared across
    multiple consumers, common in micro-batched or gradient-accumulation
    pipelines.
    """
    arr = np.asarray(grad_output)
    _validate_numeric_array(arr, "grad_output")
    return arr.copy()
