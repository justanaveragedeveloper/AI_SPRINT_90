"""
day50_tensor_parallelism.py

What this file is
-----------------
A small NumPy simulation of Megatron-LM-style tensor parallelism. It splits
a dense linear layer across a fake "world" of N ranks so the same math can
be checked without real GPUs.

Two layers are provided:

    ColumnParallelLinear
        Splits the output features (columns of W) across ranks. Each rank
        computes a slice of the output. Concatenating the slices along
        axis 1 gives back the dense result.

    RowParallelLinear
        Splits the input features (rows of W) across ranks. Each rank
        computes a partial product. Summing the partials and adding the
        bias once gives back the dense result.

What it costs
-------------
Let B = batch size, I = input features, O = output features, N = world size.

    Dense matmul         B * I * O multiplies
    Column-parallel      B * I * O / N multiplies per rank
    Row-parallel         B * (I / N) * O multiplies per rank
    Concat / sum         proportional to B * O per shard, B * O * N total

Each rank stores about I * O / N weights after sharding. If the rank
generates its own full weight matrix first (using a seed), it briefly
needs I * O memory. So peak memory during setup is not reduced by N.

These are arithmetic counts derived from the shapes. They are not
measurements. There are no timings, no GPU runs, and no benchmarks in
this file.

What this is NOT
----------------
    - Not real multi-GPU code. No NCCL, no CUDA, no distributed processes.
    - Not a full transformer block. Just the two linear layers.
    - Not a training framework. Forward only; no backward, no optimizer.
    - Not a real All-Reduce. It sums NumPy arrays in one process.
    - No speedup claims. There are no timings and no benchmarks here.

Reproducibility
---------------
    - If you want random initialization, pass an integer `seed`. The seed
      feeds `numpy.random.default_rng(seed)`, so the global NumPy RNG is
      never touched.
    - Same seed + same (in_features, out_features) = same full weight
      matrix on every rank. That is what makes the shards consistent
      slices of one matrix.
    - In practice, different seeds produce different weights. The test
      suite checks one specific pair of seeds; collisions are not ruled
      out for every possible pair.
    - If you pass `full_weight`, no randomness is used at all.

Input rules
-----------
    - Every public array must be a numpy.ndarray with dtype float32.
    - Non-finite inputs (NaN, Inf) are rejected.
    - Outputs are checked for finiteness before they are returned. If a
      matmul overflows to Inf, the caller gets a ValueError instead of Inf.
    - Any other dtype raises TypeError. There is no silent casting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FLOAT32 = np.float32


# ---------------------------------------------------------------------------
# Small input-check helpers.
#
# Each helper raises TypeError for the wrong kind of thing and ValueError
# for a bad value. That distinction matters: a caller catching TypeError
# knows they passed the wrong type, while ValueError means the type was
# fine but the value was not.
# ---------------------------------------------------------------------------


def _is_bool(value: object) -> bool:
    """True if `value` is a Python or NumPy bool.

    Needed because `bool` is a subclass of `int` in Python, so a naive
    integer check would let True through as 1.
    """
    return isinstance(value, (bool, np.bool_))


def _require_positive_int(name: str, value: object) -> int:
    """Return `value` as a plain int after checking it is a positive int.

    Raises TypeError if it is a bool or not an integer type.
    Raises ValueError if it is zero or negative.
    """
    if _is_bool(value) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be a positive integer, got {type(value).__name__}"
        )
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _require_valid_rank(rank: object, world_size: int) -> int:
    """Return `rank` as a plain int after checking 0 <= rank < world_size."""
    if _is_bool(rank) or not isinstance(rank, (int, np.integer)):
        raise TypeError(f"rank must be an integer, got {type(rank).__name__}")
    result = int(rank)
    if not (0 <= result < world_size):
        raise ValueError(f"rank must satisfy 0 <= rank < world_size, got {result}")
    return result


def _require_int_seed(seed: object) -> int:
    """Return `seed` as a plain int. Rejects bools (which are ints in Python)."""
    if _is_bool(seed) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    return int(seed)


def _require_float32_array(
    name: str,
    arr: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    """Check that `arr` is a float32 ndarray with the given rank and shape.

    Pass None for any axis whose size you do not want to check (the batch
    axis, for example). Raises TypeError for the wrong container or dtype;
    raises ValueError for the wrong number of dimensions or a bad axis size.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.dtype != FLOAT32:
        raise TypeError(f"{name} must have dtype float32, got {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr.ndim}")
    if len(shape) != ndim:
        raise ValueError("internal shape specification mismatch")
    for axis, expected in enumerate(shape):
        if expected is not None and arr.shape[axis] != expected:
            raise ValueError(
                f"{name} axis {axis} must have size {expected}, got {arr.shape[axis]}"
            )
    return arr


def _require_finite(name: str, arr: np.ndarray) -> None:
    """Raise ValueError if `arr` contains any NaN or Infinity."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


def _make_full_weight(in_features: int, out_features: int, seed: int) -> np.ndarray:
    """Build one full (in_features, out_features) float32 weight matrix.

    Uses a He-style scale of sqrt(2 / in_features), matching the starter
    blueprint. A real transformer might use a different scale; the exercise
    only needs every rank to produce the same matrix given the same seed.

    The random generator is `numpy.random.default_rng(seed)`, which is a
    local object. This function does not read or write the global NumPy
    RNG state.
    """
    rng = np.random.default_rng(seed)
    scale = FLOAT32(np.sqrt(2.0 / in_features))
    return rng.standard_normal((in_features, out_features)).astype(FLOAT32) * scale


# ---------------------------------------------------------------------------
# Column-parallel layer
# ---------------------------------------------------------------------------


class ColumnParallelLinear:
    """Column-parallel linear layer (Megatron-LM style).

    The full weight matrix W has shape (in_features, out_features). It is
    split across `world_size` ranks along the output-feature axis. Rank r
    gets columns:

        [r * local_out_features : (r + 1) * local_out_features]

    where `local_out_features = out_features // world_size`.

    Each rank computes:

        Y_r = X @ W_r + b_r     shape: (batch_size, local_out_features)

    Stacking the per-rank outputs side by side along axis 1 gives back the
    dense result:

        concat([Y_0, ..., Y_{N-1}], axis=1) == X @ W + b

    For the slices to line up, every rank must derive its shard from the
    SAME full matrix. Pass the same `full_weight` to every rank, or pass the
    same `seed` and dimensions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        *,
        full_weight: np.ndarray | None = None,
        full_bias: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self.in_features = _require_positive_int("in_features", in_features)
        self.out_features = _require_positive_int("out_features", out_features)
        self.world_size = _require_positive_int("world_size", world_size)
        self.rank = _require_valid_rank(rank, self.world_size)

        if self.out_features % self.world_size != 0:
            raise ValueError("out_features must be divisible by world_size")

        self.local_out_features = self.out_features // self.world_size

        if full_weight is None:
            if seed is None:
                raise ValueError(
                    "seed must be provided when full_weight is not supplied"
                )
            full_weight = _make_full_weight(
                self.in_features, self.out_features, _require_int_seed(seed)
            )
        else:
            _require_float32_array(
                "full_weight",
                full_weight,
                ndim=2,
                shape=(self.in_features, self.out_features),
            )
        _require_finite("full_weight", full_weight)

        if full_bias is None:
            full_bias = np.zeros(self.out_features, dtype=FLOAT32)
        else:
            _require_float32_array(
                "full_bias",
                full_bias,
                ndim=1,
                shape=(self.out_features,),
            )
        _require_finite("full_bias", full_bias)

        start = self.rank * self.local_out_features
        end = start + self.local_out_features
        # Copy the slices so the caller's arrays and this instance cannot
        # mutate each other through shared memory.
        self.weight = full_weight[:, start:end].copy()
        self.bias = full_bias[start:end].copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute this rank's slice of the output.

        Parameters
        ----------
        x : np.ndarray
            Shape (batch_size, in_features), dtype float32, finite only.

        Returns
        -------
        np.ndarray
            Shape (batch_size, local_out_features), dtype float32, finite.

        Raises
        ------
        TypeError
            If `x` is not an ndarray or not float32.
        ValueError
            If `x` has the wrong shape, contains non-finite values, or the
            matmul overflows to non-finite values.
        """
        _require_float32_array("x", x, ndim=2, shape=(None, self.in_features))
        _require_finite("x", x)
        # Suppress numpy's overflow warnings; we check the result directly
        # and raise a ValueError if it is not finite.
        with np.errstate(over="ignore", invalid="ignore"):
            y = x @ self.weight + self.bias
        _require_finite("column parallel output", y)
        return y


# ---------------------------------------------------------------------------
# Row-parallel layer
# ---------------------------------------------------------------------------


class RowParallelLinear:
    """Row-parallel linear layer (Megatron-LM style).

    The full weight matrix W has shape (in_features, out_features). It is
    split across `world_size` ranks along the input-feature axis. Rank r
    gets rows:

        [r * local_in_features : (r + 1) * local_in_features]

    where `local_in_features = in_features // world_size`.

    The input X is split along its feature axis into matching shards X_r,
    shape (batch_size, local_in_features). Each rank computes a partial
    product:

        Z_r = X_r @ W_r         shape: (batch_size, out_features)

    The full output is:

        Y = sum_r Z_r + b

    The bias is added exactly ONCE, after the sum. Adding it inside each
    rank's forward would multiply the bias by N.

    Note on `self.bias`
    -------------------
    Every rank stores a copy of the full bias for API symmetry with
    ColumnParallelLinear, but `forward` does NOT add it. Only
    `all_reduce_sum` applies it. This keeps the "bias once" rule in one
    place.

    Note on the name `all_reduce_sum`
    ---------------------------------
    The name is borrowed from the distributed collective it simulates.
    It is a single-process array sum, not a real collective. It also
    cannot verify that exactly one contribution arrived from each intended
    rank; it only checks that the sequence it was given is non-empty and
    internally consistent. The caller is responsible for supplying one
    partial per rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        *,
        full_weight: np.ndarray | None = None,
        full_bias: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        self.in_features = _require_positive_int("in_features", in_features)
        self.out_features = _require_positive_int("out_features", out_features)
        self.world_size = _require_positive_int("world_size", world_size)
        self.rank = _require_valid_rank(rank, self.world_size)

        if self.in_features % self.world_size != 0:
            raise ValueError("in_features must be divisible by world_size")

        self.local_in_features = self.in_features // self.world_size

        if full_weight is None:
            if seed is None:
                raise ValueError(
                    "seed must be provided when full_weight is not supplied"
                )
            full_weight = _make_full_weight(
                self.in_features, self.out_features, _require_int_seed(seed)
            )
        else:
            _require_float32_array(
                "full_weight",
                full_weight,
                ndim=2,
                shape=(self.in_features, self.out_features),
            )
        _require_finite("full_weight", full_weight)

        if full_bias is None:
            full_bias = np.zeros(self.out_features, dtype=FLOAT32)
        else:
            _require_float32_array(
                "full_bias",
                full_bias,
                ndim=1,
                shape=(self.out_features,),
            )
        _require_finite("full_bias", full_bias)

        start = self.rank * self.local_in_features
        end = start + self.local_in_features
        self.weight = full_weight[start:end, :].copy()
        # Full bias is stored on every rank; only `all_reduce_sum` applies it.
        self.bias = full_bias.copy()

    def forward(self, x_shard: np.ndarray) -> np.ndarray:
        """Compute this rank's partial product. Bias is NOT added here.

        Parameters
        ----------
        x_shard : np.ndarray
            Shape (batch_size, local_in_features), dtype float32, finite only.

        Returns
        -------
        np.ndarray
            Shape (batch_size, out_features), dtype float32, finite.

        Raises
        ------
        TypeError
            If `x_shard` is not an ndarray or not float32.
        ValueError
            If `x_shard` has the wrong shape, contains non-finite values,
            or the matmul overflows to non-finite values.
        """
        _require_float32_array(
            "x_shard", x_shard, ndim=2, shape=(None, self.local_in_features)
        )
        _require_finite("x_shard", x_shard)
        with np.errstate(over="ignore", invalid="ignore"):
            z = x_shard @ self.weight
        _require_finite("row parallel partial output", z)
        return z

    @staticmethod
    def all_reduce_sum(
        local_outputs: Sequence[np.ndarray],
        bias: np.ndarray,
    ) -> np.ndarray:
        """Sum the partial outputs from every rank, then add the bias once.

        In a real distributed run this would be an All-Reduce collective.
        Here it is a plain sum over a list of arrays. The bias is added
        exactly once, after summing — that is why it belongs here and not
        inside each rank's forward.

        The helper cannot verify that exactly one contribution arrived from
        each intended rank, because it does not know `world_size`. It only
        checks that the sequence is non-empty and that all elements agree
        in shape and dtype. The caller supplies one partial per rank.

        Parameters
        ----------
        local_outputs : Sequence[np.ndarray]
            One array per rank, each shape (batch_size, out_features),
            dtype float32, finite values only. Must not be empty.
        bias : np.ndarray
            Shape (out_features,), dtype float32, finite. Added once.

        Returns
        -------
        np.ndarray
            Shape (batch_size, out_features), dtype float32, finite.

        Raises
        ------
        TypeError
            If `local_outputs` is not a sequence, or if any element or the
            bias is not an ndarray or not float32.
        ValueError
            If `local_outputs` is empty, if the shapes do not all match,
            if the bias has the wrong shape, or if any input contains
            non-finite values.
        """
        if not isinstance(local_outputs, Sequence):
            raise TypeError(
                f"local_outputs must be a sequence, got {type(local_outputs).__name__}"
            )
        if len(local_outputs) == 0:
            raise ValueError("local_outputs must be a non-empty sequence")

        first = local_outputs[0]
        _require_float32_array("local_outputs[0]", first, ndim=2, shape=(None, None))
        _require_finite("local_outputs[0]", first)

        expected_shape = first.shape
        # Accumulate into a fresh array. Do not write into any caller array.
        total = np.zeros_like(first)

        for i, out in enumerate(local_outputs):
            _require_float32_array(
                f"local_outputs[{i}]", out, ndim=2, shape=expected_shape
            )
            _require_finite(f"local_outputs[{i}]", out)
            with np.errstate(over="ignore", invalid="ignore"):
                total = total + out

        _require_float32_array("bias", bias, ndim=1, shape=(expected_shape[1],))
        _require_finite("bias", bias)

        with np.errstate(over="ignore", invalid="ignore"):
            result = total + bias
        _require_finite("all_reduce_sum output", result)
        return result
