"""
A small, readable simulation of FSDP / ZeRO-3 parameter sharding.

Why this exists
---------------
In normal data parallelism every GPU holds a full copy of the weights,
the gradients, and the optimizer state. FSDP shards all three across the
N GPUs and re-assembles them just in time: All-Gather the weights before
a forward pass, Reduce-Scatter the gradients before the optimizer step.
This file simulates that ownership model with plain NumPy so the
invariants can be tested without a GPU cluster.

What it is
----------
- Each FSDPModule is one simulated rank. It stores only its own
  contiguous slice of the flat weight (and bias, if enabled).
- The full weight is reconstructed on demand from a rank-ordered peer
  list. The module never keeps a hidden copy.
- The optimizer is SGD. It updates only the module it's bound to, and
  it validates everything before mutating anything.

What it is not
--------------
- Not a distributed runtime. No GPUs, no NCCL, no process forks, no
  async collectives, no overlap of communication with compute.
- Not an autograd engine. The gradient helpers assume Y = X @ W + b
  and a caller-supplied dY.
- Not a mixed-precision or checkpointing implementation.
- Not a memory benchmark. The MemoryAccounting class does arithmetic
  on numbers you supply; it doesn't measure anything.

Complexity (P parameters, N ranks)
----------------------------------
- Shard / reconstruct: O(P) time, O(P/N) per-rank storage.
- Simulated All-Gather: O(P) work on the gathering rank.
- Simulated Reduce-Scatter: O(P·N) in Python (real NCCL is O(P) bytes
  with a tree reduction — this simulation does not model that).
- Optimizer step: O(P/N) per rank.

Determinism
-----------
No randomness. Same inputs → bit-identical outputs. No seed needed.

Reproducibility
---------------
Partition policy is fully specified by shard_sizes(P, N).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_world_size(world_size: int) -> None:
    # bool is an int in Python; rejecting it here catches `world_size=True`.
    if isinstance(world_size, bool) or not isinstance(world_size, int):
        raise TypeError(f"world_size must be int, got {type(world_size).__name__}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")


def _check_rank(rank: int, world_size: int) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank must be int, got {type(rank).__name__}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")


def _check_dim(d: object, name: str) -> int:
    if isinstance(d, bool) or not isinstance(d, int):
        raise TypeError(f"{name} must be int, got {type(d).__name__}")
    if d < 0:
        raise ValueError(f"{name} must be >= 0, got {d}")
    return d


def _check_shape(shape: tuple[int, ...]) -> int:
    if not isinstance(shape, tuple):
        raise TypeError(f"shape must be a tuple, got {type(shape).__name__}")
    numel = 1
    for i, d in enumerate(shape):
        numel *= _check_dim(d, f"shape[{i}]")
    return numel


# ---------------------------------------------------------------------------
# Contiguous partition policy
# ---------------------------------------------------------------------------


def shard_sizes(total: int, world_size: int) -> list[int]:
    """
    Split `total` elements across `world_size` ranks. The first
    `total % world_size` ranks get one extra element, so the sum always
    equals `total`. Empty shards are allowed (P < N).

    Contiguous, not strided. Contiguous keeps each rank's shard in one
    memory block, matches the flat-parameter layout real frameworks use,
    and makes reconstruction a plain concatenate. Strided (rank::N) works
    too but is easy to get wrong when merging — one swapped index and
    you have a silent corruption.
    """
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError(f"total must be int, got {type(total).__name__}")
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    _check_world_size(world_size)
    base, rem = divmod(total, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def shard_offsets(total: int, world_size: int) -> list[int]:
    """Boundaries into the flat vector; length world_size + 1."""
    offsets = [0]
    for s in shard_sizes(total, world_size):
        offsets.append(offsets[-1] + s)
    return offsets


# ---------------------------------------------------------------------------
# Simulated collectives
# ---------------------------------------------------------------------------


def all_gather(
    shards: Sequence[np.ndarray],
    full_shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> np.ndarray:
    """
    Concatenate rank-ordered shards into the full tensor. Returns a new
    buffer — the caller cannot accidentally alias a peer's shard. Source
    shards are not modified.
    """
    dtype = np.dtype(dtype)
    world_size = len(shards)
    _check_world_size(world_size)
    numel = _check_shape(full_shape)
    expected = shard_sizes(numel, world_size)

    parts: list[np.ndarray] = []
    for rank, shard in enumerate(shards):
        arr = np.asarray(shard)
        if arr.ndim != 1:
            raise ValueError(f"rank {rank} shard must be 1-D, got {arr.ndim}-D")
        if arr.shape[0] != expected[rank]:
            raise ValueError(
                f"rank {rank} shard length {arr.shape[0]} != expected {expected[rank]}"
            )
        if arr.dtype != dtype:
            raise TypeError(f"rank {rank} shard dtype {arr.dtype} != {dtype}")
        parts.append(arr)

    flat = np.concatenate(parts) if parts else np.empty(0, dtype=dtype)
    if flat.shape[0] != numel:
        raise ValueError(f"gathered {flat.shape[0]} elements, expected {numel}")
    return flat.reshape(full_shape).copy()


def reduce_scatter_sum(
    contributions: Sequence[np.ndarray],
    full_shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> list[np.ndarray]:
    """
    Sum the full-shape contributions elementwise, then slice the result
    into N contiguous shards. Exactly one shard per rank; every element
    of the sum appears exactly once.

    This is SUM, not average. If you want the mean, use
    reduce_scatter_avg — do not just divide on the caller side, or you
    will eventually apply the factor twice.
    """
    dtype = np.dtype(dtype)
    world_size = len(contributions)
    _check_world_size(world_size)
    numel = _check_shape(full_shape)

    accum = np.zeros(numel, dtype=dtype)
    for rank, c in enumerate(contributions):
        arr = np.asarray(c)
        if arr.shape != full_shape:
            raise ValueError(
                f"rank {rank} contribution shape {arr.shape} != {full_shape}"
            )
        if arr.dtype != dtype:
            raise TypeError(f"rank {rank} contribution dtype {arr.dtype} != {dtype}")
        accum += arr.reshape(-1)

    offsets = shard_offsets(numel, world_size)
    return [accum[offsets[r] : offsets[r + 1]].copy() for r in range(world_size)]


def reduce_scatter_avg(
    contributions: Sequence[np.ndarray],
    full_shape: tuple[int, ...],
    dtype: np.dtype | type,
) -> list[np.ndarray]:
    """
    Reduce-Scatter with averaging. Requires a floating dtype: an integer
    dtype would truncate a non-integral average (e.g. (1 + 2) / 2 = 1.5
    becomes 1 in int32), which is a silent gradient-magnitude bug.
    """
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"reduce_scatter_avg requires floating dtype, got {dtype}")
    world_size = len(contributions)
    _check_world_size(world_size)
    shards = reduce_scatter_sum(contributions, full_shape, dtype)
    inv = 1.0 / float(world_size)
    return [(s * inv).astype(dtype, copy=False) for s in shards]


# ---------------------------------------------------------------------------
# FSDPModule: one simulated rank
# ---------------------------------------------------------------------------


class FSDPModule:
    """
    One rank's view of a sharded linear layer.

    Stores only `local_weight_shard` (and `local_bias_shard` if enabled).
    There is no hidden full copy. To compute a forward pass you must
    pass all `world_size` peer modules, rank-ordered, and `self` must be
    one of them — see `forward`.
    """

    def __init__(
        self,
        local_weight_shard: np.ndarray,
        full_weight_shape: tuple[int, int],
        rank: int,
        world_size: int,
        local_bias_shard: np.ndarray | None = None,
        full_bias_shape: tuple[int, ...] | None = None,
    ) -> None:
        _check_world_size(world_size)
        _check_rank(rank, world_size)

        if not isinstance(full_weight_shape, tuple) or len(full_weight_shape) != 2:
            raise ValueError(
                f"full_weight_shape must be a 2-tuple, got {full_weight_shape}"
            )
        self.full_weight_shape = (
            _check_dim(full_weight_shape[0], "full_weight_shape[0]"),
            _check_dim(full_weight_shape[1], "full_weight_shape[1]"),
        )
        self.rank = rank
        self.world_size = world_size

        arr = np.asarray(local_weight_shard)
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(
                f"local_weight_shard dtype must be floating, got {arr.dtype}"
            )
        self.dtype = arr.dtype

        expected = shard_sizes(
            self.full_weight_shape[0] * self.full_weight_shape[1], world_size
        )
        if arr.ndim != 1:
            raise ValueError(f"local_weight_shard must be 1-D, got {arr.ndim}-D")
        if arr.shape[0] != expected[rank]:
            raise ValueError(
                f"local_weight_shard length {arr.shape[0]} != "
                f"expected {expected[rank]} for rank {rank}"
            )
        self.local_weight_shard = arr.copy()

        if (local_bias_shard is None) != (full_bias_shape is None):
            raise ValueError(
                "local_bias_shard and full_bias_shape must both be given "
                "or both be None"
            )
        if local_bias_shard is not None:
            assert full_bias_shape is not None
            if len(full_bias_shape) != 1:
                raise ValueError(f"full_bias_shape must be 1-D, got {full_bias_shape}")
            b0 = _check_dim(full_bias_shape[0], "full_bias_shape[0]")
            self.full_bias_shape: tuple[int, ...] | None = (b0,)
            bias_expected = shard_sizes(b0, world_size)
            barr = np.asarray(local_bias_shard)
            if barr.ndim != 1:
                raise ValueError("local_bias_shard must be 1-D")
            if barr.shape[0] != bias_expected[rank]:
                raise ValueError(
                    f"local_bias_shard length {barr.shape[0]} != "
                    f"expected {bias_expected[rank]} for rank {rank}"
                )
            if barr.dtype != self.dtype:
                raise TypeError(f"bias dtype {barr.dtype} != weight dtype {self.dtype}")
            self.local_bias_shard: np.ndarray | None = barr.copy()
        else:
            self.full_bias_shape = None
            self.local_bias_shard = None

    @classmethod
    def from_full(
        cls,
        full_weights: np.ndarray,
        rank: int,
        world_size: int,
        bias: np.ndarray | None = None,
    ) -> FSDPModule:
        """
        Build one rank's module from a full weight matrix. Slices the
        flattened weight contiguously and copies the local shard in.
        The caller's array is never modified or retained.

        Non-contiguous inputs are materialized with ascontiguousarray
        first so the result is the same regardless of the input's
        strides.
        """
        full_weights = np.asarray(full_weights)
        if full_weights.ndim != 2:
            raise ValueError(f"full_weights must be 2-D, got {full_weights.ndim}-D")
        _check_world_size(world_size)
        _check_rank(rank, world_size)

        numel = full_weights.shape[0] * full_weights.shape[1]
        offsets = shard_offsets(numel, world_size)
        flat = np.ascontiguousarray(full_weights).reshape(-1)
        local = flat[offsets[rank] : offsets[rank + 1]].copy()

        local_bias: np.ndarray | None = None
        full_bias_shape: tuple[int, ...] | None = None
        if bias is not None:
            bias = np.asarray(bias)
            if bias.ndim != 1:
                raise ValueError("bias must be 1-D")
            boff = shard_offsets(bias.shape[0], world_size)
            local_bias = bias[boff[rank] : boff[rank + 1]].copy()
            full_bias_shape = bias.shape

        return cls(
            local_weight_shard=local,
            full_weight_shape=full_weights.shape,
            rank=rank,
            world_size=world_size,
            local_bias_shard=local_bias,
            full_bias_shape=full_bias_shape,
        )

    # --- peer validation ----------------------------------------------------

    def _check_peers(self, peers: Sequence[FSDPModule]) -> None:
        """
        Reject a peer list that doesn't match this module's mesh:
          - wrong length;
          - self not present (by identity, not equality);
          - ranks out of order;
          - peers disagree about world_size, shape, or dtype.

        The identity check catches a subtle failure: a caller who builds
        a substitute mesh with the same shapes and dtypes but different
        parameter values. Every other check would pass, and forward
        would quietly compute on the substitute weights.
        """
        if len(peers) != self.world_size:
            raise ValueError(f"expected {self.world_size} peers, got {len(peers)}")
        if not any(p is self for p in peers):
            raise ValueError(f"self (rank={self.rank}) not present in the peer list")
        ref = peers[0]
        for i, m in enumerate(peers):
            if m.rank != i:
                raise ValueError(f"peers[{i}].rank == {m.rank}, expected {i}")
            if m.world_size != self.world_size:
                raise ValueError(
                    f"peer {i} world_size={m.world_size} != {self.world_size}"
                )
            if m.full_weight_shape != ref.full_weight_shape:
                raise ValueError(
                    f"peer {i} shape {m.full_weight_shape} != {ref.full_weight_shape}"
                )
            if m.dtype != ref.dtype:
                raise TypeError(f"peer {i} dtype {m.dtype} != {ref.dtype}")
            if m.full_bias_shape != ref.full_bias_shape:
                raise ValueError(
                    f"peer {i} bias shape {m.full_bias_shape} != {ref.full_bias_shape}"
                )

    # --- collectives --------------------------------------------------------

    def all_gather_weights(self, peers: Sequence[FSDPModule]) -> np.ndarray:
        self._check_peers(peers)
        return all_gather(
            [m.local_weight_shard for m in peers],
            self.full_weight_shape,
            self.dtype,
        )

    def all_gather_bias(self, peers: Sequence[FSDPModule]) -> np.ndarray | None:
        if self.full_bias_shape is None:
            return None
        self._check_peers(peers)
        shards: list[np.ndarray] = []
        for i, m in enumerate(peers):
            if m.local_bias_shard is None:
                raise ValueError(f"peer {i} has no bias shard")
            shards.append(m.local_bias_shard)
        assert self.full_bias_shape is not None
        return all_gather(shards, self.full_bias_shape, self.dtype)

    # --- forward ------------------------------------------------------------

    def forward(self, X: np.ndarray, peers: Sequence[FSDPModule]) -> np.ndarray:
        """
        Y = X @ W_full (+ b_full). The full weight is gathered on demand
        from the peers, used, and dropped. Python does not promise to
        return the buffer to the OS immediately when the reference dies.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D")
        if X.shape[1] != self.full_weight_shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, module expects "
                f"{self.full_weight_shape[0]}"
            )
        if X.dtype != self.dtype:
            raise TypeError(f"X dtype {X.dtype} != module dtype {self.dtype}")

        W = self.all_gather_weights(peers)
        Y = X @ W
        b = self.all_gather_bias(peers)
        if b is not None:
            Y = Y + b
        return Y

    # --- gradient helpers ---------------------------------------------------

    @staticmethod
    def compute_weight_grad(X: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """dL/dW = X.T @ dY for Y = X @ W + b."""
        X = np.asarray(X)
        dY = np.asarray(dY)
        if X.ndim != 2 or dY.ndim != 2:
            raise ValueError(f"X and dY must be 2-D, got {X.ndim}-D and {dY.ndim}-D")
        if X.shape[0] != dY.shape[0]:
            raise ValueError(
                f"batch mismatch: X has {X.shape[0]} rows, dY has {dY.shape[0]}"
            )
        if X.dtype != dY.dtype:
            raise TypeError(f"X dtype {X.dtype} != dY dtype {dY.dtype}")
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError(f"X and dY must be floating, got {X.dtype}")
        return X.T @ dY

    @staticmethod
    def compute_bias_grad(dY: np.ndarray) -> np.ndarray:
        """dL/db = sum over the batch dimension of dY."""
        dY = np.asarray(dY)
        if dY.ndim != 2:
            raise ValueError(f"dY must be 2-D, got {dY.ndim}-D")
        if not np.issubdtype(dY.dtype, np.floating):
            raise TypeError(f"dY must be floating, got {dY.dtype}")
        return dY.sum(axis=0)


# ---------------------------------------------------------------------------
# FSDPOptimizer: sharded SGD
# ---------------------------------------------------------------------------


class FSDPOptimizer:
    """
    SGD bound to one FSDPModule. Updates only that module's shards.

    Two properties matter here:

    Atomicity. step() validates both gradients and both computed
    updates before touching any shard. If anything fails — wrong shape,
    wrong dtype, non-finite input, overflowing arithmetic — both shards
    are left exactly as they were. A training loop that catches and
    retries will not double-apply a weight update.

    No caller mutation. step() reads the caller's gradient arrays but
    never writes into them, so a caller can safely reuse an accumulation
    buffer across micro-batches.

    This optimizer has no per-parameter state. No momentum, no moments,
    no step counters. If you want to model what Adam would cost in
    memory, pass bytes_per_state to MemoryAccounting — that number is
    arithmetic, not an allocation.
    """

    def __init__(self, module: FSDPModule, lr: float) -> None:
        if isinstance(lr, bool) or not isinstance(lr, (int, float)):
            raise TypeError(f"lr must be a real number, got {type(lr).__name__}")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {lr}")
        self.module = module
        self.lr = float(lr)

    def step(
        self,
        weight_grad_shard: np.ndarray,
        bias_grad_shard: np.ndarray | None = None,
    ) -> None:
        # --- 1. Check the weight gradient ----------------------------------
        g = np.asarray(weight_grad_shard)
        w = self.module.local_weight_shard
        if g.shape != w.shape:
            raise ValueError(f"weight grad shape {g.shape} != param shape {w.shape}")
        if g.dtype != w.dtype:
            raise TypeError(f"weight grad dtype {g.dtype} != param dtype {w.dtype}")
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("weight grad contains NaN or inf")

        # --- 2. Check the bias gradient (if present) -----------------------
        bg: np.ndarray | None = None
        if bias_grad_shard is not None:
            b = self.module.local_bias_shard
            if b is None:
                raise ValueError("bias grad provided but module has no bias")
            bg = np.asarray(bias_grad_shard)
            if bg.shape != b.shape:
                raise ValueError(f"bias grad shape {bg.shape} != bias shape {b.shape}")
            if bg.dtype != b.dtype:
                raise TypeError(f"bias grad dtype {bg.dtype} != bias dtype {b.dtype}")
            if not np.all(np.isfinite(bg)):
                raise FloatingPointError("bias grad contains NaN or inf")

        # --- 3. Compute both proposed updates into fresh buffers -----------
        # errstate silences the RuntimeWarning numpy would emit on
        # overflow; we then check finiteness ourselves so the caller
        # gets a domain-level FloatingPointError, not a stray warning.
        with np.errstate(over="ignore", invalid="ignore"):
            delta_w = (g * self.lr).astype(w.dtype, copy=False)
            new_w = w - delta_w
        if not np.all(np.isfinite(new_w)):
            raise FloatingPointError(
                f"weight update overflowed shard dtype {w.dtype} "
                f"(lr={self.lr}); clip or rescale before step()"
            )

        new_b: np.ndarray | None = None
        if bg is not None:
            assert self.module.local_bias_shard is not None
            b = self.module.local_bias_shard
            with np.errstate(over="ignore", invalid="ignore"):
                delta_b = (bg * self.lr).astype(b.dtype, copy=False)
                new_b = b - delta_b
            if not np.all(np.isfinite(new_b)):
                raise FloatingPointError(
                    f"bias update overflowed shard dtype {b.dtype}"
                )

        # --- 4. Commit. Everything validated, so this cannot half-apply. ---
        self.module.local_weight_shard[:] = new_w
        if new_b is not None:
            assert self.module.local_bias_shard is not None
            self.module.local_bias_shard[:] = new_b


# ---------------------------------------------------------------------------
# Memory accounting (arithmetic, not measurement)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryAccounting:
    """
    Compute what sharded training would cost in bytes, given the byte
    widths you pass in. This class allocates nothing and measures
    nothing.

    The per-rank estimate uses the LARGEST shard size, not the average.
    When P is not divisible by N, the biggest rank holds ⌈P/N⌉ elements
    — using ⌊P/N⌋ would understate its peak.

    bytes_per_state is a caller-supplied number. This module's SGD has
    no per-parameter state; if you pass 8 for Adam-style m and v, you
    are asking "what would Adam cost", not "what is allocated now".

    Excludes: activations, output buffers, communication buffers,
    allocator fragmentation, framework metadata, and the temporary full
    weight materialized during forward (reported separately by
    transient_full_gather).
    """

    num_params: int
    world_size: int
    bytes_per_param: int
    bytes_per_grad: int
    bytes_per_state: int

    def __post_init__(self) -> None:
        _check_world_size(self.world_size)
        for name, val in (
            ("num_params", self.num_params),
            ("bytes_per_param", self.bytes_per_param),
            ("bytes_per_grad", self.bytes_per_grad),
            ("bytes_per_state", self.bytes_per_state),
        ):
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    def persistent_per_rank(self) -> dict[str, int]:
        max_shard = max(shard_sizes(self.num_params, self.world_size))
        return {
            "param_shard": max_shard * self.bytes_per_param,
            "grad_shard": max_shard * self.bytes_per_grad,
            "state_shard": max_shard * self.bytes_per_state,
        }

    def persistent_total_per_rank(self) -> int:
        return sum(self.persistent_per_rank().values())

    def transient_full_gather(self) -> int:
        return self.num_params * self.bytes_per_param

    def standard_dp_per_rank(self) -> int:
        return self.num_params * (
            self.bytes_per_param + self.bytes_per_grad + self.bytes_per_state
        )
