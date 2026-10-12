"""
day48_ring_allreduce.py
=======================

Ring-AllReduce: a single-process simulation.

What it does
------------
Takes N gradient vectors (one per simulated worker) and returns their
elementwise mean:

    g_sync = (1/N) * sum_i g_i

How it does it
--------------
Two phases, each N-1 rounds around a logical ring:

    rank:  0 ──► 1 ──► 2 ──► 0    (sends go from w to w+1)

    Reduce-scatter, round r (0..N-2):
        w sends chunk (w - r) to w+1
        w receives chunk (w - r - 1) from w-1 and adds it to its own
        After N-1 rounds, rank w owns chunk (w + 1) fully reduced.

    All-gather, round r (0..N-2):
        w sends chunk (w - r + 1) to w+1
        w receives chunk (w - r) from w-1 and copies it (no addition)
        After N-1 rounds, every rank holds every reduced chunk.

    Chunks: array_split-style boundaries. The first M % N chunks get
    one extra element. M < N is allowed (some chunks are empty).

Cost
----
Simulation: O(N*M) array work + O(N^2) list bookkeeping. Peak memory
is O(N*M) because all N buffers live in one process (a real ring uses
O(M) per worker).

Real ring: per-worker volume 2(N-1)/N * M elements; time under the
latency-bandwidth model is 2(N-1)*alpha + 2(N-1)/N * M * beta. This
module implements the schedule that model describes; it does not
implement the model.

This is NOT
-----------
A real distributed system. No sockets, no NCCL, no CUDA, no MPI, no
processes, no network. It's a single-process simulation of the ring
schedule.

Reproducibility
---------------
Fully deterministic. No randomness, no seeds, no reliance on hash
iteration order. Cross-NumPy-version bit-equality is not claimed:
SIMD lowering can change intermediate rounding inside one add.

Numerical notes
---------------
* float64 accumulation. float32 would lose small offsets on gradients
  near 1e8, where float32's ULP is 8.
* Overflow: if partial sums overflow to inf, we raise
  FloatingPointError. A cancellation-first addition order could avoid
  some overflows (e.g. [1e308, 1e308, -1e308]); this module implements
  one specific ring order, so it raises rather than silently returning
  inf. See TEST_STRATEGY.md for the full note.
* Integer inputs above 2**53 are silently rounded by astype(float64).

Design note: an earlier revision exposed a per-round `last_trace`.
It was removed as dead weight; the schedule is verified through the
returned mean. If the protocol itself needs to be inspected, re-add
the trace or read TEST_STRATEGY.md §3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

logger = logging.getLogger(__name__)

DTYPE = np.dtype(np.float64)
ChunkBounds = list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_world_size(n: object) -> int:
    """Reject bool, non-integers, and values < 1."""
    if isinstance(n, bool):  # bool is an int subclass; reject explicitly
        raise TypeError(f"world_size must be an int, not bool; got {n!r}")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"world_size must be an int; got {type(n).__name__}")
    n = int(n)
    if n < 1:
        raise ValueError(f"world_size must be >= 1; got {n}")
    return n


def _as_vector(x: object, name: str) -> np.ndarray:
    """
    Return a fresh 1-D float64 copy of `x`.

    Rejects non-1-D inputs (no silent flattening), non-numeric or
    complex dtypes, and non-finite values. Integer inputs above 2**53
    are rounded by astype; the caller is expected to know this.
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector; got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must have a real numeric dtype; got {arr.dtype}")
    out = arr.astype(DTYPE, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or infinite entries")
    return out


def _chunk_bounds(m: int, n: int) -> ChunkBounds:
    """
    Contiguous (start, stop) chunks covering [0, m). The first m % n
    chunks get one extra element, matching np.array_split. Returned as
    explicit bounds so every worker uses identical boundaries and the
    schedule is auditable without running the code.
    """
    base, rem = divmod(m, n)
    bounds: ChunkBounds = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RingAllReduceWorker:
    """
    A simulated ring participant. Holds a rank, the shared world size,
    and a private copy of the worker's input gradient. The buffer is
    replaced on every set, so repeated calls cannot reuse stale state.
    """

    __slots__ = ("local_gradient", "rank", "world_size")

    def __init__(self, rank: int, world_size: int) -> None:
        self.world_size = _validate_world_size(world_size)
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
            raise TypeError(f"rank must be an int; got {rank!r}")
        rank = int(rank)
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}); got {rank}")
        self.rank = rank
        self.local_gradient: np.ndarray | None = None

    def set_local_gradient(self, grad: object) -> None:
        self.local_gradient = _as_vector(grad, f"gradient for rank {self.rank}")


class DistributedDataParallelEngine:
    """
    Orchestrates the two-phase Ring-AllReduce schedule across
    `world_size` simulated workers.

    `ring_allreduce(gradients)` accepts any iterable of exactly
    `world_size` equal-shaped 1-D finite arrays and returns their
    elementwise mean as a float64 1-D array of length M. Empty vectors
    are supported.

    Raises ValueError or TypeError on bad inputs, and FloatingPointError
    if the reduction overflows to inf/NaN. Deterministic.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = _validate_world_size(world_size)
        self.workers = [
            RingAllReduceWorker(rank=i, world_size=self.world_size)
            for i in range(self.world_size)
        ]

    def ring_allreduce(self, gradients: Iterable[np.ndarray]) -> np.ndarray:
        prepared = self._validate_inputs(gradients)
        n = self.world_size
        m = int(prepared[0].shape[0])
        logger.debug("ring_allreduce: N=%d M=%d", n, m)

        bounds = _chunk_bounds(m, n)

        # chunks[w][c] is rank w's current copy of chunk c. Slices are
        # copied so the caller's arrays stay untouched, and so slot
        # reassignment never writes back into worker.local_gradient.
        chunks = [[g[s:e].copy() for s, e in bounds] for g in prepared]

        # Suppress the overflow warning during accumulation; the final
        # isfinite check below raises FloatingPointError instead, which
        # is the documented contract.
        with np.errstate(over="ignore"):
            self._reduce_scatter(chunks, n)
            self._all_gather(chunks, n)
            summed = np.concatenate(chunks[0])
            result = summed / DTYPE.type(n)

        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                f"ring_allreduce overflowed in dtype {DTYPE}. Every input "
                "was finite, but a partial sum exceeded the dtype's range. "
                "Rescale the inputs or use a numerically robust reduction."
            )
        return result

    # -- internals ---------------------------------------------------------

    def _validate_inputs(self, gradients: Iterable[np.ndarray]) -> list[np.ndarray]:
        """
        Check the caller's arguments, install validated copies into
        worker buffers, and return the prepared list.
        """
        if isinstance(gradients, np.ndarray):
            raise TypeError(
                "gradients must be an iterable of 1-D arrays, not a single "
                "ndarray; wrap each worker's gradient in a list"
            )
        try:
            grads = list(gradients)
        except TypeError as exc:
            raise TypeError("gradients must be an iterable of 1-D arrays") from exc

        n = self.world_size
        if len(grads) != n:
            raise ValueError(
                f"ring_allreduce expects exactly world_size={n} gradient "
                f"vectors; got {len(grads)}"
            )

        prepared = [_as_vector(g, f"gradients[{i}]") for i, g in enumerate(grads)]
        shape = prepared[0].shape
        for i, g in enumerate(prepared):
            if g.shape != shape:
                raise ValueError(
                    f"gradients[{i}] has shape {g.shape}, expected {shape}"
                )

        for worker, g in zip(self.workers, prepared):
            worker.set_local_gradient(g)
        return prepared

    @staticmethod
    def _reduce_scatter(chunks: list[list[np.ndarray]], n: int) -> None:
        """
        Phase 1: N-1 rounds. After round r, rank w owns chunk
        (w - k) reduced over {w, w-1, ..., w-k} for each k <= r+1.
        After round N-2, rank w owns chunk (w+1) fully reduced.

        Safe to write in place: worker w writes slot (w - r - 1) of its
        own table, and its only reader (worker w+1) reads slot
        (w - r) of that table in the same round. Different slots.
        """
        for r in range(n - 1):
            for w in range(n):
                prev = (w - 1) % n
                c = (w - r - 1) % n
                chunks[w][c] = chunks[w][c] + chunks[prev][c]

    @staticmethod
    def _all_gather(chunks: list[list[np.ndarray]], n: int) -> None:
        """
        Phase 2: N-1 rounds. After round r, rank w holds fully-reduced
        chunks (w + 1 - k) for each k <= r+1. After round N-2, every
        rank holds every reduced chunk.

        `.copy()` is required: without it, worker w and worker prev
        would share the same array object, and a future per-worker
        post-process would corrupt every sharer at once.
        """
        for r in range(n - 1):
            for w in range(n):
                prev = (w - 1) % n
                c = (w - r) % n
                chunks[w][c] = chunks[prev][c].copy()


__all__ = ["DistributedDataParallelEngine", "RingAllReduceWorker"]
