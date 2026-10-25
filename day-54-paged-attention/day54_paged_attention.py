"""Paged KV cache with single-head attention.

In autoregressive inference every generated token needs the keys and values
of every previous token -- the KV cache. A naive cache reserves one
contiguous buffer per sequence, sized for the worst-case length, which
wastes memory on short sequences and fragments the allocator.
PagedAttention borrows the OS page-table trick: K/V live in fixed-size
physical blocks, each sequence gets a page table mapping logical blocks to
physical ones, and a sequence's tokens can sit anywhere in the pool.

This module implements that addressing scheme -- the block allocator, the
page table, and a single-head attention read path -- as a NumPy simulation.
It is *not* a GPU kernel or a production inference engine.

Layout
------
key_pool, value_pool : float32, shape (num_blocks, block_size, d_model)
page_tables          : {seq_id: [physical block ids, in logical order]}
seq_lengths          : {seq_id: number of valid tokens written so far}

Token t of a sequence lives at:

    physical_block = page_tables[seq_id][t // block_size]
    slot           = t % block_size

Valid prefix
------------
seq_lengths[seq_id] is the length of the *contiguous* prefix that has been
written. Reads are bounded by it. Writes may overwrite existing positions
or extend at the current end; they may not leave a gap. That's the
standard autoregressive contract, and it keeps the read path from handing
back zero-filled slots as if they were real tokens.

Corruption
----------
page_tables is public -- useful for teaching, and needed by the redirect
and corruption tests. Direct mutation is unsupported, but every read and
write re-checks the table it uses: entries must be non-negative integers
in range, the blocks must still be owned by this sequence, and the table
must still cover the recorded valid length. A corrupted entry raises
rather than quietly reading the wrong block. release_sequence additionally
requires unique block ids, so a table like [1, 1] can't push the same block
onto the free list twice.

Costs
-----
allocate_sequence    O(A), A = ceil(tokens / block_size)
release_sequence     O(A) plus zeroing O(A * block_size * d_model)
write / read         O(n * d_model + A), dominated by the gather/scatter
attention (1 query)  O(T * d_model) for QK^T and the V sum, O(T) for softmax
                     (page-table indirection doesn't make attention cheaper)

Out of scope
------------
No GPU, no fused kernel, no zero-copy: attention reconstructs contiguous
K/V arrays and runs a dense matmul. No layer/head axes, no batching, no
concurrency, no prefix sharing, no eviction, no live growth. No performance
is measured here -- the "~4% waste / 4x concurrency" figures from the
literature are not produced by this code.

Reproducibility
---------------
No RNG. The free list is a FIFO deque starting at 0..num_blocks-1, so block
assignment is a pure function of the call sequence. Everything is float32,
including the softmax scale and epsilon, so the result stays float32 on
NumPy < 2.0 (where dividing by a Python float would upcast).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

__all__ = ["PagedKVCache", "paged_attention_forward"]

_DTYPE = np.float32
_EPS = np.float32(1e-8)  # keep float32 on NumPy < 2.0


def _require_int(name: str, value, minimum: int) -> int:
    """Return value as an int >= minimum, or raise TypeError / ValueError.

    Booleans are rejected even though bool is a subclass of int -- passing
    True where a dimension is expected is almost always a typo, not intent.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _as_token_matrix(name: str, arr, d_model: int) -> np.ndarray:
    """Validate a (n_tokens, d_model) finite array and return a float32 copy.

    The copy protects the cache from later mutation of the caller's buffer,
    and lets the finiteness check run on the values that will actually be
    stored (a float64 1e300 becomes inf as float32).
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D of shape (n_tokens, d_model); got shape {a.shape}"
        )
    if a.shape[1] != d_model:
        raise ValueError(f"{name} must have width d_model={d_model}; got {a.shape[1]}")
    with np.errstate(over="ignore", invalid="ignore"):
        a32 = a.astype(_DTYPE, copy=True)
    if not np.all(np.isfinite(a32)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a32


class PagedKVCache:
    """Block pool + per-sequence page tables.

    Blocks are exclusively owned: a physical block belongs to at most one
    live sequence. Sharing would need reference counting, which this cache
    doesn't implement. Allocation is atomic -- either every required block
    is assigned and the page table published, or nothing changes.

    Errors:
      TypeError  -- non-integer where an int is expected (including bool),
                    or a non-integer physical block id in a page table.
      ValueError -- bad dimensions/shapes, non-finite values, writes that
                    exceed capacity or leave a gap, reads past the valid
                    length, duplicate seq_id, corrupted page tables.
      KeyError   -- unknown seq_id.
      RuntimeError -- allocation can't be satisfied; state is unchanged.
    """

    def __init__(self, num_blocks: int, block_size: int, d_model: int) -> None:
        self._num_blocks = _require_int("num_blocks", num_blocks, 1)
        self._block_size = _require_int("block_size", block_size, 1)
        self._d_model = _require_int("d_model", d_model, 1)

        self.key_pool: np.ndarray = np.zeros(
            (self._num_blocks, self._block_size, self._d_model), dtype=_DTYPE
        )
        self.value_pool: np.ndarray = np.zeros_like(self.key_pool)

        # Public for teaching. Direct mutation is unsupported, but
        # _validated_pages runs on every read/write so corruption is caught.
        self.page_tables: dict[int, list[int]] = {}
        self.seq_lengths: dict[int, int] = {}

        self._block_owner: np.ndarray = np.full(self._num_blocks, -1, dtype=np.int64)
        # deque over list: list.pop(0) is O(n), so allocating A blocks from
        # an M-block list would cost O(A*M) per call. Matters once the pool
        # holds thousands of blocks and a block is allocated per decode step.
        self._free_blocks: deque[int] = deque(range(self._num_blocks))

    # ---- introspection ----------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def dtype(self) -> np.dtype:
        return self.key_pool.dtype

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def _pages_for(self, seq_id: int) -> list[int]:
        """Return the page table for seq_id, or raise KeyError."""
        sid = _require_int("seq_id", seq_id, 0)
        if sid not in self.page_tables:
            raise KeyError(f"sequence {sid} is not allocated")
        return self.page_tables[sid]

    def _validated_pages(self, seq_id: int) -> list[int]:
        """Return the page table after checking capacity, bounds, and ownership.

        Three checks, all before the caller may index into the pool:
          1. capacity  -- len(pages) * block_size >= valid_length
          2. bounds    -- every entry is a non-negative int < num_blocks
          3. ownership -- every entry's block is currently owned by seq_id

        Uniqueness is *not* checked here: a redirect like [1, 1] is a valid
        test scaffold for the read path. It's checked in release_sequence,
        where duplicates would corrupt the free list. Cost: O(A).
        """
        pages = self._pages_for(seq_id)
        valid_length = int(self.seq_lengths[seq_id])
        token_capacity = len(pages) * self._block_size
        if token_capacity < valid_length:
            raise ValueError(
                f"page table for sequence {seq_id} has {len(pages)} block(s) "
                f"({token_capacity} token slots) but the valid length is "
                f"{valid_length}: the table was shortened below the recorded "
                f"prefix"
            )
        for block_id in pages:
            if not isinstance(block_id, (int, np.integer)) or isinstance(
                block_id, (bool, np.bool_)
            ):
                raise TypeError(
                    f"page table for sequence {seq_id} contains a non-integer "
                    f"physical block id {block_id!r}"
                )
            block_index = int(block_id)
            if block_index < 0 or block_index >= self._num_blocks:
                raise ValueError(
                    f"page table for sequence {seq_id} contains out-of-range "
                    f"physical block id {block_index} (valid range "
                    f"[0, {self._num_blocks}))"
                )
            owner = int(self._block_owner[block_index])
            if owner != seq_id:
                raise ValueError(
                    f"page table for sequence {seq_id} references physical "
                    f"block {block_index} which is owned by "
                    f"{owner if owner >= 0 else 'no one'}"
                )
        return pages

    def owned_blocks(self, seq_id: int) -> list[int]:
        """Copy of the sequence's page table (logical -> physical)."""
        return list(self._pages_for(seq_id))

    def sequence_length(self, seq_id: int) -> int:
        """Number of tokens actually written; reads are bounded by this."""
        self._pages_for(seq_id)
        return int(self.seq_lengths[seq_id])

    def allocated_capacity(self, seq_id: int) -> int:
        """Token slots reserved for the sequence (blocks_owned * block_size)."""
        return len(self._pages_for(seq_id)) * self._block_size

    def block_owner(self, block_id: int) -> int | None:
        """Return the owning seq_id, or None if the block is free."""
        b = _require_int("block_id", block_id, 0)
        if b >= self._num_blocks:
            raise ValueError(f"block_id must be < {self._num_blocks}, got {b}")
        owner = int(self._block_owner[b])
        return None if owner < 0 else owner

    def memory_stats(self) -> dict[str, int | str]:
        """Sizes derived from the declared arrays' shape and dtype.

        Includes only key_pool + value_pool payload. Excludes page-table
        lists, _block_owner, the free-list deque, attention temporaries,
        queries, outputs. This is a lower bound on real memory, not a
        measurement.
        """
        return {
            "num_blocks": self._num_blocks,
            "block_size": self._block_size,
            "d_model": self._d_model,
            "dtype": self.key_pool.dtype.name,
            "kv_pool_bytes": int(self.key_pool.nbytes + self.value_pool.nbytes),
            "blocks_free": self.free_block_count,
            "blocks_allocated": self._num_blocks - self.free_block_count,
        }

    # ---- allocation -------------------------------------------------------

    def allocate_sequence(self, seq_id: int, initial_tokens: int = 0) -> None:
        """Reserve ceil(initial_tokens / block_size) blocks for seq_id.

        initial_tokens=0 is legal and allocates nothing. Capacity is fixed
        here; this cache doesn't grow a live sequence. Ceiling division
        means an exact multiple of block_size doesn't waste an extra block.
        """
        sid = _require_int("seq_id", seq_id, 0)
        token_count = _require_int("initial_tokens", initial_tokens, 0)
        if sid in self.page_tables:
            raise ValueError(f"sequence {sid} is already allocated")

        blocks_needed = (token_count + self._block_size - 1) // self._block_size

        # Atomicity: check capacity *before* popping the free list, so a
        # failed allocation can't leak or partially assign blocks.
        if blocks_needed > len(self._free_blocks):
            raise RuntimeError(
                f"cannot allocate {blocks_needed} block(s) for sequence {sid}: "
                f"only {len(self._free_blocks)} free of {self._num_blocks}"
            )

        physical_blocks = [self._free_blocks.popleft() for _ in range(blocks_needed)]
        for block_id in physical_blocks:
            self._block_owner[block_id] = sid
        self.page_tables[sid] = physical_blocks
        self.seq_lengths[sid] = 0

    def release_sequence(self, seq_id: int) -> None:
        """Free every block of seq_id and zero its payload.

        Zeroing costs O(A * block_size * d_model) but means a recycled block
        never exposes the previous tenant's values. Capacity, ownership,
        and uniqueness are checked before any state changes -- a corrupted
        table like [1, 1] refuses the release rather than pushing the same
        block onto the free list twice.
        """
        pages = self._validated_pages(seq_id)
        if len(set(pages)) != len(pages):
            raise ValueError(
                f"page table for sequence {seq_id} contains duplicate "
                f"physical block ids: {pages}. Refusing to release -- "
                f"duplicates would push the same block onto the free list "
                f"more than once and corrupt later allocations."
            )
        for block_id in pages:
            self.key_pool[block_id].fill(0.0)
            self.value_pool[block_id].fill(0.0)
            self._block_owner[block_id] = -1
            self._free_blocks.append(block_id)
        del self.page_tables[seq_id]
        del self.seq_lengths[seq_id]

    # ---- data movement ----------------------------------------------------

    def write_tokens(
        self,
        seq_id: int,
        start_idx: int,
        k_tokens,
        v_tokens,
    ) -> None:
        """Scatter n tokens at logical positions [start_idx, start_idx + n).

        Writes may overwrite any already-valid position or extend at the
        current end. They may not leave a gap (start_idx > valid_length)
        and may not exceed the allocated capacity. Both rules are checked
        before the pool is touched, so a rejected write leaves the cache
        unchanged.

        A zero-token write (n == 0) is a no-op regardless of start_idx; the
        gap check is skipped because no position is marked valid by a
        zero-length write.
        """
        pages = self._validated_pages(seq_id)
        start_position = _require_int("start_idx", start_idx, 0)

        keys = _as_token_matrix("k_tokens", k_tokens, self._d_model)
        values = _as_token_matrix("v_tokens", v_tokens, self._d_model)
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"k_tokens and v_tokens must have the same length; "
                f"got {keys.shape[0]} and {values.shape[0]}"
            )
        num_new_tokens = keys.shape[0]
        token_capacity = len(pages) * self._block_size
        if start_position + num_new_tokens > token_capacity:
            raise ValueError(
                f"write of {num_new_tokens} token(s) at logical index "
                f"{start_position} exceeds allocated capacity {token_capacity} "
                f"for sequence {seq_id}"
            )
        current_valid_length = int(self.seq_lengths[seq_id])
        if num_new_tokens > 0 and start_position > current_valid_length:
            # Gap: without this check, positions [valid, start) would be
            # marked valid and reads would return zero-filled "tokens".
            raise ValueError(
                f"write at logical index {start_position} would create a gap: "
                f"sequence {seq_id} currently has valid length "
                f"{current_valid_length}"
            )
        if num_new_tokens == 0:
            return

        # One scatter per pool; no per-token loop. Logical positions are
        # unique, so the (block, slot) pairs are too -- order doesn't matter.
        logical_positions = np.arange(
            start_position, start_position + num_new_tokens, dtype=np.int64
        )
        block_ids = np.asarray(pages, dtype=np.int64)[
            logical_positions // self._block_size
        ]
        slot_offsets = logical_positions % self._block_size
        self.key_pool[block_ids, slot_offsets] = keys
        self.value_pool[block_ids, slot_offsets] = values
        self.seq_lengths[seq_id] = max(
            current_valid_length, start_position + num_new_tokens
        )

    def read_sequence(
        self, seq_id: int, num_tokens: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather the first num_tokens valid tokens in logical order.

        num_tokens=None means the full valid length. Requesting more than
        the valid length raises -- unwritten slots are never returned as
        tokens. Returned arrays are fresh float32 copies, not views into
        the pool.
        """
        pages = self._validated_pages(seq_id)
        valid_length = int(self.seq_lengths[seq_id])
        requested_count = (
            valid_length
            if num_tokens is None
            else _require_int("num_tokens", num_tokens, 0)
        )
        if requested_count > valid_length:
            raise ValueError(
                f"sequence {seq_id} has only {valid_length} valid token(s); "
                f"requested {requested_count}"
            )
        if requested_count == 0:
            return (
                np.zeros((0, self._d_model), dtype=_DTYPE),
                np.zeros((0, self._d_model), dtype=_DTYPE),
            )

        logical_positions = np.arange(requested_count, dtype=np.int64)
        block_ids = np.asarray(pages, dtype=np.int64)[
            logical_positions // self._block_size
        ]
        slot_offsets = logical_positions % self._block_size
        # Fancy indexing already returns new arrays; the .copy() is a
        # contract statement: callers may mutate what they get back.
        return (
            self.key_pool[block_ids, slot_offsets].copy(),
            self.value_pool[block_ids, slot_offsets].copy(),
        )


def paged_attention_forward(
    Q,
    cache: PagedKVCache,
    seq_id: int,
    seq_len: int | None = None,
) -> np.ndarray:
    """Single-head attention of one query over a sequence's cached K/V.

    scores = Q @ K.T / sqrt(d_model), stable softmax over the T keys,
    out = weights @ V. K and V come from the cache through its page table
    -- there is no hidden dense copy of the source arrays.

    seq_len defaults to the valid length and must be in [1, valid].
    Attention over zero tokens is undefined and is rejected.

    Stability: scores are max-subtracted before exp; the denominator is
    clamped at 1e-8 (unreachable for finite scores). Non-finite scores or
    outputs raise FloatingPointError. Everything is float32 -- the scale
    factor and epsilon are explicitly np.float32 so the result stays
    float32 on NumPy < 2.0 (where dividing by a Python float would
    upcast). Float32 overflow on the matmul paths is silenced locally with
    np.errstate; the isfinite check that follows is the actual guard.

    Cost: this reconstructs contiguous (T, d_model) K/V copies and runs
    dense matmuls. Temporaries include K/V (2*T*d_model*4 bytes),
    scores/weights (~4*T*4 bytes), and the block-index arrays
    (T*8 bytes each). Arithmetic is O(T*d_model), the same as plain dense
    attention -- the page table changes *where* the data lives, not the
    amount of work.
    """
    if not isinstance(cache, PagedKVCache):
        raise TypeError("cache must be a PagedKVCache instance")

    valid_length = cache.sequence_length(seq_id)  # raises KeyError if unknown

    query = _as_token_matrix("Q", Q, cache.d_model)
    if query.shape[0] != 1:
        raise ValueError(f"Q must have shape (1, d_model); got {query.shape}")

    attention_length = (
        valid_length if seq_len is None else _require_int("seq_len", seq_len, 0)
    )
    if attention_length > valid_length:
        raise ValueError(
            f"seq_len={attention_length} exceeds the {valid_length} valid "
            f"cached token(s) of sequence {seq_id}"
        )
    if attention_length < 1:
        raise ValueError(
            "attention over zero tokens is undefined; seq_len must be >= 1"
        )

    K, V = cache.read_sequence(seq_id, attention_length)

    # Explicit np.float32: prevents float32 -> float64 promotion on
    # NumPy < 2.0. No-op under NEP 50.
    scale_factor = np.float32(1.0 / math.sqrt(cache.d_model))
    # Finite Q/K can still overflow float32 in the dot product. That's
    # expected to be caught by the isfinite check below, not by NumPy's
    # warning machinery; errstate silences the warning.
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (query @ K.T) * scale_factor
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("attention scores overflowed float32")

    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(shifted_scores)
    normalizer = np.maximum(np.sum(attention_weights, axis=-1, keepdims=True), _EPS)
    attention_weights = attention_weights / normalizer

    with np.errstate(over="ignore", invalid="ignore"):
        attention_output = attention_weights @ V
    if not np.all(np.isfinite(attention_output)):
        raise FloatingPointError("attention output is not finite")
    return attention_output
