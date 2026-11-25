"""Day 63 — Disaggregated KV-cache cluster simulation (in-process, deterministic).

What this models
----------------
The control plane of a prefill/decode split cluster:
- Requests are routed round-robin (sticky per request) to disjoint node pools.
- Prefill nodes hold K/V blocks; writes snapshot the caller's arrays.
- Transfers validate block identity and completeness before publishing a
  copied sequence to a decode node with a single dict assignment.
- Reads return the requested token prefix from private copies.

This is NOT a distributed system
--------------------------------
Storage is an in-process Python dict of NumPy arrays — not remote memory.
No sockets, no RDMA / InfiniBand / RoCE, no GPU kernels, no network stack,
no scheduler, no retries, no queueing, no clock. "Transfer" is a deep copy.
The word "zero-copy" does not apply.

Not claimed
-----------
- RDMA or any network transport
- Zero-copy transfer
- 2x–3x cluster throughput uplift (requires benchmarks)
- TTFT improvement (requires end-to-end timing)
- GPU utilisation improvement (requires profiling)
- Network saturation prevention (no network model)
Disaggregation is a trade-off: for small prompts, short generations, or
congested links, transfer overhead can exceed the specialisation benefit.

Atomicity
---------
Validation and snapshot both complete before ``_install`` publishes the entry
via a single dict assignment. A validation failure, a missing block, or a
snapshot exception therefore leaves the destination unchanged: no partial
sequence is ever visible to a reader. Snapshot mid-copy failure is tested via
monkeypatch; ``_install`` itself is one assignment and cannot partially fail.

Complexity (success and failure paths)
--------------------------------------
route_prefill / route_decode:
    O(1) expected in pool size (dict get + list index + insert). First-time
    hashing of a new seq_id costs O(|seq_id|); hash worst-case is O(n). The
    claim is *expected* O(1), not worst-case.

write_prefill_blocks:
    O(S) time, O(S) storage, S = payload bytes.

transfer_kv_cache:
    Success path: O(B + S) expected — set membership O(B), snapshot O(S),
    frozenset pool membership O(1) expected.
    Failure path: O(B log B) when the block set is wrong — the error message
    sorts set differences. Diagnostic cost only.

read_sequence:
    O(S_prefix) payload work; only touched blocks are copied.

Read authorization
------------------
There is no caller identity in this simulator. ``read_sequence`` trusts the
``seq_id`` supplied by the caller; a caller that knows a sequence stored on a
node can read it. ``BlockId`` is a *validation* token (fields must match
source), not an *authentication* token (proving the block was actually minted
by ``write_prefill_blocks``). If request isolation is required, wrap the
manager in an authorization layer.

Node immutability
-----------------
Node configuration (``node_id``, ``role``, ``block_size``, ``d_model``) is
read-only after construction. Enforced via read-only properties and
``__slots__``. Mutating ``block_size`` after a write would make subsequent
reads interpret stored blocks with the wrong geometry; mutating ``node_id``
would invalidate previously issued ``BlockId``s; mutating ``role`` would
bypass the write guard.

Reproducibility
---------------
Deterministic: no RNG, no clock, no environment reads. Block iteration is
by ascending ``block_index`` (never ``set`` iteration order). No seed
parameter needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BlockId",
    "DisaggregatedClusterManager",
    "RemoteKVNode",
    "ideal_transfer_seconds",
    "kv_cache_bytes",
]

# Role labels. Used to distinguish the two disjoint node pools and to gate
# write_prefill_blocks so a decode node cannot bypass the transfer step.
PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"

# Accepted NumPy dtype kinds for KV payloads:
#   "f" = floating-point, "i" = signed integer, "u" = unsigned integer.
# Excluded: "b" (bool), "O" (object), "U"/"S" (string), "c" (complex).
SUPPORTED_DTYPE_KINDS = frozenset({"f", "i", "u"})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_integer_at_least(value: object, parameter_name: str, minimum: int) -> int:
    """Reject non-int (and bool, which is an int subclass) and enforce a lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{parameter_name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{parameter_name} must be >= {minimum}, got {value}")
    return int(value)


def _validate_positive_integer(value: object, parameter_name: str) -> int:
    return _validate_integer_at_least(value, parameter_name, 1)


def _validate_sequence_id(seq_id: object) -> str:
    """A sequence id must be a non-empty string.

    The empty string would silently share routing state between distinct
    requests, which almost always indicates a caller bug that would otherwise
    surface much later as a corrupted cache entry.
    """
    if not isinstance(seq_id, str):
        raise TypeError(f"seq_id must be a str, got {type(seq_id).__name__}")
    if not seq_id:
        raise ValueError("seq_id must be a non-empty string")
    return seq_id


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockId:
    """Identity of one KV block: ``(seq_id, node_id, block_index)``.

    A *validation* token, not an *authentication* token: the triple pins
    ownership at transfer-validation time (fields must match the source node
    and sequence), but a caller who forges a consistent triple has already
    guessed the sequence identity — there is no auth boundary in this
    simulator. Do not use as a security primitive.

    A bare ``int`` index would be guessable and would let a caller move another
    request's blocks; the triple pins ownership at validation time.
    """

    seq_id: str
    node_id: int
    block_index: int


@dataclass(frozen=True)
class _SequenceEntry:
    """Immutable metadata record holding privately owned K/V block arrays.

    Frozen guards attribute rebinding. The NumPy arrays themselves are
    mutable; external mutation is prevented by construction — the node is
    the only holder, writes snapshot caller arrays, and reads return copies.
    Do not describe this as an "immutable tensor"; it is an immutable
    metadata record whose arrays are privately owned.
    """

    k_blocks: tuple[np.ndarray, ...]
    v_blocks: tuple[np.ndarray, ...]
    dtype: np.dtype
    token_count: int


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class RemoteKVNode:
    """A single simulated prefill or decode node.

    Configuration is immutable after construction. Partial final blocks are
    supported and keep their exact token count (never padded). Writes
    snapshot the caller's arrays; reads return private copies.
    ``write_prefill_blocks`` is prefill-only.

    Role enforcement: ``write_prefill_blocks`` is a prefill-only operation.
    Calling it on a decode node is a ``ValueError`` so callers cannot bypass
    the prefill→decode transfer boundary.
    """

    __slots__ = ("_block_size", "_d_model", "_node_id", "_role", "_sequences")

    def __init__(
        self,
        node_id: int,
        role: str,
        block_size: int,
        d_model: int,
    ) -> None:
        if role not in (PREFILL_ROLE, DECODE_ROLE):
            raise ValueError(
                f"role must be {PREFILL_ROLE!r} or {DECODE_ROLE!r}, got {role!r}"
            )
        self._node_id = _validate_integer_at_least(node_id, "node_id", 0)
        self._role = role
        self._block_size = _validate_positive_integer(block_size, "block_size")
        self._d_model = _validate_positive_integer(d_model, "d_model")
        self._sequences: dict[str, _SequenceEntry] = {}

    # ---- read-only configuration accessors ------------------------------

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def role(self) -> str:
        return self._role

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def d_model(self) -> int:
        return self._d_model

    def __repr__(self) -> str:
        # Kept for readable pytest failure output. Not a public contract.
        return (
            f"RemoteKVNode(id={self._node_id}, "
            f"role={self._role!r}, held={len(self._sequences)})"
        )

    # ---- public API ------------------------------------------------------

    def write_prefill_blocks(
        self, seq_id: str, k_data: np.ndarray, v_data: np.ndarray
    ) -> list[BlockId]:
        """Partition + snapshot K/V into blocks. Prefill-only.

        Re-writing an existing ``seq_id`` replaces its entry on this node (the
        prefill node is the owner, so replacement is the natural semantics).
        The returned BlockIds carry ``(seq_id, node_id, block_index)`` so
        transfer validation can prove ownership without trusting the caller.
        """
        seq_id = _validate_sequence_id(seq_id)
        if self._role != PREFILL_ROLE:
            raise ValueError(
                f"write_prefill_blocks is only valid on prefill nodes; "
                f"node {self._node_id} has role {self._role!r}"
            )

        k_tensor = self._validate_kv_tensor(k_data, "k_data")
        v_tensor = self._validate_kv_tensor(v_data, "v_data")
        if k_tensor.shape != v_tensor.shape:
            raise ValueError(
                f"k_data and v_data must match in shape; "
                f"got {k_tensor.shape} vs {v_tensor.shape}"
            )
        # K/V must be one tensor pair. Diverging dtypes would silently change
        # the stored dtype on read once per-sequence dtype is stored once.
        if k_tensor.dtype != v_tensor.dtype:
            raise TypeError(
                f"k_data and v_data must match in dtype; "
                f"got {k_tensor.dtype} vs {v_tensor.dtype}"
            )

        num_tokens = int(k_tensor.shape[0])
        block_size = self._block_size
        # One defensive copy per block. The node owns its own storage from
        # this point on; the caller may mutate its arrays freely.
        k_blocks = tuple(
            np.array(k_tensor[start : start + block_size], copy=True)
            for start in range(0, num_tokens, block_size)
        )
        v_blocks = tuple(
            np.array(v_tensor[start : start + block_size], copy=True)
            for start in range(0, num_tokens, block_size)
        )
        self._sequences[seq_id] = _SequenceEntry(
            k_blocks=k_blocks,
            v_blocks=v_blocks,
            dtype=k_tensor.dtype,
            token_count=num_tokens,
        )
        return [
            BlockId(seq_id, self._node_id, block_index)
            for block_index in range(len(k_blocks))
        ]

    def read_sequence(
        self, seq_id: str, num_tokens: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored prefix. ``None`` = full sequence. Missing seq → ``KeyError``.

        No authorization check: see the module docstring's "Read
        authorization" section. The caller is trusted with the ``seq_id``.
        """
        seq_id = _validate_sequence_id(seq_id)
        stored_entry = self._sequences[seq_id]  # KeyError is the documented miss signal

        if num_tokens is None:
            requested_tokens = stored_entry.token_count
        else:
            # Bool is an int subclass: `read_sequence(s, True)` must not
            # silently mean "1 token", and a negative count must not become a
            # tail slice.
            if isinstance(num_tokens, bool) or not isinstance(
                num_tokens, (int, np.integer)
            ):
                raise TypeError(
                    f"num_tokens must be an int, got {type(num_tokens).__name__}"
                )
            if num_tokens < 0:
                raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
            if num_tokens > stored_entry.token_count:
                raise ValueError(
                    f"num_tokens={num_tokens} exceeds stored length "
                    f"{stored_entry.token_count} for {seq_id!r}"
                )
            requested_tokens = int(num_tokens)

        return (
            self._copy_token_prefix(
                stored_entry.k_blocks, requested_tokens, stored_entry.dtype
            ),
            self._copy_token_prefix(
                stored_entry.v_blocks, requested_tokens, stored_entry.dtype
            ),
        )

    def release_sequence(self, seq_id: str) -> None:
        """Remove stored blocks for a sequence. Idempotent; safe on unknown ids."""
        seq_id = _validate_sequence_id(seq_id)
        self._sequences.pop(seq_id, None)

    # ---- internals used by DisaggregatedClusterManager -------------------
    # Deliberately ``_``-prefixed: manager-only helpers, not part of the
    # node's public API. Do not depend on them from outside.

    def _block_count(self, seq_id: str) -> int:
        return len(self._sequences[seq_id].k_blocks)

    def _snapshot(self, seq_id: str, block_count: int) -> _SequenceEntry:
        """Copy the first ``block_count`` blocks out of the source; source untouched.

        Iterates ``range(block_count)`` because
        ``_validate_complete_block_set`` already proved the requested set is
        exactly ``{0 .. block_count-1}`` — no sorting needed, and no risk of
        caller-supplied order affecting reconstruction.

        If this raises mid-copy, nothing has been published yet: ``_install``
        runs after this returns.
        """
        source_entry = self._sequences[seq_id]
        return _SequenceEntry(
            k_blocks=tuple(
                np.array(source_entry.k_blocks[i], copy=True)
                for i in range(block_count)
            ),
            v_blocks=tuple(
                np.array(source_entry.v_blocks[i], copy=True)
                for i in range(block_count)
            ),
            dtype=source_entry.dtype,
            token_count=source_entry.token_count,
        )

    def _install(self, seq_id: str, sequence_entry: _SequenceEntry) -> None:
        # Single dict assignment. From a reader's perspective this is atomic:
        # the entry is either absent or fully present, never half-populated.
        self._sequences[seq_id] = sequence_entry

    def _validate_kv_tensor(self, tensor: object, parameter_name: str) -> np.ndarray:
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"{parameter_name} must be np.ndarray, got {type(tensor).__name__}"
            )
        if tensor.ndim != 2:
            raise ValueError(f"{parameter_name} must be rank 2; got rank {tensor.ndim}")
        if tensor.shape[1] != self._d_model:
            raise ValueError(
                f"{parameter_name} width must be {self._d_model}; got {tensor.shape[1]}"
            )
        if tensor.dtype.kind not in SUPPORTED_DTYPE_KINDS:
            raise TypeError(
                f"{parameter_name} dtype {tensor.dtype!r} unsupported; "
                f"no implicit coercion"
            )
        return tensor

    def _copy_token_prefix(
        self,
        stored_blocks: tuple[np.ndarray, ...],
        token_count: int,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Copy exactly ``token_count`` leading tokens by selecting only touched blocks.

        Complexity: O(token_count) payload + O(ceil(token_count / block_size))
        block selection. Does not concatenate blocks beyond the requested
        prefix.
        """
        if token_count == 0:
            return np.empty((0, self._d_model), dtype=dtype)

        full_blocks_count, remainder = divmod(token_count, self._block_size)
        selected_blocks = stored_blocks[:full_blocks_count]

        if remainder:
            tail_block = np.array(
                stored_blocks[full_blocks_count][:remainder], copy=True
            )
            if full_blocks_count == 0:
                return tail_block
            return np.concatenate(selected_blocks + (tail_block,), axis=0)

        if full_blocks_count == 1:
            return np.array(stored_blocks[0], copy=True)
        return np.concatenate(selected_blocks, axis=0)


# ---------------------------------------------------------------------------
# Cluster manager
# ---------------------------------------------------------------------------


class DisaggregatedClusterManager:
    """Deterministic round-robin over two disjoint pools.

    Sticky per seq; separate cursors and sticky tables per role. Prefill IDs
    ``0..P-1``, decode IDs ``P..P+D-1`` — unique cluster-wide.

    Chose round-robin over least-loaded: round-robin is O(1) and directly
    assertable, versus an O(N) scan with a tie-break rule that is easy to
    get wrong.

    Lifecycle: ``release_request(seq_id)`` clears routing and stored blocks
    for a completed request. Idempotent; safe on unknown ids.
    """

    def __init__(
        self,
        num_prefill_nodes: int,
        num_decode_nodes: int,
        block_size: int,
        d_model: int,
    ) -> None:
        num_prefill = _validate_positive_integer(num_prefill_nodes, "num_prefill_nodes")
        num_decode = _validate_positive_integer(num_decode_nodes, "num_decode_nodes")
        self.block_size = _validate_positive_integer(block_size, "block_size")
        self.d_model = _validate_positive_integer(d_model, "d_model")

        self._prefill_nodes = tuple(
            RemoteKVNode(node_id, PREFILL_ROLE, self.block_size, self.d_model)
            for node_id in range(num_prefill)
        )
        self._decode_nodes = tuple(
            RemoteKVNode(
                num_prefill + node_id, DECODE_ROLE, self.block_size, self.d_model
            )
            for node_id in range(num_decode)
        )
        # frozenset of node objects uses identity-based membership (nodes do
        # not define __eq__/__hash__), giving O(1) expected pool checks. A
        # foreign-cluster node with the same node_id is NOT in this set.
        self._prefill_members: frozenset[RemoteKVNode] = frozenset(self._prefill_nodes)
        self._decode_members: frozenset[RemoteKVNode] = frozenset(self._decode_nodes)

        self._prefill_cursor = 0
        self._decode_cursor = 0
        self._prefill_routes: dict[str, RemoteKVNode] = {}
        self._decode_routes: dict[str, RemoteKVNode] = {}

    @property
    def prefill_nodes(self) -> tuple[RemoteKVNode, ...]:
        return self._prefill_nodes

    @property
    def decode_nodes(self) -> tuple[RemoteKVNode, ...]:
        return self._decode_nodes

    # ---- routing ---------------------------------------------------------

    def route_prefill(self, seq_id: str) -> RemoteKVNode:
        seq_id = _validate_sequence_id(seq_id)
        already_assigned = self._prefill_routes.get(seq_id)
        if already_assigned is not None:
            return already_assigned
        assigned_node = self._prefill_nodes[
            self._prefill_cursor % len(self._prefill_nodes)
        ]
        self._prefill_cursor += 1
        self._prefill_routes[seq_id] = assigned_node
        return assigned_node

    def route_decode(self, seq_id: str) -> RemoteKVNode:
        """Assign (or re-return) the decode node for ``seq_id``. Sticky.

        The decode cursor is independent of the prefill cursor and of the
        decode sticky table: a repeated call must not advance it.
        """
        seq_id = _validate_sequence_id(seq_id)
        already_assigned = self._decode_routes.get(seq_id)
        if already_assigned is not None:
            return already_assigned
        assigned_node = self._decode_nodes[
            self._decode_cursor % len(self._decode_nodes)
        ]
        self._decode_cursor += 1
        self._decode_routes[seq_id] = assigned_node
        return assigned_node

    def release_request(self, seq_id: str) -> None:
        """Release routing state and stored blocks for a completed request.

        Idempotent. Safe on unknown ids. A subsequent ``route_*`` call for the
        same ``seq_id`` treats it as a fresh request and advances the cursor.
        """
        seq_id = _validate_sequence_id(seq_id)
        prefill_node = self._prefill_routes.pop(seq_id, None)
        decode_node = self._decode_routes.pop(seq_id, None)
        if prefill_node is not None:
            prefill_node.release_sequence(seq_id)
        if decode_node is not None:
            decode_node.release_sequence(seq_id)

    # ---- transfer --------------------------------------------------------

    def transfer_kv_cache(
        self,
        seq_id: str,
        prefill_node: RemoteKVNode,
        decode_node: RemoteKVNode,
        blocks: list[BlockId] | tuple[BlockId, ...],
    ) -> None:
        """Atomic transfer. Duplicate transfer = idempotent replacement.

        Validate + build the complete destination entry, then publish with
        one dict assignment. A failure — whether during validation or during
        the snapshot copy — leaves the destination untouched (existing entry
        or none). Source is never modified.

        ``blocks`` must be the *complete* set for the sequence: missing,
        duplicated, or foreign block IDs are rejected rather than partially
        applied, because a partial publish would let a decode node read a
        sequence that is silently shorter or longer than the real one.
        """
        seq_id = _validate_sequence_id(seq_id)
        self._validate_pool_member(
            prefill_node, self._prefill_members, PREFILL_ROLE, "prefill_node"
        )
        self._validate_pool_member(
            decode_node, self._decode_members, DECODE_ROLE, "decode_node"
        )

        block_count = self._validate_complete_block_set(seq_id, prefill_node, blocks)
        # Build the whole destination entry first; nothing is published yet.
        copied_entry = prefill_node._snapshot(seq_id, block_count)
        # Publish atomically from the reader's perspective: one dict assignment.
        decode_node._install(seq_id, copied_entry)

    # ---- validation ------------------------------------------------------

    @staticmethod
    def _validate_pool_member(
        node: object,
        pool: frozenset[RemoteKVNode],
        expected_role: str,
        parameter_name: str,
    ) -> None:
        if not isinstance(node, RemoteKVNode):
            raise TypeError(
                f"{parameter_name} must be RemoteKVNode, got {type(node).__name__}"
            )
        # Identity-based membership via frozenset (RemoteKVNode does not
        # override __hash__/__eq__, so the default id-based hash is used).
        # Two clusters both number prefill nodes from 0; an id-equality
        # check would accept a foreign node.
        if node not in pool:
            raise ValueError(
                f"{parameter_name} (id={node.node_id}, role={node.role!r}) "
                f"is not in this cluster's {expected_role} pool"
            )
        if node.role != expected_role:
            raise ValueError(
                f"{parameter_name} must have role {expected_role!r}, got {node.role!r}"
            )

    def _validate_complete_block_set(
        self,
        seq_id: str,
        prefill_node: RemoteKVNode,
        blocks: list[BlockId] | tuple[BlockId, ...],
    ) -> int:
        """Return the block count after proving ``blocks == {0 .. count-1}``.

        O(B) set membership + range comparison. Chose this over
        ``sorted(seen) == list(range(n))`` (O(B log B)) because
        ``_snapshot(seq_id, count)`` iterates ``range(count)``, which is
        already in block order. The error-message sort is on the failure
        path only.
        """
        if isinstance(blocks, BlockId) or not isinstance(blocks, (list, tuple)):
            raise TypeError(
                f"blocks must be list/tuple of BlockId, got {type(blocks).__name__}"
            )

        seen_block_indices: set[int] = set()
        for block in blocks:
            if not isinstance(block, BlockId):
                raise TypeError(
                    f"each block must be BlockId, got {type(block).__name__}"
                )
            if block.seq_id != seq_id:
                raise ValueError(
                    f"block {block!r} belongs to {block.seq_id!r}, not {seq_id!r}"
                )
            if block.node_id != prefill_node.node_id:
                raise ValueError(
                    f"block {block!r} minted by node {block.node_id}, "
                    f"not source {prefill_node.node_id}"
                )
            if block.block_index in seen_block_indices:
                raise ValueError(f"duplicate block_index {block.block_index}")
            seen_block_indices.add(block.block_index)

        try:
            expected_block_count = prefill_node._block_count(seq_id)
        except KeyError:
            raise ValueError(
                f"source node {prefill_node.node_id} holds no sequence {seq_id!r}"
            ) from None

        expected_block_indices = set(range(expected_block_count))
        if seen_block_indices != expected_block_indices:
            missing_indices = sorted(expected_block_indices - seen_block_indices)
            out_of_range_indices = sorted(seen_block_indices - expected_block_indices)
            raise ValueError(
                f"incomplete block set for {seq_id!r}: "
                f"missing={missing_indices}, "
                f"out_of_range={out_of_range_indices}"
            )
        return expected_block_count


# ---------------------------------------------------------------------------
# Latency / bandwidth mathematics (pure functions)
# ---------------------------------------------------------------------------


def kv_cache_bytes(
    num_tokens: int,
    num_layers: int,
    d_model: int,
    element_size_bytes: int,
) -> int:
    """KV payload bytes: ``2 * tokens * layers * d_model * s`` (2 = K+V).

    The reduced form ``2*B*L*d_model*s`` is valid only when ``d_model`` is
    the total KV width (``d_model == H_kv * D_h``). The commonly quoted
    ``2*B*d_model*L`` is a byte count only when ``s == 1``; with fp16 it
    understates by 2x. ``element_size_bytes`` is explicit so this error
    cannot be silent.
    """
    token_count = _validate_integer_at_least(num_tokens, "num_tokens", 0)
    layer_count = _validate_positive_integer(num_layers, "num_layers")
    model_width = _validate_positive_integer(d_model, "d_model")
    element_bytes = _validate_positive_integer(element_size_bytes, "element_size_bytes")
    return 2 * token_count * layer_count * model_width * element_bytes


def ideal_transfer_seconds(
    payload_bytes: float, bandwidth_bytes_per_second: float
) -> float:
    """Payload-only lower bound: ``payload / bandwidth`` (bytes / (bytes/s) = s).

    Excludes fixed latency, protocol/serialisation overhead, contention,
    multi-hop routing. Rejects invalid bandwidth rather than clamping with
    eps — a clamp would turn a 0 B/s link into a plausible-looking finite
    latency, which is worse than an error.
    """
    if isinstance(payload_bytes, bool) or not isinstance(
        payload_bytes, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"payload_bytes must be real, got {type(payload_bytes).__name__}"
        )
    payload = float(payload_bytes)
    if not math.isfinite(payload) or payload < 0.0:
        raise ValueError(f"payload_bytes must be finite >= 0, got {payload_bytes!r}")

    if isinstance(bandwidth_bytes_per_second, bool) or not isinstance(
        bandwidth_bytes_per_second, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"bandwidth must be real, got {type(bandwidth_bytes_per_second).__name__}"
        )
    bandwidth = float(bandwidth_bytes_per_second)
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(
            f"bandwidth must be finite > 0, got {bandwidth_bytes_per_second!r}"
        )
    return payload / bandwidth
