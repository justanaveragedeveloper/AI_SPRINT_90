"""RadixAttention-style prefix cache — a metadata-only reference model.

What this module does
---------------------
When many LLM requests share a long prefix (system prompt, few-shot
examples, retrieved documents), re-running prefill for that prefix every
time is wasteful.  vLLM's RadixAttention eliminates the waste by keeping
cached prefixes in a radix tree and reusing the physical KV-cache blocks
behind matching prefixes.

This file models the *metadata* side of that idea on CPU, stdlib only:

  - a compressed-edge radix (patricia) tree over validated token IDs
  - longest-prefix matching against full cached blocks
  - physical block IDs shared between sequences whose full prefixes match
  - explicit leases with reference counting, so blocks cannot be evicted
    while anyone is still using them
  - deterministic LRU leaf eviction, with cascading: a parent becomes
    evictable once its last child is gone
  - capacity-refusal atomicity: if insert raises CapacityExceededError,
    nothing in the cache has been mutated

Complexity (algorithmic bounds — not measured)
----------------------------------------------
Let N = total input length, L = cacheable prefix length (floor(N/B) * B),
B = block size, R = resident block count, k = blocks in a lease.

  validate               O(N)          — every token checked, including
                                          the partial tail we later drop
  probe (LCP walk)       O(L)          — each token compared at most once
  structural insert      O(L)          — at most L/B new nodes and blocks
  insert, no eviction    O(N)          — dominated by validation (L <= N)
  eviction order         O(R log R)    — one heap push per eligible
                                          resident, plus bounded ancestor
                                          walks
  insert, with eviction  O(N + R log R)
  release                O(k)          — one validation pass, one decrement
                                          per leased block
  active_physical_blocks O(R log R)    — sorts the resident set

Edges are at most B tokens and never cross a block boundary.  _create_chain
splits suffixes at every block boundary, and _split can only fire strictly
inside one edge (a boundary inside an edge would contradict the previous
invariant).  That's what lets every physical block ID live on exactly one
node.

What this module is NOT
-----------------------
  - Not an attention implementation.  No hidden states, no K/V tensors, no
    GPU kernels, no TTFT measurement.
  - Not "zero-copy".  Only integer block IDs and metadata are shared; there
    are no tensor payloads here to copy or alias.
  - Not "automatic prefix caching" in the serving sense.  insert() returns
    a lease whose matched_prefix_blocks tells you what was reused.  Nothing
    schedules requests or skips prefill.
  - Not production-grade.  No locking, no tensor integration, no GPU memory
    management, no serving loop, no metrics, no benchmarks.
  - d_model is metadata kept for starter-API compatibility.  No cache
    operation reads it.

Determinism and concurrency
---------------------------
Deterministic within one process: no RNG is created or consumed, no seed
parameter exists, and identical operation sequences produce identical
trees, leases, and eviction choices.  LRU ties break on
(last_used, block_id), never on set/dict ordering or object addresses.

Cross-process determinism (varying PYTHONHASHSEED) is not tested.  The
single-process guarantee is the one this module provides.

The cache is single-writer.  No locks.  Concurrent insert/release from
multiple threads is unsupported; a serving integration would sit behind a
scheduler with its own synchronization.

Atomicity scope
---------------
Transactional only in a narrow, specific sense:

  - _evict computes the complete eviction order before touching any node.
    If it cannot free enough blocks, CapacityExceededError is raised with
    zero mutations.  That is what the tests verify.
  - After a *successful* eviction, _insert_structure runs.  Its internal
    AssertionError canaries (unreachable for valid input) would leave
    partial state if they fired.  There is no rollback path.  This is not
    general transactionality.

Python visibility caveats
-------------------------
Two things are enforced by convention, not by the interpreter:

  - "Immutable config" means max_blocks, block_size, d_model have no
    setters.  You can still do `cache._block_size = 2` and corrupt the
    tree.
  - "Private tree root" means there is no `cache.root` attribute.  It *is*
    reachable as `cache._root` and mutable through that name.

Design decisions (with rejected alternatives)
---------------------------------------------
  - Compressed-edge radix tree, not a dict keyed by whole prompts: a
    prompt-keyed dict cannot answer "what is the longest cached prefix of
    this new prompt?" without enumerating keys.  Matters when prompts share
    long prefixes but diverge — the exact workload this models.

  - Split edges at block boundaries during insert: every full block gets a
    dedicated node holding its block ID.  The alternative (block IDs on
    arbitrary nodes) would require tracking which blocks span which edges.

  - Explicit leases, not inferred ownership from token values: two requests
    can legitimately send identical tokens and must not accidentally release
    each other's references.

  - Monotonic physical IDs (never recycled): makes uniqueness among resident
    blocks trivial to prove and test.

  - Dry-run eviction order, not a rollback log: a rollback path that never
    runs in production is a liability.  (See "Atomicity scope" for exactly
    what this does and does not cover.)

  - Immutable public config properties: if a caller mutated block_size via
    the public API after construction, the tree would be built against the
    old value.  (Python does not stop `cache._block_size = 2`; see "Python
    visibility caveats".)
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RadixCacheError(RuntimeError):
    """Base class for every error this cache raises.

    CapacityExceededError and LeaseError both inherit from it, so callers
    can catch `RadixCacheError` to handle any cache-layer failure with a
    single clause.  No test catches the base directly, but the hierarchy is
    intentional public API.
    """


class CapacityExceededError(RadixCacheError):
    """Raised when a request cannot be satisfied without evicting a block
    that is still referenced by an outstanding lease.

    Atomicity contract: when _evict raises this, no tree node, block ID,
    reference count, LRU recency, lease-table entry, or ID-allocator value
    has been mutated.  See the module docstring's "Atomicity scope" for the
    exact extent of that guarantee.
    """


class LeaseError(RadixCacheError):
    """Raised when a lease is released twice, is unknown to this cache, or
    is not the exact object returned by `insert`.
    """


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _is_int(value: object) -> bool:
    """True only for real Python ints.  False for bool and everything else."""
    return isinstance(value, int) and not isinstance(value, bool)


def _common_prefix_len(
    edge_label: tuple[int, ...],
    tokens: Sequence[int],
    start: int,
) -> int:
    """Length of the longest common prefix of `edge_label` and `tokens[start:]`."""
    max_compare = min(len(edge_label), len(tokens) - start)
    matched = 0
    while matched < max_compare and edge_label[matched] == tokens[start + matched]:
        matched += 1
    return matched


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheLease:
    """An outstanding reference on a contiguous run of physical blocks.

    `matched_prefix_blocks == matched_prefix_tokens // block_size` always
    holds.  Both fields are exposed so callers don't need to re-derive one
    from the other.
    """

    lease_id: int
    block_ids: tuple[int, ...]
    matched_prefix_blocks: int
    matched_prefix_tokens: int
    allocated_blocks: int
    token_length: int

    def __len__(self) -> int:
        return len(self.block_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.block_ids)

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        return self.block_ids[index]


# ---------------------------------------------------------------------------
# Radix tree node
# ---------------------------------------------------------------------------
class RadixTreeNode:
    """One node in the compressed-edge radix tree.

    `block_id` is set iff `depth > 0` and `depth % block_size == 0`.
    A non-empty `children` dict means the node is internal and therefore
    never evictable.
    """

    __slots__ = (
        "block_id",
        "children",
        "depth",
        "key",
        "last_used",
        "parent",
        "ref_count",
    )

    def __init__(
        self,
        key: tuple[int, ...] = (),
        depth: int = 0,
        parent: RadixTreeNode | None = None,
    ) -> None:
        self.key: tuple[int, ...] = key
        self.depth: int = depth
        self.block_id: int | None = None
        self.ref_count: int = 0
        self.last_used: int = -1
        self.children: dict[int, RadixTreeNode] = {}
        self.parent: RadixTreeNode | None = parent

    def __repr__(self) -> str:
        # Kept for interactive debugging; not used by any test or caller.
        return (
            f"RadixTreeNode(depth={self.depth}, block_id={self.block_id}, "
            f"ref_count={self.ref_count}, key_len={len(self.key)}, "
            f"children={len(self.children)})"
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class RadixCache:
    """Block-granular prefix cache: radix tree plus LRU leaf eviction.

    Configuration (max_blocks, block_size, d_model) is readable through
    read-only public properties.  There are no setters.  The underscore-
    prefixed storage (_max_blocks, _block_size, _d_model) is Python-
    convention-private, not enforced.

    The internal tree is reachable only via `_root`; there is no public
    `root` attribute.  `_root` is mutable through that name; mutating it
    breaks the invariants asserted by `_check_invariants`.
    """

    def __init__(self, max_blocks: int, block_size: int, d_model: int) -> None:
        for name, value in (
            ("max_blocks", max_blocks),
            ("block_size", block_size),
            ("d_model", d_model),
        ):
            if not _is_int(value):
                raise TypeError(
                    f"{name} must be a non-boolean int, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self._max_blocks: int = max_blocks
        self._block_size: int = block_size
        self._d_model: int = d_model  # validated metadata; no tensors allocated

        self._root = RadixTreeNode()
        self._resident: dict[int, RadixTreeNode] = {}
        self._active_leases: dict[int, CacheLease] = {}
        self._next_block_id: int = 0
        self._next_lease_id: int = 0
        self._clock: int = 0

    # -- read-only configuration properties (no setters) -------------------
    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def d_model(self) -> int:
        return self._d_model

    # -- public API ---------------------------------------------------------
    def insert(self, tokens: Iterable[int]) -> CacheLease:
        """Insert `tokens` and return a lease over its full-block prefix.

        Empty input, or input shorter than one full block, produces an empty
        lease with `block_ids == ()` and allocates nothing.

        On CapacityExceededError, the cache is left exactly as it was before
        the call.  This is capacity-refusal atomicity, not a general
        rollback mechanism.
        """
        token_tuple = self._validate_tokens(tokens)
        block_size = self._block_size
        full_block_count = len(token_tuple) // block_size
        if full_block_count == 0:
            return self._register_lease(
                block_ids=(),
                matched_blocks=0,
                matched_tokens=0,
                allocated_blocks=0,
            )

        cacheable = token_tuple[: full_block_count * block_size]

        matched_token_count, protected_path = self._probe(cacheable)
        matched_block_count = matched_token_count // block_size
        new_block_count = full_block_count - matched_block_count

        resident_count = len(self._resident)
        if resident_count + new_block_count > self._max_blocks:
            self._evict(
                resident_count + new_block_count - self._max_blocks,
                protected=set(protected_path),
            )

        block_nodes = self._insert_structure(cacheable)
        if len(block_nodes) != full_block_count:  # internal invariant canary
            raise AssertionError(
                f"structural insertion produced {len(block_nodes)} block nodes, "
                f"expected {full_block_count}"
            )

        self._clock += 1
        block_ids: list[int] = []
        for node in block_nodes:
            if node.block_id is None:
                raise AssertionError("block-boundary node without a physical block id")
            node.ref_count += 1
            node.last_used = self._clock
            block_ids.append(node.block_id)

        if len(self._resident) > self._max_blocks:  # internal invariant canary
            raise AssertionError("capacity invariant violated after insertion")

        return self._register_lease(
            block_ids=tuple(block_ids),
            matched_blocks=matched_block_count,
            matched_tokens=matched_token_count,
            allocated_blocks=new_block_count,
        )

    def release(self, lease: CacheLease) -> None:
        """Drop every block reference held by `lease`, exactly once.

        Raises LeaseError if the lease was already released, is unknown to
        this cache, or is not the exact object `insert` returned.  Releasing
        does not trigger eviction; the blocks become eligible but stay
        resident until something else needs the space.

        Validation runs fully before any mutation, so a corrupted cache does
        not leave a half-released lease behind.
        """
        if not isinstance(lease, CacheLease):
            raise TypeError(
                f"release() expects a CacheLease, got {type(lease).__name__}"
            )
        existing = self._active_leases.get(lease.lease_id)
        if existing is None:
            raise LeaseError(
                f"lease id {lease.lease_id} is not active (already released or unknown)"
            )
        if existing is not lease:
            raise LeaseError(
                f"lease id {lease.lease_id} does not match the registered lease object"
            )

        # Validate everything before touching any state.
        nodes_to_release: list[RadixTreeNode] = []
        for block_id in lease.block_ids:
            node = self._resident.get(block_id)
            if node is None:
                raise AssertionError(
                    f"invariant violation: referenced block {block_id} is not resident"
                )
            if node.ref_count <= 0:
                raise AssertionError(
                    f"invariant violation: block {block_id} refcount is {node.ref_count}"
                )
            nodes_to_release.append(node)

        del self._active_leases[lease.lease_id]
        for node in nodes_to_release:
            node.ref_count -= 1

    def active_physical_blocks(self) -> tuple[int, ...]:
        """Sorted tuple of the physical block IDs currently resident."""
        return tuple(sorted(self._resident))

    # -- internal invariant check (for tests) ------------------------------
    def _check_invariants(self) -> None:
        """Assert every structural invariant; raise AssertionError otherwise.

        Intended for tests only.  Safe to call after any committed
        operation; does not modify state.  Does not check lease / refcount
        correspondence (test_resident_refcounts_match_active_leases covers
        that separately).
        """
        if len(self._resident) > self._max_blocks:
            raise AssertionError(
                f"resident block count {len(self._resident)} exceeds "
                f"max_blocks {self._max_blocks}"
            )
        if (
            self._root.key != ()
            or self._root.depth != 0
            or self._root.parent is not None
        ):
            raise AssertionError("root has non-default state")
        if self._root.block_id is not None:
            raise AssertionError("root must not hold a physical block id")

        visited: list[RadixTreeNode] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if any(node is seen for seen in visited):
                raise AssertionError("cycle detected in radix tree")
            visited.append(node)
            if node is not self._root:
                self._check_node(node)
            stack.extend(node.children.values())

        for block_id, resident_node in self._resident.items():
            if resident_node.block_id != block_id:
                raise AssertionError(
                    f"resident[{block_id}].block_id = {resident_node.block_id}"
                )
            if not any(resident_node is seen for seen in visited):
                raise AssertionError(
                    f"resident block {block_id} is not reachable from the root"
                )

    def _check_node(self, node: RadixTreeNode) -> None:
        if not node.key:
            raise AssertionError(f"node at depth {node.depth} has empty edge label")
        if node.parent is None:
            raise AssertionError(f"non-root node at depth {node.depth} has no parent")
        if node.depth != node.parent.depth + len(node.key):
            raise AssertionError(
                f"node depth {node.depth} != parent depth {node.parent.depth} "
                f"+ edge length {len(node.key)}"
            )
        if node.parent.children.get(node.key[0]) is not node:
            raise AssertionError(
                f"parent child index missing/incorrect for first token {node.key[0]}"
            )
        if node.block_id is not None:
            if node.depth == 0 or node.depth % self._block_size != 0:
                raise AssertionError(
                    f"block_id set on node at depth {node.depth}, not a block boundary"
                )
            if self._resident.get(node.block_id) is not node:
                raise AssertionError(
                    f"resident index does not map block_id {node.block_id} to this node"
                )
            if node.ref_count < 0:
                raise AssertionError(
                    f"node at depth {node.depth} has negative refcount {node.ref_count}"
                )
        elif node.ref_count != 0:
            raise AssertionError(
                f"node without block_id has nonzero refcount {node.ref_count}"
            )

    # -- validation ---------------------------------------------------------
    def _validate_tokens(self, tokens: Iterable[int]) -> tuple[int, ...]:
        if isinstance(tokens, (str, bytes)):
            raise TypeError("tokens must be a sequence of integers, not a string/bytes")
        try:
            token_iter = iter(tokens)
        except TypeError as exc:
            raise TypeError("tokens must be an iterable of integers") from exc
        validated: list[int] = []
        for position, token in enumerate(token_iter):
            if not _is_int(token):
                raise TypeError(
                    f"token at index {position} must be a non-boolean int, "
                    f"got {type(token).__name__}"
                )
            validated.append(token)
        return tuple(validated)

    # -- probing ------------------------------------------------------------
    def _probe(self, tokens: tuple[int, ...]) -> tuple[int, list[RadixTreeNode]]:
        """Walk the tree to find the longest cached prefix of `tokens`.

        Returns (matched_token_count, protected_path).  `protected_path` is
        the list of fully matched nodes from root to the last node whose
        edge matched in full; the divergent edge node is deliberately
        excluded because _insert_structure will split it.
        """
        node = self._root
        path: list[RadixTreeNode] = [node]
        position = 0
        token_count = len(tokens)
        while position < token_count:
            child = node.children.get(tokens[position])
            if child is None:
                break
            edge_label = child.key
            common_len = _common_prefix_len(edge_label, tokens, position)
            if common_len == len(edge_label):
                position += common_len
                node = child
                path.append(node)
            else:
                # Divergence mid-edge: report the match up to the divergence.
                return position + common_len, path
        return position, path

    # -- structural insertion ----------------------------------------------
    def _insert_structure(self, tokens: tuple[int, ...]) -> list[RadixTreeNode]:
        """Wire `tokens` into the tree; return its block-boundary nodes in order."""
        node = self._root
        block_nodes: list[RadixTreeNode] = []
        position = 0
        token_count = len(tokens)
        while position < token_count:
            child = node.children.get(tokens[position])
            if child is None:
                block_nodes.extend(self._create_chain(node, tokens, position))
                return block_nodes
            edge_label = child.key
            common_len = _common_prefix_len(edge_label, tokens, position)
            if common_len == len(edge_label):
                position += common_len
                node = child
                if node.block_id is not None:
                    block_nodes.append(node)
            else:
                mid_node = self._split(child, common_len)
                block_nodes.extend(
                    self._create_chain(mid_node, tokens, position + common_len)
                )
                return block_nodes
        return block_nodes

    def _split(self, child: RadixTreeNode, split_at: int) -> RadixTreeNode:
        # Precondition, enforced by the caller: 0 < split_at < len(child.key).
        parent = child.parent
        if parent is None:
            raise AssertionError("cannot split the root")
        old_key = child.key
        first_token = old_key[0]
        mid_node = RadixTreeNode(
            key=old_key[:split_at],
            depth=parent.depth + split_at,
            parent=parent,
        )
        child.key = old_key[split_at:]
        child.parent = mid_node
        mid_node.children[old_key[split_at]] = child
        parent.children[first_token] = mid_node
        return mid_node

    def _create_chain(
        self, node: RadixTreeNode, tokens: tuple[int, ...], start: int
    ) -> list[RadixTreeNode]:
        """Append fresh block-boundary nodes covering tokens[start:]."""
        block_size = self._block_size
        token_count = len(tokens)
        created: list[RadixTreeNode] = []
        position = start
        while position < token_count:
            current_depth = node.depth
            next_boundary = ((current_depth // block_size) + 1) * block_size
            chunk_length = next_boundary - current_depth
            if chunk_length <= 0 or position + chunk_length > token_count:
                raise AssertionError(
                    "internal invariant: block-boundary chunk out of range"
                )
            chunk = tokens[position : position + chunk_length]
            new_node = RadixTreeNode(key=chunk, depth=next_boundary, parent=node)
            self._allocate_block(new_node)
            node.children[chunk[0]] = new_node
            node = new_node
            created.append(new_node)
            position += chunk_length
        return created

    def _allocate_block(self, node: RadixTreeNode) -> int:
        new_id = self._next_block_id
        self._next_block_id += 1
        node.block_id = new_id
        self._resident[new_id] = node
        return new_id

    # -- eviction -----------------------------------------------------------
    def _evict(self, blocks_needed: int, protected: set[RadixTreeNode]) -> None:
        """Evict `blocks_needed` cacheable blocks in deterministic LRU order.

        Cascades: evicting a leaf may make its parent a leaf, which may
        become evictable in the same call.  Atomic on capacity refusal — the
        full eviction order is computed before any node is mutated.
        """
        eviction_order = self._eviction_order(protected)
        if len(eviction_order) < blocks_needed:
            raise CapacityExceededError(
                f"cannot free {blocks_needed} block(s): only "
                f"{len(eviction_order)} block(s) are cascade-evictable; "
                f"{len(self._active_leases)} lease(s) still active"
            )
        for node in eviction_order[:blocks_needed]:
            self._evict_one(node, protected)

    def _eviction_order(self, protected: set[RadixTreeNode]) -> list[RadixTreeNode]:
        """Return evictable block nodes, children before parents, LRU first."""
        candidate_heap: list[tuple[int, int, RadixTreeNode]] = []
        already_queued: set[RadixTreeNode] = set()
        for node in self._resident.values():
            if node.ref_count == 0 and node not in protected and not node.children:
                heapq.heappush(candidate_heap, (node.last_used, node.block_id, node))
                already_queued.add(node)

        result: list[RadixTreeNode] = []
        already_handled: set[RadixTreeNode] = set()
        while candidate_heap:
            _, _, node = heapq.heappop(candidate_heap)
            if node in already_handled:
                continue
            already_handled.add(node)
            result.append(node)

            # Walk up: any ancestor that is now unreferenced and whose
            # children are all gone becomes newly eligible.
            ancestor = node.parent
            while (
                ancestor is not None
                and ancestor is not self._root
                and ancestor not in protected
            ):
                if not all(
                    child in already_handled for child in ancestor.children.values()
                ):
                    break
                if ancestor.ref_count > 0:
                    break
                if ancestor.block_id is not None:
                    if ancestor not in already_queued:
                        heapq.heappush(
                            candidate_heap,
                            (ancestor.last_used, ancestor.block_id, ancestor),
                        )
                        already_queued.add(ancestor)
                    break
                # Mid-split node (no block_id): _detach_and_prune will drop
                # it once its last child is evicted.  Keep walking up.
                already_handled.add(ancestor)
                ancestor = ancestor.parent
        return result

    def _evict_one(self, node: RadixTreeNode, protected: set[RadixTreeNode]) -> None:
        block_id = node.block_id
        if block_id is None:
            raise AssertionError("eviction candidate without a physical block id")
        del self._resident[block_id]
        node.block_id = None
        self._detach_and_prune(node, protected)

    def _detach_and_prune(
        self, node: RadixTreeNode, protected: set[RadixTreeNode]
    ) -> None:
        """Detach `node` if it became empty, then prune empty ancestors."""
        while node is not self._root and node not in protected:
            if node.block_id is not None or node.children:
                return
            parent = node.parent
            if parent is None:
                raise AssertionError("non-root node without a parent")
            first_token = node.key[0]
            if parent.children.get(first_token) is not node:
                raise AssertionError("tree / child index inconsistency")
            del parent.children[first_token]
            node.parent = None
            node = parent

    # -- lease bookkeeping --------------------------------------------------
    def _register_lease(
        self,
        block_ids: tuple[int, ...],
        matched_blocks: int,
        matched_tokens: int,
        allocated_blocks: int,
    ) -> CacheLease:
        lease = CacheLease(
            lease_id=self._next_lease_id,
            block_ids=block_ids,
            matched_prefix_blocks=matched_blocks,
            matched_prefix_tokens=matched_tokens,
            allocated_blocks=allocated_blocks,
            token_length=len(block_ids) * self._block_size,
        )
        self._next_lease_id += 1
        self._active_leases[lease.lease_id] = lease
        return lease
