"""Day 62 verification suite.

Three categories of test:

  - **Mutation tests** ("robust"): the assertions would fail against a
    specific named wrong implementation on the scenario the test body
    actually exercises.  Docstring names the mutation.
  - **Contract tests**: verify public API shape — return types, attribute
    existence, exception types, setter absence.  Not counted as mutation
    coverage.
  - **Guard tests**: verify defensive validation — reject non-int, reject
    bool, reject zero/negative configuration.

Five property-test *proposals* are recorded as comments at the end of the
file.  They are not implemented and are not counted toward any total.

What this suite proves, and what it does not:

  - Every mutation claim here is static: the assertion demonstrably
    distinguishes the named wrong implementation on the scenario the test
    exercises.  No mutation run has been executed.
  - The snapshot-based transactional tests capture selected mutable state
    (resident IDs, refcounts, recency, lease table, ID counters, tree
    topology with node fields).  They do not attempt general rollback
    coverage; see the module docstring's "Atomicity scope".
  - Python underscore convention does not enforce immutability.  The
    public-property test and the private-root test verify the public API,
    not the internals.
"""

import pytest
from day62_radix_attention import (
    CacheLease,
    CapacityExceededError,
    LeaseError,
    RadixCache,
    RadixTreeNode,
)


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------
def _snapshot_cache(cache: RadixCache) -> dict:
    """Capture selected mutable state for capacity-refusal atomicity checks.

    Covers resident IDs, allocator counters, clock, active-lease entry
    fields, per-node ref_count and last_used, tree topology and edge labels.
    Does not capture parent pointers directly (implied by traversal) and
    does not claim to prove no other state exists.
    """
    return {
        "resident_ids": frozenset(cache._resident),
        "next_block_id": cache._next_block_id,
        "next_lease_id": cache._next_lease_id,
        "clock": cache._clock,
        "leases": tuple(
            (
                lease_id,
                lease.block_ids,
                lease.matched_prefix_blocks,
                lease.matched_prefix_tokens,
                lease.allocated_blocks,
                lease.token_length,
            )
            for lease_id, lease in sorted(cache._active_leases.items())
        ),
        "tree": _describe_tree(cache._root),
    }


def _describe_tree(node: RadixTreeNode) -> tuple:
    return (
        node.depth,
        node.block_id,
        node.ref_count,
        node.last_used,
        tuple(node.key),
        tuple(sorted((k, _describe_tree(v)) for k, v in node.children.items())),
    )


# ---------------------------------------------------------------------------
# Reuse / longest-prefix matching
# ---------------------------------------------------------------------------
def test_identical_prompt_reuses_same_physical_block_ids():
    """Fails if insert re-allocates instead of reusing matched blocks.

    Mutation caught: _insert_structure ignores existing block_id when
    matching and always calls _create_chain.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    first = cache.insert(tokens)
    cache.release(first)
    second = cache.insert(tokens)
    assert second.block_ids == first.block_ids
    assert second.allocated_blocks == 0
    assert second.matched_prefix_blocks == 2
    assert len(cache.active_physical_blocks()) == 2


def test_shared_complete_prefix_reuses_exactly_shared_prefix_blocks():
    """Fails if a divergent suffix is aliased to the cached suffix.

    Mutation caught: returning the first lease's full block_ids for any
    request whose prefix matches, ignoring the divergent suffix.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    first = cache.insert(prefix + [9, 10])
    cache.release(first)
    second = cache.insert(prefix + [11, 12])
    assert second.block_ids == first.block_ids
    cache.release(second)

    third = cache.insert([1, 2, 3, 4, 11, 12, 13, 14])
    cache.release(third)
    assert third.block_ids[0] == first.block_ids[0]
    assert third.block_ids[1] != first.block_ids[1]
    assert third.allocated_blocks == 1


def test_divergence_inside_compressed_edge_splits_correctly():
    """Fails if a mid-edge divergence overwrites the cached edge.

    Mutation caught: _split is skipped and child.key is mutated in place.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    base = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    cache.release(base)
    divergent = cache.insert([1, 2, 3, 4, 5, 6, 99, 100])
    assert divergent.block_ids[0] == base.block_ids[0]
    assert divergent.block_ids[1] != base.block_ids[1]
    assert divergent.matched_prefix_blocks == 1
    assert divergent.matched_prefix_tokens == 6
    assert divergent.allocated_blocks == 1
    cache.release(divergent)

    again = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    assert again.block_ids == base.block_ids
    assert again.allocated_blocks == 0
    assert set(cache.active_physical_blocks()) == {0, 1, 2}


def test_three_way_divergence_shares_only_the_shared_prefix_blocks():
    """Fails if any of three branches aliases another branch's suffix block,
    or if re-inserting the first branch returns the wrong suffix.

    Mutation caught: giving _split the wrong split depth, or reusing an
    existing sibling block for a divergent suffix.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    a = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    cache.release(a)
    b = cache.insert([1, 2, 3, 4, 5, 6, 9, 10])
    cache.release(b)
    c = cache.insert([1, 2, 3, 4, 5, 6, 11, 12])
    cache.release(c)

    assert a.block_ids[0] == b.block_ids[0] == c.block_ids[0]
    assert a.block_ids[1] != b.block_ids[1]
    assert b.block_ids[1] != c.block_ids[1]
    assert a.block_ids[1] != c.block_ids[1]

    again_a = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    assert again_a.block_ids == a.block_ids
    cache.release(again_a)


def test_shorter_prompt_matches_prefix_of_longer_cached_path():
    """Fails if a shorter prompt allocates a fresh block.

    Mutation caught: refusing to match when input length is less than the
    cached path length.  Scoped to a shorter request ending at a block
    boundary; an unaligned partial-block shorter request is not exercised.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    longer = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    cache.release(longer)
    shorter = cache.insert([1, 2, 3, 4])
    assert shorter.block_ids == (longer.block_ids[0],)
    assert shorter.allocated_blocks == 0


def test_longer_prompt_matches_shorter_cached_path_and_allocates_only_suffix():
    """Fails if a longer prompt re-allocates a prefix that was already cached.

    Mutation caught: new_block_count = full_block_count instead of
    full_block_count - matched_block_count.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    short = cache.insert([1, 2, 3, 4])
    cache.release(short)
    longer = cache.insert([1, 2, 3, 4, 5, 6, 7, 8])
    assert longer.block_ids[0] == short.block_ids[0]
    assert longer.allocated_blocks == 1
    assert len(longer.block_ids) == 2


def test_offset_alignment_respected_different_token_content_at_same_offset():
    """Fails if blocks are keyed by offset only, not by content.

    Mutation caught: per-offset dict instead of a radix tree.  Scoped to
    different contents at the same offset.  The symmetric case (identical
    contents at different offsets) is not exercised directly.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    a = cache.insert([1, 2, 3, 4])
    b = cache.insert([5, 6, 7, 8])
    assert a.block_ids[0] != b.block_ids[0]
    assert b.allocated_blocks == 1
    cache.release(a)
    again = cache.insert([1, 2, 3, 4])
    assert again.block_ids == a.block_ids


# ---------------------------------------------------------------------------
# Partial-block / empty-input semantics
# ---------------------------------------------------------------------------
def test_partial_trailing_block_not_cached_or_shared():
    """Fails if a trailing partial block is cached.

    Mutation caught: n_full = ceil(N/B) for a non-empty request whose
    length is not a multiple of block_size.
    """
    cache = RadixCache(max_blocks=8, block_size=4, d_model=8)
    tokens = [1, 2, 3, 4, 5, 6]
    first = cache.insert(tokens)
    assert first.block_ids == (0,)
    assert first.token_length == 4
    cache.release(first)
    second = cache.insert(tokens)
    assert second.block_ids == (0,)
    assert second.allocated_blocks == 0
    assert len(cache.active_physical_blocks()) == 1


def test_empty_input_allocates_no_phantom_blocks():
    """Fails if empty input allocates a placeholder block.

    Mutation caught: n_full = max(1, N // B) — force at least one block even
    when N == 0.

    Note: removing only the full_block_count == 0 short-circuit does not
    change behavior — _probe and _insert_structure both produce empty
    results for empty input — so the short-circuit is an optimization, not
    the policy under test.
    """
    cache = RadixCache(max_blocks=4, block_size=4, d_model=8)
    lease = cache.insert([])
    assert lease.block_ids == ()
    assert lease.token_length == 0
    assert lease.allocated_blocks == 0
    assert cache.active_physical_blocks() == ()
    cache.release(lease)


def test_sub_block_input_allocates_no_phantom_blocks():
    """Fails if sub-block input is rounded up to a full block.

    Mutation caught: rounding `len(tokens) // block_size` up.
    """
    cache = RadixCache(max_blocks=4, block_size=4, d_model=8)
    lease = cache.insert([1, 2, 3])
    assert lease.block_ids == ()
    assert cache.active_physical_blocks() == ()


# ---------------------------------------------------------------------------
# Reference counting / lease lifecycle
# ---------------------------------------------------------------------------
def test_reference_count_increments_once_per_lease():
    """Fails if refcounts are bumped during traversal, or a lease bumps the
    same block twice.

    Mutation caught: incrementing ref_count inside _probe; or incrementing
    it twice per lease.  The test observes the consequence (a block with one
    outstanding lease is not evictable), not the count directly.
    """
    cache = RadixCache(max_blocks=1, block_size=2, d_model=4)
    lease_a = cache.insert([1, 2])
    lease_b = cache.insert([1, 2])
    with pytest.raises(CapacityExceededError):
        cache.insert([3, 4])
    cache.release(lease_a)
    with pytest.raises(CapacityExceededError):
        cache.insert([3, 4])
    cache.release(lease_b)
    lease_c = cache.insert([3, 4])
    assert lease_c.block_ids == (1,)
    assert 0 not in cache.active_physical_blocks()


def test_resident_refcounts_match_active_leases():
    """Fails if any block's refcount differs from how many active leases
    reference it.

    Mutation caught: increment/decrement on the wrong path, or a leak on
    failed insertion.
    """
    cache = RadixCache(max_blocks=8, block_size=2, d_model=4)
    lease_a = cache.insert([1, 2, 3, 4])
    lease_b = cache.insert([1, 2, 5, 6])
    lease_c = cache.insert([7, 8])
    cache.release(lease_b)

    expected: dict[int, int] = {}
    for lease in cache._active_leases.values():
        for block_id in lease.block_ids:
            expected[block_id] = expected.get(block_id, 0) + 1

    for block_id, node in cache._resident.items():
        assert node.ref_count == expected.get(block_id, 0), (
            f"block {block_id}: refcount={node.ref_count}, "
            f"active leases expect {expected.get(block_id, 0)}"
        )

    cache.release(lease_a)
    cache.release(lease_c)


def test_double_release_rejected():
    """Fails if release is silently idempotent.

    Mutation caught: removing the `lease_id in _active_leases` guard.
    """
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    lease = cache.insert([1, 2])
    cache.release(lease)
    with pytest.raises(LeaseError):
        cache.release(lease)


def test_referenced_blocks_survive_capacity_pressure():
    """Fails if eviction ignores ref_count.

    Mutation caught: dropping the `node.ref_count == 0` candidate filter.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    held = cache.insert([1, 2])
    temp_b = cache.insert([3, 4])
    temp_c = cache.insert([5, 6])
    cache.release(temp_b)
    cache.release(temp_c)
    pressure = cache.insert([7, 8])
    assert held.block_ids[0] in cache.active_physical_blocks()
    assert pressure.block_ids[0] in cache.active_physical_blocks()
    cache.release(held)
    cache.release(pressure)


def test_release_validates_before_mutating():
    """Fails if release removes the lease from the active table before
    validating the resident blocks it references.

    Mutation caught: moving `del self._active_leases[...]` ahead of the
    validation loop.
    """
    cache = RadixCache(4, 2, 4)
    lease = cache.insert([1, 2])
    del cache._resident[0]
    with pytest.raises(AssertionError):
        cache.release(lease)
    assert lease.lease_id in cache._active_leases


# ---------------------------------------------------------------------------
# Eviction ordering, tie-breaking, and leaf-only invariant
# ---------------------------------------------------------------------------
def test_released_lru_leaf_evicted_first():
    """Fails if LRU ordering is reversed.

    Mutation caught: sorting candidates with reverse=True.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    lease_a = cache.insert([1, 2])
    lease_b = cache.insert([3, 4])
    lease_c = cache.insert([5, 6])
    cache.release(lease_a)
    cache.release(lease_b)
    cache.release(lease_c)
    assert set(cache.active_physical_blocks()) == {0, 1, 2}

    first_pressure = cache.insert([7, 8])
    assert 0 not in cache.active_physical_blocks()
    assert {1, 2}.issubset(set(cache.active_physical_blocks()))

    second_pressure = cache.insert([9, 10])
    assert 1 not in cache.active_physical_blocks()
    assert 2 in cache.active_physical_blocks()
    cache.release(first_pressure)
    cache.release(second_pressure)


def test_lru_eviction_uses_recency_not_block_id():
    """Fails if eviction sorts by block_id only, dropping last_used.

    Mutation caught: key=lambda node: node.block_id.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    lease_a = cache.insert([1, 2])  # block 0, lu=1
    lease_b = cache.insert([3, 4])  # block 1, lu=2
    lease_c = cache.insert([5, 6])  # block 2, lu=3
    cache.release(lease_a)
    cache.release(lease_b)
    cache.release(lease_c)

    retouch = cache.insert([1, 2])  # block 0 re-touched, lu=4
    assert retouch.block_ids == (0,)

    pressure = cache.insert([7, 8])
    assert 1 not in cache.active_physical_blocks()  # evicted (oldest)
    assert 0 in cache.active_physical_blocks()  # re-touched, survives
    assert 2 in cache.active_physical_blocks()
    cache.release(retouch)
    cache.release(pressure)


def test_eviction_tie_breaks_by_block_id():
    """Fails if equal-recency candidates are ordered by dict iteration order
    instead of by the documented (last_used, block_id) key.

    Mutation caught: a stable `sorted(key=lambda node: node.last_used)`
    whose stable order happens to coincide with insertion order.

    The resident dict is re-inserted in reverse so insertion order and
    block_id order disagree; a stable sort by last_used alone would evict
    block 1, while (last_used, block_id) evicts block 0.
    """
    cache = RadixCache(2, 2, 4)
    lease_a = cache.insert([1, 2])
    lease_b = cache.insert([3, 4])
    cache.release(lease_a)
    cache.release(lease_b)
    cache._resident[0].last_used = 5
    cache._resident[1].last_used = 5
    cache._resident = {1: cache._resident[1], 0: cache._resident[0]}
    pressure = cache.insert([5, 6])
    assert 0 not in cache.active_physical_blocks()
    assert 1 in cache.active_physical_blocks()
    cache.release(pressure)


def test_internal_blocks_not_evicted_while_descendants_live():
    """Fails if internal (non-leaf) blocks are eviction candidates.

    Mutation caught: removing the `not node.children` filter.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    a = cache.insert([1, 2, 3, 4])
    b = cache.insert([1, 2, 5, 6])
    cache.release(a)
    cache.release(b)
    assert set(cache.active_physical_blocks()) == {0, 1, 2}

    evictor = cache.insert([7, 8])
    assert 0 in cache.active_physical_blocks()
    assert 1 not in cache.active_physical_blocks()
    cache.release(evictor)


def test_eviction_cascades_through_chain_when_children_evicted():
    """Fails if _evict only looks at current leaves and raises as soon as
    blocks_needed exceeds the number of current leaves.

    Mutation caught: missing the cascade in _eviction_order.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    chain = cache.insert([1, 2, 3, 4, 5, 6])
    cache.release(chain)
    assert set(cache.active_physical_blocks()) == {0, 1, 2}

    pressure = cache.insert([7, 8, 9, 10])
    assert len(cache.active_physical_blocks()) == 3
    assert pressure.block_ids[0] not in {0, 1, 2}
    assert pressure.block_ids[1] not in {0, 1, 2}
    assert 0 in cache.active_physical_blocks()
    cache.release(pressure)


def test_eviction_removes_stale_tree_references():
    """Fails if eviction clears block_id but leaves the parent's children
    entry in place, creating a ghost path.

    Mutation caught: skipping `del parent.children[first_token]`.
    """
    cache = RadixCache(max_blocks=2, block_size=2, d_model=4)
    a = cache.insert([1, 2])
    b = cache.insert([3, 4])
    cache.release(a)
    cache.release(b)
    pressure = cache.insert([5, 6])
    assert 0 not in cache.active_physical_blocks()
    again = cache.insert([1, 2])
    assert again.block_ids != (0,)
    assert again.allocated_blocks == 1
    cache.release(again)
    cache.release(pressure)


def test_active_physical_blocks_never_exceed_capacity():
    """Fails if capacity accounting drifts on eviction.

    Mutation caught: forgetting to remove an evicted ID from _resident.
    """
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    held = [cache.insert([1, 2]), cache.insert([3, 4])]
    evictable = [cache.insert([5, 6]), cache.insert([7, 8])]
    for lease in evictable:
        cache.release(lease)
    assert len(cache.active_physical_blocks()) == 4

    # Each insert forces at least one eviction; capacity must hold after each.
    for tokens in ([9, 10], [11, 12], [1, 2, 3, 4]):
        lease = cache.insert(tokens)
        assert len(cache.active_physical_blocks()) <= cache.max_blocks
        cache.release(lease)

    for lease in held:
        cache.release(lease)


# ---------------------------------------------------------------------------
# Capacity-refusal atomicity
# ---------------------------------------------------------------------------
def test_all_referenced_capacity_exhaustion_raises_without_corruption():
    """Fails if a capacity failure under fully-pinned blocks mutates the
    resident ID set.

    Mutation caught: evicting (or phasing out) a referenced block, or
    short-circuiting the CapacityExceededError path.

    Scope: verifies the public resident-ID set is unchanged.  Full snapshot
    equivalence (refcounts, recency, lease table, allocators, topology) is
    covered by test_transactional_failure_leaves_no_trace.
    """
    cache = RadixCache(max_blocks=2, block_size=2, d_model=4)
    a = cache.insert([1, 2])
    b = cache.insert([3, 4])
    before = cache.active_physical_blocks()
    with pytest.raises(CapacityExceededError):
        cache.insert([5, 6])
    assert cache.active_physical_blocks() == before
    cache.release(a)
    cache.release(b)


def test_insufficient_evictable_blocks_fails_transactionally():
    """Fails if partial eviction happens before the sufficiency check.

    Mutation caught: evicting candidates as they are found instead of after
    the `len(eviction_order) < blocks_needed` guard.
    """
    cache = RadixCache(max_blocks=3, block_size=2, d_model=4)
    a = cache.insert([1, 2])
    b = cache.insert([3, 4])
    c = cache.insert([5, 6])
    cache.release(c)
    before = cache.active_physical_blocks()
    with pytest.raises(CapacityExceededError):
        cache.insert([7, 8, 9, 10])
    assert cache.active_physical_blocks() == before
    cache.release(a)
    cache.release(b)


def test_failed_insert_does_not_consume_block_id():
    """Fails if a failed insertion advances _next_block_id.

    Mutation caught: calling _allocate_block (or incrementing the counter)
    before the capacity check.
    """
    cache = RadixCache(max_blocks=2, block_size=2, d_model=4)
    a = cache.insert([1, 2])
    b = cache.insert([3, 4])
    with pytest.raises(CapacityExceededError):
        cache.insert([5, 6])
    cache.release(a)
    c = cache.insert([5, 6])
    assert c.block_ids == (2,)
    cache.release(b)
    cache.release(c)


def test_transactional_failure_leaves_no_trace():
    """Fails if any state captured by _snapshot_cache changes on a capacity
    failure.

    Mutation caught: any state mutated before the _evict capacity check —
    including tree topology, refcounts, recency, resident set, lease table,
    ID allocators, or clock.

    Scope: _snapshot_cache covers resident IDs, allocator counters, clock,
    active-lease entry fields, per-node ref_count and last_used, and tree
    topology (depth, block_id, key, children).  It does not attempt to
    prove general rollback after a hypothetical post-eviction internal
    failure (see module docstring "Atomicity scope").
    """
    cache = RadixCache(2, 2, 4)
    lease_a = cache.insert([1, 2])
    lease_b = cache.insert([3, 4])
    before = _snapshot_cache(cache)
    with pytest.raises(CapacityExceededError):
        cache.insert([5, 6])
    after = _snapshot_cache(cache)
    assert before == after
    cache.release(lease_a)
    cache.release(lease_b)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
def test_tree_structural_invariants_hold_after_mixed_operations():
    """Fails if any operation leaves an empty edge label, a parent/child
    index mismatch, a depth arithmetic error, or a block ID at a
    non-boundary position.

    Mutation caught: _split producing a node with an empty key, _create_chain
    attaching at the wrong depth, or _detach_and_prune breaking the parent
    chain.

    Scope: verifies the invariants _check_invariants actually checks.
    Lease / refcount correspondence is covered separately by
    test_resident_refcounts_match_active_leases.
    """
    cache = RadixCache(8, 4, 8)
    leases = []
    for seq in (
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 99, 100],
        [1, 2, 3, 4, 50, 51],
        [200, 201, 202, 203],
    ):
        leases.append(cache.insert(seq))
        cache._check_invariants()
    for lease in leases[:2]:
        cache.release(lease)
    pressure = cache.insert([300, 301, 302, 303])
    cache._check_invariants()
    for lease in leases[2:] + [pressure]:
        cache.release(lease)
    cache._check_invariants()


# ---------------------------------------------------------------------------
# Determinism (same process)
# ---------------------------------------------------------------------------
def test_identical_operation_sequences_produce_identical_results():
    """Fails if the same operation sequence produces different outcomes
    across two fresh cache instances in the same process.

    Catches dependence on hash seed, object addresses, or dict iteration
    order that varies *between instances* in the same process.

    Does NOT catch: cross-process determinism (varying PYTHONHASHSEED), or
    an id()-based tie-break whose same-process allocation pattern happens to
    be stable.  Tie-break determinism is covered directly by
    test_eviction_tie_breaks_by_block_id.
    """

    def run():
        cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
        log = []
        lease_a = cache.insert([1, 2, 3, 4])
        log.append((lease_a.block_ids, cache.active_physical_blocks()))
        cache.release(lease_a)
        lease_b = cache.insert([5, 6, 7, 8])
        log.append((lease_b.block_ids, cache.active_physical_blocks()))
        cache.release(lease_b)
        lease_c = cache.insert([1, 2, 9, 10])
        log.append((lease_c.block_ids, cache.active_physical_blocks()))
        cache.release(lease_c)
        return log

    assert run() == run()


# ---------------------------------------------------------------------------
# Public contract (not counted toward mutation coverage)
# ---------------------------------------------------------------------------
def test_insert_returns_cachelease_with_documented_fields():
    """Fails if insert returns a bare list or drops fields."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    lease = cache.insert([1, 2, 3, 4])
    assert isinstance(lease, CacheLease)
    assert isinstance(lease.block_ids, tuple)
    assert isinstance(lease.matched_prefix_blocks, int)
    assert isinstance(lease.matched_prefix_tokens, int)
    assert isinstance(lease.allocated_blocks, int)
    assert isinstance(lease.token_length, int)
    assert len(lease) == 2
    assert list(lease) == list(lease.block_ids)
    assert lease[:1] == lease.block_ids[:1]
    cache.release(lease)


def test_active_physical_blocks_returns_sorted_tuple():
    """Fails if the accessor returns an unordered set or list."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    lease = cache.insert([1, 2, 3, 4])
    ids = cache.active_physical_blocks()
    assert isinstance(ids, tuple)
    assert list(ids) == sorted(ids)
    cache.release(lease)


def test_release_rejects_non_lease_objects():
    """Fails if release accepts arbitrary sequences."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    with pytest.raises(TypeError):
        cache.release((0, 1))  # type: ignore[arg-type]


def test_configuration_attributes_are_readable_via_properties():
    """Fails if max_blocks / block_size / d_model stop being readable after
    construction.
    """
    cache = RadixCache(max_blocks=7, block_size=3, d_model=11)
    assert cache.max_blocks == 7
    assert cache.block_size == 3
    assert cache.d_model == 11


def test_configuration_is_immutable():
    """Fails if the *public property* is assignable.

    Scope: verifies the public API has no setters.  Python's underscore
    convention does not prevent `cache._block_size = 2`; that internal
    mutation is out of scope for this test.
    """
    cache = RadixCache(8, 4, 8)
    with pytest.raises(AttributeError):
        cache.block_size = 2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cache.max_blocks = 4  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cache.d_model = 16  # type: ignore[misc]


def test_tree_root_is_private():
    """Fails if the tree root is exposed under a public name.

    Scope: verifies there is no `cache.root` attribute.  `cache._root` is
    deliberately accessible for tests and interactive debugging; that
    underscore-prefixed access is not prevented by Python.
    """
    cache = RadixCache(4, 2, 4)
    assert isinstance(cache._root, RadixTreeNode)
    assert not hasattr(cache, "root")


# ---------------------------------------------------------------------------
# Defensive validation (guard tests)
# ---------------------------------------------------------------------------
def test_constructor_rejects_boolean_max_blocks():
    """Fails if True is accepted where 1 was intended."""
    with pytest.raises(TypeError):
        RadixCache(max_blocks=True, block_size=4, d_model=8)  # type: ignore[arg-type]


def test_constructor_rejects_zero_block_size():
    """Fails if a zero block_size reaches the division paths."""
    with pytest.raises(ValueError):
        RadixCache(max_blocks=4, block_size=0, d_model=8)


def test_constructor_rejects_negative_d_model():
    """Fails if d_model is allowed to go negative."""
    with pytest.raises(ValueError):
        RadixCache(max_blocks=4, block_size=4, d_model=-1)


def test_insert_rejects_boolean_token():
    """Fails if True / False leak into the token stream."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    with pytest.raises(TypeError):
        cache.insert([1, True, 3, 4])


def test_insert_rejects_non_integer_token():
    """Fails if floats or strings reach _common_prefix_len."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    with pytest.raises(TypeError):
        cache.insert([1, 2.0, 3, 4])


def test_insert_rejects_string_input():
    """Fails if a str is silently iterated as characters."""
    cache = RadixCache(max_blocks=4, block_size=2, d_model=4)
    with pytest.raises(TypeError):
        cache.insert("1234")


def test_generator_input_matches_list_and_is_consumed_once():
    """Fails if insert consumes the input generator more than once in a way
    that loses elements.

    Mutation caught: materialising the iterable (`tuple(tokens)`) and then
    iterating it again; the second pass over an exhausted generator yields
    nothing, so the result diverges from the equivalent-list case.

    Note: calling `iter(tokens)` twice on a generator returns the *same*
    iterator, so a mere double iter() call is not what this test catches.
    """
    cache = RadixCache(8, 4, 8)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    first = cache.insert(tokens)
    cache.release(first)

    calls = 0

    def gen():
        nonlocal calls
        for token in tokens:
            calls += 1
            yield token

    second = cache.insert(gen())
    assert second.block_ids == first.block_ids
    assert second.allocated_blocks == 0
    assert calls == len(tokens)  # exactly one pass


# ---------------------------------------------------------------------------
# PROPOSED property tests (descriptive; NOT implemented, NOT counted).
#
# Hypothesis is not a dependency of this artifact.  The following properties
# describe what a property-based suite would check; none is executed by
# pytest.  Do not count them toward any test total.
#
# PROPERTY 1 — prefix-reuse identity:
#   Draw block_size ∈ [1,8], max_blocks ∈ [1,64], and non-empty shared /
#   suffix_a / suffix_b as lists of non-boolean ints.  For any pair of
#   sequences sharing a prefix that contains at least one full block, the
#   first floor(len(shared)/block_size) lease entries must be identical;
#   the first lease entry immediately after the shared prefix must not be
#   aliased when the suffixes diverge in a full block.
#   Constraints: exclude bool; require len(shared) >= block_size.
#
# PROPERTY 2 — capacity invariant:
#   Draw an op sequence of inserts and releases.  After every successful
#   insert: resident block count <= max_blocks and every resident node's
#   ref_count >= 0.
#
# PROPERTY 3 — reference non-negativity under arbitrary release order:
#   Draw a list of inserts and a permutation of their lease ids.  Release
#   in permutation order; every ref_count stays >= 0; any second release
#   raises LeaseError.
#
# PROPERTY 4 — cascade eviction:
#   Draw a chain-shaped cache (one token path of length N*block_size with
#   N blocks), release it, then request an insert of M full blocks where
#   1 <= M and M + (resident after request) <= max_blocks + N.  Assert the
#   insert succeeds; the evicted set is the M oldest blocks in
#   children-first order.
#
# PROPERTY 5 — capacity-refusal atomicity:
#   Draw a random pre-state, force a capacity failure by inserting a
#   request that needs more blocks than are evictable, and assert
#   _snapshot_cache(cache) is identical before and after the raised
#   CapacityExceededError.
# ---------------------------------------------------------------------------


# Not tested (and why):
# - Real KV tensor contents: this module is a metadata model; no payload.
# - GPU execution / CUDA kernels / device memory: stdlib-only CPU.
# - End-to-end TTFT or any 5x-10x speedup: no model, tokenizer, prefill
#   kernel, workload, or timing harness is included.  No latency claim is
#   made by this artifact.
# - Concurrency / thread-safety: the cache is single-writer; no locks.
# - Serving-loop integration with Day 61's continuous batcher: separate
#   artifact.
# - Tokenizer behaviour (BPE boundaries, special tokens): tokens are opaque
#   integer IDs.
# - Eviction of internal (non-leaf) nodes: forbidden by contract.
# - Cross-process determinism (PYTHONHASHSEED): the determinism test verifies
#   same-process reproducibility only.  A subprocess-based variant is
#   omitted because the cost would dominate the suite and the single-process
#   guarantee is what the module actually provides.
# - General rollback after eviction has begun: an internal AssertionError
#   inside _insert_structure (post-eviction) would leave partial state; the
#   documented and tested atomicity is capacity-refusal only.
# - Direct mutation of underscore-prefixed internals (_block_size,
#   _max_blocks, _d_model, _root): Python does not enforce these as private;
#   the "immutable configuration" and "private root" claims apply only to
#   the public property API.
# - Offset-sensitive identical-token-at-different-offset behaviour: the
#   offset test covers different content at the same offset; the symmetric
#   case is not exercised.
# - ID recycling across eviction: deliberately not implemented (monotonic
#   IDs); uniqueness among residents is proven by construction and by
#   _check_invariants.
# - Mutation-run verification: every "Fails if X" claim in this file is
#   static (established by reading the code path), not confirmed by running
#   a mutated implementation.
