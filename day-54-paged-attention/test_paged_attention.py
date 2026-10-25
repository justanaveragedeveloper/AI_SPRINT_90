"""Mutation-targeted tests for the paged KV cache.

Each test names a specific wrong implementation it catches. Tags:

  [ROBUST]   catches a concrete wrong implementation
  [CONTRACT] checks a documented interface/behavior
  [THEATER]  only happy path (none here)

Run: pytest -v test_paged_attention.py
"""

from __future__ import annotations

import numpy as np
import pytest
from day54_paged_attention import PagedKVCache, paged_attention_forward

try:  # optional dependency; skip the property test if it's absent
    from hypothesis import given, settings  # type: ignore[import-untyped]
    from hypothesis import strategies as st  # type: ignore[import-untyped]

    _HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    _HAS_HYPOTHESIS = False


def _tokens(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _dense_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Float64 dense attention used as an independent oracle.

    Deliberately not the implementation's code path -- computing the
    reference in float64 means the tolerance is validated against real
    float32 rounding on the implementation side, not against the same
    rounding on both sides.
    """
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)
    scores = (Q64 @ K64.T) / np.sqrt(K64.shape[1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    return w @ V64


# --------------------------------------------------------------------------- #
# Block counting and logical/physical mapping
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if allocate_sequence uses floor division (6 // 4 == 1 block)
# or drops the ceiling, because 6 tokens need a second, partially filled block.
def test_six_tokens_with_block_size_four_allocate_two_blocks():
    cache = PagedKVCache(num_blocks=10, block_size=4, d_model=8)
    cache.allocate_sequence(0, initial_tokens=6)

    assert len(cache.owned_blocks(0)) == 2
    assert cache.allocated_capacity(0) == 8
    assert cache.free_block_count == 8


# [ROBUST] Fails if read_sequence concatenates whole blocks without trimming
# the partial final block (would return 8 rows for 6 tokens), or computes the
# in-block offset as `t` instead of `t % block_size`.
def test_read_across_block_boundary_returns_tokens_in_logical_order():
    cache = PagedKVCache(num_blocks=4, block_size=4, d_model=3)
    cache.allocate_sequence(0, initial_tokens=6)
    k = _tokens(6, 3, seed=11)
    v = _tokens(6, 3, seed=12)
    cache.write_tokens(0, 0, k, v)

    k_out, v_out = cache.read_sequence(0, 6)

    assert k_out.shape == (6, 3)
    np.testing.assert_array_equal(k_out, k)
    np.testing.assert_array_equal(v_out, v)


# [ROBUST] Fails if reads ignore the page table and walk physical blocks in
# ascending order, or if the page table is stored/consumed as a set.
def test_non_contiguous_out_of_order_physical_blocks_reconstruct_logical_order():
    cache = PagedKVCache(num_blocks=6, block_size=2, d_model=3)
    cache.allocate_sequence(10, initial_tokens=1)  # block 0
    cache.allocate_sequence(11, initial_tokens=1)  # block 1
    cache.allocate_sequence(12, initial_tokens=1)  # block 2
    cache.allocate_sequence(13, initial_tokens=3)  # block_size=2 -> 2 blocks: 3, 4
    cache.allocate_sequence(14, initial_tokens=1)  # block 5
    assert cache.free_block_count == 0

    cache.release_sequence(12)  # free list -> [2]
    cache.release_sequence(10)  # free list -> [2, 0]
    cache.allocate_sequence(20, initial_tokens=4)  # 2 blocks -> [2, 0]

    assert cache.owned_blocks(20) == [2, 0]

    k = _tokens(4, 3, seed=21)
    v = _tokens(4, 3, seed=22)
    cache.write_tokens(20, 0, k, v)

    k_out, v_out = cache.read_sequence(20, 4)
    np.testing.assert_array_equal(k_out, k)
    np.testing.assert_array_equal(v_out, v)

    np.testing.assert_array_equal(cache.key_pool[2, 0], k[0])
    np.testing.assert_array_equal(cache.key_pool[2, 1], k[1])
    np.testing.assert_array_equal(cache.key_pool[0, 0], k[2])
    np.testing.assert_array_equal(cache.key_pool[0, 1], k[3])
    np.testing.assert_array_equal(cache.value_pool[2, 1], v[1])


# [ROBUST] Fails if writes ignore start_idx (always write from slot 0), use
# `t` as the in-block offset, or write every token into the first owned
# block.
def test_write_at_nonzero_logical_position_hits_mapped_block_and_offset():
    cache = PagedKVCache(num_blocks=4, block_size=4, d_model=3)
    cache.allocate_sequence(0, initial_tokens=8)
    k = _tokens(8, 3, seed=31)
    v = _tokens(8, 3, seed=32)
    cache.write_tokens(0, 0, k, v)
    pages = cache.owned_blocks(0)
    assert len(pages) == 2

    sentinel_k = np.full((1, 3), 7.0, dtype=np.float32)
    sentinel_v = np.full((1, 3), -3.0, dtype=np.float32)
    cache.write_tokens(0, 5, sentinel_k, sentinel_v)

    np.testing.assert_array_equal(cache.key_pool[pages[1], 1], sentinel_k[0])
    np.testing.assert_array_equal(cache.value_pool[pages[1], 1], sentinel_v[0])
    np.testing.assert_array_equal(cache.key_pool[pages[1], 0], k[4])
    np.testing.assert_array_equal(cache.key_pool[pages[0], 1], k[1])

    k_out, v_out = cache.read_sequence(0, 8)
    np.testing.assert_array_equal(k_out[5], sentinel_k[0])
    np.testing.assert_array_equal(v_out[5], sentinel_v[0])
    np.testing.assert_array_equal(k_out[4], k[4])
    assert cache.sequence_length(0) == 8


# [ROBUST] Fails if a multi-token write that starts mid-block and crosses a
# boundary writes every token into the first block or uses `start_idx + i`
# as the in-block offset instead of `(start_idx + i) % block_size`.
def test_write_crossing_block_boundary_from_mid_block():
    cache = PagedKVCache(num_blocks=3, block_size=4, d_model=3)
    cache.allocate_sequence(0, initial_tokens=8)  # 2 blocks
    k = _tokens(8, 3, seed=401)
    v = _tokens(8, 3, seed=402)
    cache.write_tokens(0, 0, k, v)
    pages = cache.owned_blocks(0)

    new_k = _tokens(4, 3, seed=403)
    new_v = _tokens(4, 3, seed=404)
    cache.write_tokens(
        0, 2, new_k, new_v
    )  # spans block 0 slots 2,3 + block 1 slots 0,1

    np.testing.assert_array_equal(cache.key_pool[pages[0], 2], new_k[0])
    np.testing.assert_array_equal(cache.key_pool[pages[0], 3], new_k[1])
    np.testing.assert_array_equal(cache.key_pool[pages[1], 0], new_k[2])
    np.testing.assert_array_equal(cache.key_pool[pages[1], 1], new_k[3])

    k_out, _ = cache.read_sequence(0, 8)
    np.testing.assert_array_equal(k_out[0:2], k[0:2])
    np.testing.assert_array_equal(k_out[2:6], new_k)
    np.testing.assert_array_equal(k_out[6:8], k[6:8])


# [ROBUST] Fails if allocate_sequence computes `T // B + 1` unconditionally,
# because an exact multiple (8 with B=4) would reserve a third, empty block.
def test_exact_block_multiple_allocates_no_extra_block():
    cache = PagedKVCache(num_blocks=5, block_size=4, d_model=2)
    cache.allocate_sequence(0, initial_tokens=8)
    assert len(cache.owned_blocks(0)) == 2
    assert cache.allocated_capacity(0) == 8

    cache.allocate_sequence(1, initial_tokens=4)
    assert len(cache.owned_blocks(1)) == 1

    cache.allocate_sequence(2, initial_tokens=1)
    assert len(cache.owned_blocks(2)) == 1


# [ROBUST] Fails if the ceiling is computed as `T // B` (would give 1 block
# for B+1 tokens).
def test_sequence_of_block_size_plus_one_allocates_two_blocks():
    cache = PagedKVCache(num_blocks=5, block_size=4, d_model=2)
    cache.allocate_sequence(0, initial_tokens=5)
    assert len(cache.owned_blocks(0)) == 2
    assert cache.allocated_capacity(0) == 8


# --------------------------------------------------------------------------- #
# Zero-token contract
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if allocate_sequence(initial_tokens=0) is rejected, or
# allocates a placeholder block, or if read_sequence(None) is not "read all
# valid tokens" (here zero), or if attention over zero tokens silently
# returns zeros instead of raising.
def test_zero_token_allocation_contract():
    cache = PagedKVCache(num_blocks=2, block_size=4, d_model=3)
    cache.allocate_sequence(0, initial_tokens=0)

    assert cache.owned_blocks(0) == []
    assert cache.sequence_length(0) == 0
    assert cache.allocated_capacity(0) == 0
    assert cache.free_block_count == 2

    k0, v0 = cache.read_sequence(0, 0)
    assert k0.shape == (0, 3) and v0.shape == (0, 3)
    k_all, v_all = cache.read_sequence(0)
    assert k_all.shape == (0, 3) and v_all.shape == (0, 3)

    cache.write_tokens(0, 0, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    assert cache.sequence_length(0) == 0

    with pytest.raises(ValueError):
        cache.read_sequence(0, 1)
    with pytest.raises(ValueError):
        paged_attention_forward(np.zeros((1, 3), np.float32), cache, 0)


# --------------------------------------------------------------------------- #
# Gap-write rejection
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if write_tokens accepts a write past the current valid
# length, which would mark unwritten positions as valid and cause
# read_sequence to return zero-filled "tokens". The failure must leave the
# pool, page table, and valid-length counter unchanged.
def test_write_cannot_create_a_gap_in_the_valid_prefix():
    cache = PagedKVCache(num_blocks=1, block_size=4, d_model=2)
    cache.allocate_sequence(0, initial_tokens=4)
    assert cache.sequence_length(0) == 0

    before_pool = cache.key_pool.copy()
    sentinel = np.full((1, 2), 9.0, np.float32)

    with pytest.raises(ValueError):
        cache.write_tokens(0, 2, sentinel, sentinel)  # gap: current valid is 0

    assert cache.sequence_length(0) == 0
    np.testing.assert_array_equal(cache.key_pool, before_pool)

    cache.write_tokens(0, 0, _tokens(2, 2, seed=501), _tokens(2, 2, seed=502))
    assert cache.sequence_length(0) == 2
    cache.write_tokens(0, 1, sentinel, sentinel)  # overwrite, not a gap
    assert cache.sequence_length(0) == 2
    with pytest.raises(ValueError):
        cache.write_tokens(0, 3, sentinel, sentinel)  # gap: valid is 2


# --------------------------------------------------------------------------- #
# Allocation failure atomicity
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if allocate_sequence pops blocks one at a time and raises on
# exhaustion: the free count would drop, or a partial page table would be
# published, or the ownership map would be disturbed.
def test_failed_allocation_leaves_no_partial_state():
    cache = PagedKVCache(num_blocks=4, block_size=2, d_model=2)
    for sid in (0, 1, 2):
        cache.allocate_sequence(sid, initial_tokens=2)
    assert cache.free_block_count == 1

    before_blocks = {sid: cache.owned_blocks(sid) for sid in (0, 1, 2)}
    before_owners = {b: cache.block_owner(b) for b in range(cache.num_blocks)}
    before_lengths = {sid: cache.sequence_length(sid) for sid in (0, 1, 2)}

    with pytest.raises(RuntimeError):
        cache.allocate_sequence(3, initial_tokens=4)  # needs 2, only 1 free

    assert 3 not in cache.page_tables
    with pytest.raises(KeyError):
        cache.sequence_length(3)
    assert cache.free_block_count == 1
    assert {sid: cache.owned_blocks(sid) for sid in (0, 1, 2)} == before_blocks
    assert {b: cache.block_owner(b) for b in range(cache.num_blocks)} == before_owners
    assert {sid: cache.sequence_length(sid) for sid in (0, 1, 2)} == before_lengths


# --------------------------------------------------------------------------- #
# Exclusive ownership
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if the allocator hands out a block that is still owned, or
# if release does not clear ownership. Makes no assumption about which
# specific ids are reused, only that ownership invariants hold.
def test_exclusive_ownership_no_block_shared_between_live_sequences():
    cache = PagedKVCache(num_blocks=8, block_size=2, d_model=2)
    for sid in (0, 1, 2):
        cache.allocate_sequence(sid, initial_tokens=4)  # 2 blocks each

    tables = {sid: cache.owned_blocks(sid) for sid in (0, 1, 2)}
    flat = [b for t in tables.values() for b in t]
    assert len(flat) == len(set(flat)) == 6
    for sid, t in tables.items():
        for b in t:
            assert cache.block_owner(b) == sid
    assert cache.free_block_count == 2

    released = tables[1]
    cache.release_sequence(1)
    assert cache.free_block_count == 4
    for b in released:
        assert cache.block_owner(b) is None

    cache.allocate_sequence(3, initial_tokens=4)
    reused = cache.owned_blocks(3)
    assert len(reused) == 2
    assert cache.free_block_count == 2
    other_blocks = [b for sid in (0, 2) for b in cache.owned_blocks(sid)]
    assert set(reused).isdisjoint(set(other_blocks))
    for b in reused:
        assert cache.block_owner(b) == 3


# --------------------------------------------------------------------------- #
# Duplicate / invalid sequence ids
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if duplicate allocation silently overwrites the existing
# page table. Negative integers raise ValueError (previous version
# incorrectly expected TypeError; -1 is an integer, just out of range).
def test_duplicate_sequence_id_is_rejected():
    cache = PagedKVCache(num_blocks=4, block_size=2, d_model=2)
    cache.allocate_sequence(7, initial_tokens=4)
    before_blocks = cache.owned_blocks(7)

    with pytest.raises(ValueError):
        cache.allocate_sequence(7, initial_tokens=2)

    assert cache.owned_blocks(7) == before_blocks
    assert cache.free_block_count == 2

    with pytest.raises(ValueError):  # integer out of range
        cache.allocate_sequence(-1, initial_tokens=2)
    with pytest.raises(TypeError):  # non-integer type
        cache.allocate_sequence(1.5, initial_tokens=2)


# --------------------------------------------------------------------------- #
# Defensive validation
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if the constructor accepts zero/negative/fractional dims.
@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"num_blocks": 0, "block_size": 4, "d_model": 2}, ValueError),
        ({"num_blocks": 4, "block_size": 0, "d_model": 2}, ValueError),
        ({"num_blocks": 4, "block_size": 4, "d_model": 0}, ValueError),
        ({"num_blocks": -3, "block_size": 4, "d_model": 2}, ValueError),
        ({"num_blocks": 4.0, "block_size": 4, "d_model": 2}, TypeError),
        ({"num_blocks": True, "block_size": 4, "d_model": 2}, TypeError),
    ],
)
def test_constructor_rejects_invalid_dimensions(kwargs, exc):
    with pytest.raises(exc):
        PagedKVCache(**kwargs)


# [ROBUST] Fails if write_tokens validates nothing and relies on NumPy
# broadcasting. Every failing call must leave the pool and valid length
# unchanged.
def test_write_rejects_invalid_shapes_values_and_ranges():
    cache = PagedKVCache(num_blocks=2, block_size=2, d_model=3)
    cache.allocate_sequence(0, initial_tokens=4)
    good = np.zeros((2, 3), np.float32)

    with pytest.raises(ValueError):  # wrong width
        cache.write_tokens(0, 0, np.zeros((2, 4), np.float32), good)
    with pytest.raises(ValueError):  # 1-D
        cache.write_tokens(0, 0, np.zeros(3, np.float32), np.zeros(3, np.float32))
    with pytest.raises(ValueError):  # length mismatch
        cache.write_tokens(0, 0, good, np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError):  # NaN
        cache.write_tokens(0, 0, np.full((2, 3), np.nan, np.float32), good)
    with pytest.raises(ValueError):  # inf
        cache.write_tokens(0, 0, good, np.full((2, 3), np.inf, np.float32))
    with pytest.raises(ValueError):  # negative logical index
        cache.write_tokens(0, -1, good, good)
    with pytest.raises(ValueError):  # start 3 + 2 tokens > capacity 4
        cache.write_tokens(0, 3, good, good)
    with pytest.raises(TypeError):  # non-integer index
        cache.write_tokens(0, 1.0, good, good)

    assert cache.sequence_length(0) == 0
    assert np.all(cache.key_pool == 0.0)


# --------------------------------------------------------------------------- #
# Reads never expose unwritten slots
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if read_sequence trusts num_tokens and returns whatever is
# in the pool.
def test_read_past_valid_length_raises_instead_of_returning_zero_slots():
    cache = PagedKVCache(num_blocks=2, block_size=4, d_model=2)
    cache.allocate_sequence(0, initial_tokens=8)
    cache.write_tokens(0, 0, _tokens(3, 2, seed=41), _tokens(3, 2, seed=42))

    assert cache.sequence_length(0) == 3
    k_out, _ = cache.read_sequence(0, 3)
    assert k_out.shape == (3, 2)

    with pytest.raises(ValueError):
        cache.read_sequence(0, 4)
    with pytest.raises(ValueError):
        cache.read_sequence(0, 8)
    with pytest.raises(ValueError):
        cache.read_sequence(0, -1)


# [ROBUST] Fails if sequence lookups fall through to a default empty page
# table, which would turn a typo'd seq_id into a silent zero-token sequence.
def test_unknown_sequence_id_raises_keyerror():
    cache = PagedKVCache(num_blocks=2, block_size=2, d_model=2)
    cache.allocate_sequence(0, initial_tokens=2)
    q = np.zeros((1, 2), np.float32)

    with pytest.raises(KeyError):
        cache.write_tokens(
            5, 0, np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32)
        )
    with pytest.raises(KeyError):
        cache.read_sequence(5, 1)
    with pytest.raises(KeyError):
        cache.release_sequence(5)
    with pytest.raises(KeyError):
        cache.sequence_length(5)
    with pytest.raises(KeyError):
        paged_attention_forward(q, cache, 5)


# --------------------------------------------------------------------------- #
# Page-table integrity
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if the read path trusts a corrupted page-table entry. In
# NumPy, indexing with -1 wraps to the last block, so without validation a
# corrupted entry would silently return data from the wrong physical block.
# A stale entry pointing at another sequence's block must also be rejected.
def test_corrupted_page_table_entry_raises():
    cache = PagedKVCache(num_blocks=3, block_size=2, d_model=2)
    cache.allocate_sequence(0, initial_tokens=4)  # 2 blocks, e.g. [0, 1]
    cache.write_tokens(0, 0, _tokens(4, 2, seed=601), _tokens(4, 2, seed=602))

    original = cache.owned_blocks(0)

    cache.page_tables[0][0] = -1  # would wrap to block 2 without validation
    with pytest.raises(ValueError):
        cache.read_sequence(0, 4)
    cache.page_tables[0][0] = 99
    with pytest.raises(ValueError):
        cache.read_sequence(0, 4)

    cache.page_tables[0] = list(original)
    cache.allocate_sequence(1, initial_tokens=2)  # takes next free block
    other_block = cache.owned_blocks(1)[0]
    cache.page_tables[0][1] = other_block  # now points at seq 1's block
    with pytest.raises(ValueError):
        cache.read_sequence(0, 4)


# [ROBUST] Fails if release_sequence trusts a corrupted page table and
# appends the same physical block to the free list twice. A duplicate
# entry would corrupt every subsequent allocation. Reads and attention may
# tolerate a redirected table like [1, 1] (that's how the redirect test
# works), but release must refuse it.
def test_release_rejects_duplicate_page_table_entries():
    cache = PagedKVCache(num_blocks=3, block_size=2, d_model=2)
    cache.allocate_sequence(0, initial_tokens=4)  # 2 blocks, e.g. [0, 1]
    cache.write_tokens(
        0,
        0,
        np.zeros((4, 2), np.float32),
        np.zeros((4, 2), np.float32),
    )
    free_before = cache.free_block_count  # 1 (block 2 is free)

    cache.page_tables[0] = [1, 1]  # corrupt: duplicate entry, block 0 orphaned

    with pytest.raises(ValueError):
        cache.release_sequence(0)

    # State unchanged: nothing appended to free list, ownership intact.
    assert cache.free_block_count == free_before
    assert cache.block_owner(1) == 0
    assert cache.block_owner(0) == 0  # orphan, but still owned by seq 0
    assert 0 in cache.page_tables  # sequence still present

    # Repairing the table makes release succeed.
    cache.page_tables[0] = [0, 1]
    cache.release_sequence(0)
    assert cache.free_block_count == 3
    assert 0 not in cache.page_tables


# [ROBUST] Fails if a shortened page table reaches a low-level IndexError
# (or silently reads whatever happens to be there) instead of raising a
# deliberate ValueError. The capacity check in `_validated_pages` is what
# converts the low-level failure into a documented error.
def test_shortened_page_table_raises():
    cache = PagedKVCache(num_blocks=3, block_size=2, d_model=2)
    cache.allocate_sequence(0, initial_tokens=4)  # 2 blocks
    cache.write_tokens(0, 0, _tokens(4, 2, seed=901), _tokens(4, 2, seed=902))
    assert cache.sequence_length(0) == 4

    del cache.page_tables[0][-1]  # truncate: 1 block = 2 slots < 4 valid

    with pytest.raises(ValueError):
        cache.read_sequence(0, 4)
    with pytest.raises(ValueError):
        cache.write_tokens(0, 0, _tokens(1, 2, seed=903), _tokens(1, 2, seed=904))
    with pytest.raises(ValueError):
        paged_attention_forward(_tokens(1, 2, seed=905), cache, 0)
    with pytest.raises(ValueError):
        cache.release_sequence(0)  # release also validates

    # The truncated table is still present (one block, below valid length).
    assert 0 in cache.page_tables
    assert len(cache.page_tables[0]) == 1


# --------------------------------------------------------------------------- #
# Cross-sequence isolation
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if `paged_attention_forward` reads K from one sequence and
# V from another — e.g. by calling `read_sequence` twice with different
# seq_ids, by indexing the raw pool with a stale map, or by using `[0]` of
# the pool for V and the sequence's page table for K. Two sequences with
# distinct K/V signatures: any cross-mix produces an output matching
# neither dense reference.
def test_attention_does_not_mix_k_from_one_sequence_with_v_from_another():
    block_size, d_model = 4, 4
    cache = PagedKVCache(num_blocks=4, block_size=block_size, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=4)  # block 0
    cache.allocate_sequence(1, initial_tokens=4)  # block 1

    Q = _tokens(1, d_model, seed=801)
    K0 = _tokens(4, d_model, seed=802)
    V0 = _tokens(4, d_model, seed=803)
    K1 = _tokens(4, d_model, seed=804)
    V1 = _tokens(4, d_model, seed=805)
    cache.write_tokens(0, 0, K0, V0)
    cache.write_tokens(1, 0, K1, V1)

    out0 = paged_attention_forward(Q, cache, 0)
    out1 = paged_attention_forward(Q, cache, 1)

    np.testing.assert_allclose(out0, _dense_reference(Q, K0, V0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, _dense_reference(Q, K1, V1), rtol=1e-5, atol=1e-5)

    # The two outputs must be genuinely different, so a constant-output
    # shortcut cannot pass by matching both.
    assert not np.allclose(out0, out1, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Paged attention
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if attention uses a hidden dense copy of the source arrays
# instead of the cache, ignores the page table, or uses an unstable softmax.
def test_paged_attention_matches_dense_stable_reference():
    block_size, d_model, seq_len = 4, 16, 8
    cache = PagedKVCache(num_blocks=8, block_size=block_size, d_model=d_model)

    for sid in range(8):
        cache.allocate_sequence(sid, initial_tokens=block_size)
    cache.release_sequence(1)
    cache.release_sequence(3)
    cache.allocate_sequence(100, initial_tokens=seq_len)  # 2 blocks -> [1, 3]
    assert cache.owned_blocks(100) == [1, 3]

    Q = _tokens(1, d_model, seed=51)
    K = _tokens(seq_len, d_model, seed=52)
    V = _tokens(seq_len, d_model, seed=53)
    cache.write_tokens(100, 0, K, V)

    out = paged_attention_forward(Q, cache, 100, seq_len=seq_len)
    expected = _dense_reference(Q, K, V)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


# [ROBUST] Fails if seq_len is trusted blindly and attention reads slots past
# the valid length.
def test_attention_rejects_seq_len_beyond_valid_tokens():
    cache = PagedKVCache(num_blocks=2, block_size=4, d_model=4)
    cache.allocate_sequence(0, initial_tokens=8)
    cache.write_tokens(0, 0, _tokens(6, 4, seed=61), _tokens(6, 4, seed=62))
    Q = _tokens(1, 4, seed=63)

    assert paged_attention_forward(Q, cache, 0, seq_len=6).shape == (1, 4)
    with pytest.raises(ValueError):
        paged_attention_forward(Q, cache, 0, seq_len=7)
    with pytest.raises(ValueError):
        paged_attention_forward(Q, cache, 0, seq_len=-1)
    with pytest.raises(TypeError):
        paged_attention_forward(Q, cache, 0, seq_len=3.5)
    with pytest.raises(ValueError):
        paged_attention_forward(np.zeros((2, 4), np.float32), cache, 0)


# [ROBUST] Fails if the output is hardcoded, cached, or computed from stale
# data.
def test_attention_output_depends_on_cached_k_and_v():
    seq_len, d_model = 6, 8
    cache = PagedKVCache(num_blocks=2, block_size=4, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=seq_len)
    Q = _tokens(1, d_model, seed=71)
    K = _tokens(seq_len, d_model, seed=72)
    V = _tokens(seq_len, d_model, seed=73)
    cache.write_tokens(0, 0, K, V)

    out_base = paged_attention_forward(Q, cache, 0)

    cache.write_tokens(0, 3, K[3:4] + 5.0, V[3:4])
    out_k = paged_attention_forward(Q, cache, 0)
    assert not np.allclose(out_base, out_k, rtol=1e-5, atol=1e-5)

    cache.write_tokens(0, 3, K[3:4] + 5.0, V[3:4] + 5.0)
    out_v = paged_attention_forward(Q, cache, 0)
    assert not np.allclose(out_k, out_v, rtol=1e-5, atol=1e-5)


# [ROBUST] Fails if `paged_attention_forward` retains a hidden dense copy of
# the source arrays: directly mutating a pool slot must change the next
# attention call.
def test_attention_reads_from_pool_not_hidden_dense_copy():
    seq_len, d_model, block_size = 4, 4, 2
    cache = PagedKVCache(num_blocks=2, block_size=block_size, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=seq_len)
    Q = _tokens(1, d_model, seed=201)
    K = _tokens(seq_len, d_model, seed=202)
    V = _tokens(seq_len, d_model, seed=203)
    cache.write_tokens(0, 0, K, V)

    out_before = paged_attention_forward(Q, cache, 0)
    physical0 = cache.owned_blocks(0)[0]
    cache.key_pool[physical0, 0] += 100.0

    out_after = paged_attention_forward(Q, cache, 0)
    assert not np.allclose(out_before, out_after, rtol=1e-5, atol=1e-5)


# [ROBUST] Fails if the attention path uses physical block order instead of
# the page table. The mutation-sensitive test is a *redirect* (both logical
# blocks point at the same physical block), then compare attention against a
# dense reference reconstructed from the current cache contents.
def test_page_table_redirect_changes_attention_output():
    block_size, d_model = 2, 4
    cache = PagedKVCache(num_blocks=2, block_size=block_size, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=4)  # blocks [0, 1]

    Q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    K = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    V = np.eye(4, dtype=np.float32)
    cache.write_tokens(0, 0, K, V)

    out_before = paged_attention_forward(Q, cache, 0)
    K_ref, V_ref = cache.read_sequence(0, 4)
    np.testing.assert_array_equal(K_ref, K)
    np.testing.assert_allclose(
        out_before, _dense_reference(Q, K_ref, V_ref), rtol=1e-5, atol=1e-5
    )

    cache.page_tables[0] = [1, 1]

    K_after, V_after = cache.read_sequence(0, 4)
    np.testing.assert_array_equal(
        K_after,
        np.array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    out_after = paged_attention_forward(Q, cache, 0)
    np.testing.assert_allclose(
        out_after, _dense_reference(Q, K_after, V_after), rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(out_before, out_after, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# No aliasing / no caller mutation
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if write_tokens stores references to caller arrays, or if
# read_sequence returns views into the pool.
def test_reads_and_writes_do_not_alias_or_mutate_caller_arrays():
    cache = PagedKVCache(num_blocks=2, block_size=4, d_model=3)
    cache.allocate_sequence(0, initial_tokens=4)
    k = _tokens(4, 3, seed=81)
    v = _tokens(4, 3, seed=82)
    k_snapshot, v_snapshot = k.copy(), v.copy()

    cache.write_tokens(0, 0, k, v)
    k[0, 0] = 999.0
    v[0, 0] = -999.0

    k_out, v_out = cache.read_sequence(0, 4)
    np.testing.assert_array_equal(k_out, k_snapshot)
    np.testing.assert_array_equal(v_out, v_snapshot)

    k_out[1, 1] = -12345.0
    assert not np.shares_memory(k_out, cache.key_pool)

    k_again, v_again = cache.read_sequence(0, 4)
    np.testing.assert_array_equal(k_again, k_snapshot)
    np.testing.assert_array_equal(v_again, v_snapshot)
    assert not np.shares_memory(k_again, k_out)


# --------------------------------------------------------------------------- #
# Release and reuse
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if release forgets to return blocks, forgets to clear
# ownership, or lets a recycled block leak the previous tenant's tokens.
def test_release_returns_block_to_free_pool_without_stale_reads():
    cache = PagedKVCache(num_blocks=4, block_size=2, d_model=2)
    cache.allocate_sequence(10, initial_tokens=2)
    cache.allocate_sequence(11, initial_tokens=6)
    old_k = np.full((2, 2), 42.0, np.float32)
    old_v = np.full((2, 2), -42.0, np.float32)
    cache.write_tokens(10, 0, old_k, old_v)
    assert cache.free_block_count == 0

    released_block = cache.owned_blocks(10)[0]
    cache.release_sequence(10)
    assert cache.free_block_count == 1
    assert cache.block_owner(released_block) is None
    with pytest.raises(KeyError):
        cache.read_sequence(10, 1)

    cache.allocate_sequence(12, initial_tokens=2)
    assert cache.owned_blocks(12) == [released_block]
    new_k = np.full((2, 2), 1.5, np.float32)
    new_v = np.full((2, 2), 2.5, np.float32)
    cache.write_tokens(12, 0, new_k, new_v)

    k_out, v_out = cache.read_sequence(12, 2)
    np.testing.assert_array_equal(k_out, new_k)
    np.testing.assert_array_equal(v_out, new_v)
    with pytest.raises(ValueError):
        cache.read_sequence(12, 3)
    assert cache.free_block_count == 0


# --------------------------------------------------------------------------- #
# Dtype and shapes
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if the pool silently upcasts to float64, if reads lose the
# (n, d_model) shape, or if attention returns a float64 result.
def test_dtype_and_output_shapes_match_contract():
    d_model = 5
    cache = PagedKVCache(num_blocks=4, block_size=2, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=3)

    k64 = np.random.default_rng(0).standard_normal((3, d_model))
    cache.write_tokens(0, 0, k64, k64)

    assert cache.key_pool.dtype == np.float32
    assert cache.value_pool.dtype == np.float32
    k_out, v_out = cache.read_sequence(0, 3)
    assert k_out.dtype == np.float32 and k_out.shape == (3, d_model)
    assert v_out.dtype == np.float32 and v_out.shape == (3, d_model)

    out = paged_attention_forward(np.zeros((1, d_model), np.float32), cache, 0)
    assert out.shape == (1, d_model) and out.dtype == np.float32

    empty_k, empty_v = cache.read_sequence(0, 0)
    assert empty_k.shape == (0, d_model) and empty_v.shape == (0, d_model)


# --------------------------------------------------------------------------- #
# Memory accounting
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if memory_stats reports the wrong payload formula. A
# hardcoded return value that matched the first config would fail on the
# second config.
def test_memory_stats_match_declared_array_shapes():
    cache = PagedKVCache(num_blocks=7, block_size=5, d_model=3)
    stats = cache.memory_stats()
    assert stats["kv_pool_bytes"] == 2 * 7 * 5 * 3 * 4
    assert stats["blocks_free"] == 7 and stats["blocks_allocated"] == 0
    assert stats["dtype"] == "float32"

    cache.allocate_sequence(0, initial_tokens=6)
    assert cache.allocated_capacity(0) == 10
    assert cache.allocated_capacity(0) - cache.sequence_length(0) == 10
    cache.write_tokens(0, 0, _tokens(6, 3, seed=91), _tokens(6, 3, seed=92))
    assert cache.allocated_capacity(0) - cache.sequence_length(0) == 4

    # Second config: a hardcoded return value that matched the 7/5/3 case
    # would fail here.
    cache2 = PagedKVCache(num_blocks=3, block_size=2, d_model=4)
    assert cache2.memory_stats()["kv_pool_bytes"] == 2 * 3 * 2 * 4 * 4  # 192


# --------------------------------------------------------------------------- #
# Numerical stability
# --------------------------------------------------------------------------- #
# [ROBUST] Fails if softmax is computed as exp(scores) / sum(exp(scores)):
# the scores reach +/-500, exp(500) overflows float32 and the ratio becomes
# NaN. Max-subtraction must yield a finite, exactly-normalised result.
def test_softmax_is_stable_for_large_scores():
    d_model = 4
    cache = PagedKVCache(num_blocks=1, block_size=4, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=4)
    K = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    V = np.eye(4, dtype=np.float32)
    Q = np.array([[1000.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    cache.write_tokens(0, 0, K, V)

    out = paged_attention_forward(Q, cache, 0)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(
        out, np.array([[0.5, 0.0, 0.5, 0.0]], dtype=np.float32), rtol=0, atol=1e-6
    )


# [ROBUST] Fails if attention silently propagates NaN/inf from Q instead of
# rejecting it.
def test_attention_rejects_non_finite_query():
    d_model = 4
    cache = PagedKVCache(num_blocks=1, block_size=4, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=4)
    cache.write_tokens(
        0, 0, np.zeros((4, d_model), np.float32), np.zeros((4, d_model), np.float32)
    )

    with pytest.raises(ValueError):
        paged_attention_forward(np.full((1, d_model), np.nan, np.float32), cache, 0)
    with pytest.raises(ValueError):
        paged_attention_forward(np.full((1, d_model), np.inf, np.float32), cache, 0)


# [ROBUST] Fails if attention returns NaN when finite Q/K produce an
# overflowing dot product in float32.
def test_attention_raises_on_score_overflow():
    d_model = 4
    cache = PagedKVCache(num_blocks=1, block_size=4, d_model=d_model)
    cache.allocate_sequence(0, initial_tokens=4)
    Q = np.full((1, d_model), 1e20, np.float32)
    K = np.full((4, d_model), 1e20, np.float32)
    V = np.zeros((4, d_model), np.float32)
    cache.write_tokens(0, 0, K, V)

    with pytest.raises(FloatingPointError):
        paged_attention_forward(Q, cache, 0)


# --------------------------------------------------------------------------- #
# Property tests (Hypothesis; defined only when installed, otherwise stubbed
# to skip so the count of "1 skipped" is stable across environments).
# --------------------------------------------------------------------------- #
if _HAS_HYPOTHESIS:

    @given(
        block_size=st.integers(min_value=1, max_value=5),
        d_model=st.integers(min_value=1, max_value=4),
        n_tokens=st.integers(min_value=1, max_value=25),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_write_read_roundtrip(block_size, d_model, n_tokens, seed):
        n_blocks = (n_tokens + block_size - 1) // block_size
        cache = PagedKVCache(
            num_blocks=n_blocks, block_size=block_size, d_model=d_model
        )
        cache.allocate_sequence(0, initial_tokens=n_tokens)
        assert cache.free_block_count == 0

        rng = np.random.default_rng(seed)
        k = rng.standard_normal((n_tokens, d_model)).astype(np.float32)
        v = rng.standard_normal((n_tokens, d_model)).astype(np.float32)
        cache.write_tokens(0, 0, k, v)

        k_out, v_out = cache.read_sequence(0, n_tokens)
        np.testing.assert_array_equal(k_out, k)
        np.testing.assert_array_equal(v_out, v)

        with pytest.raises(ValueError):
            cache.read_sequence(0, n_tokens + 1)
else:

    def test_property_write_read_roundtrip():
        pytest.skip("optional dependency 'hypothesis' is not installed")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main(["-v", __file__]))


# Not tested (and why):
# - GPU kernels, HBM bandwidth and the published "~4% waste / 4x concurrency"
#   figures: this module has no kernel and no CUDA path, so asserting those
#   numbers would be fiction rather than verification.
# - True concurrency / allocation races: the cache is single-threaded with
#   no locking; a threading test would exercise the GIL, not a
#   memory-ordering guarantee.
# - Multi-layer / multi-head pool layouts (num_layers, num_heads, head_dim):
#   deliberately out of scope; the pool is (blocks, block_size, d_model).
# - Dynamic growth of a live sequence past its allocation: not implemented.
# - Copy-on-write / prefix sharing: not implemented; ownership is exclusive.
# - Block-by-block (fused-style) attention: not implemented.
# - Non-float32 pools (float64 / float16 / bfloat16): dtype is fixed by design.
# - Exhaustive physical-block permutation coverage: only hand-built witnesses
#   are tested; the Hypothesis property for permutations is documented in the
#   comments but not encoded.
# - Direct corruption of the private `_block_owner` array: outside the
#   supported API. If a block's owner changes, `_validated_pages` rejects
#   the next read/write of the previous owner, but a later `release_sequence`
#   for the new "owner" would free that block into the free list while the
#   previous owner's page table still references it. A correct fix would
#   need reference counting, which is out of scope.
# - Repair of a corrupted page table on a live sequence: `release_sequence`
#   raises and leaves the block orphaned; the caller has to fix the table.
# - Score-overflow warning suppression: no test asserts that the warning is
#   gone (that would couple the suite to NumPy internals). The behavior the
#   test asserts is the raise, not the absence of a warning.
# - Non-integer page-table entries that raise TypeError: `_validated_pages`
#   converts a non-integer entry to TypeError, but no test exercises that
#   path. Adding `cache.page_tables[0][0] = "abc"; with pytest.raises(TypeError): ...`
#   is a future hardening test; the current `test_corrupted_page_table_entry_raises`
#   only covers integer bounds and ownership, both of which are ValueError.
