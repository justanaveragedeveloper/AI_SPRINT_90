"""Mutation-specific test suite for day63_disaggregated_kv."""

import inspect

import numpy as np
import pytest
from day63_disaggregated_kv import (
    BlockId,
    DisaggregatedClusterManager,
    RemoteKVNode,
    ideal_transfer_seconds,
    kv_cache_bytes,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers.
# No RNG: deterministic payloads make failure output readable, and the
# module under test is deterministic by contract.
# ---------------------------------------------------------------------------


def _make_arange_kv_data(num_tokens, d_model, dtype=np.float32):
    """Build deterministic K/V tensors from arange for reproducible failure output.

    K = 0..n*d-1 reshaped; V = K + 0.5. Exact, reproducible, no RNG.
    """
    k_data = (
        np.arange(num_tokens * d_model, dtype=np.float64)
        .reshape(num_tokens, d_model)
        .astype(dtype)
    )
    return k_data, k_data + dtype(0.5)


def _make_prefilled_cluster(
    num_tokens=8, block_size=4, d_model=16, num_prefill=2, num_decode=2
):
    """Return (cluster, seq_id, prefill_node, decode_node, blocks, k_data, v_data)."""
    cluster = DisaggregatedClusterManager(num_prefill, num_decode, block_size, d_model)
    seq_id = "req_99"
    prefill_node = cluster.route_prefill(seq_id)
    decode_node = cluster.route_decode(seq_id)
    k_data, v_data = _make_arange_kv_data(num_tokens, d_model)
    blocks = prefill_node.write_prefill_blocks(seq_id, k_data, v_data)
    return cluster, seq_id, prefill_node, decode_node, blocks, k_data, v_data


# ===========================================================================
# Routing
# ===========================================================================


def test_route_prefill_repeated_calls_return_the_same_sticky_node():
    """[ROBUST] Catches: routing ignores the sticky table, so a live request
    migrates mid-flight to a different prefill node on the next call.
    """
    cluster = DisaggregatedClusterManager(3, 3, 4, 8)
    first_assignment = cluster.route_prefill("req_0")
    for _ in range(5):
        assert cluster.route_prefill("req_0") is first_assignment


def test_sticky_routing_hit_does_not_advance_the_round_robin_cursor():
    """[ROBUST] Catches: `cursor += 1` hoisted outside the `if node is None`.

    A "repeated calls return the same node" test alone cannot see this bug,
    because the sticky table still returns the cached node. The tell is the
    *next* distinct seq_id skipping a node in the round-robin sequence.
    """
    cluster = DisaggregatedClusterManager(3, 3, 4, 8)
    assert cluster.route_prefill("a").node_id == 0
    for _ in range(4):
        cluster.route_prefill("a")  # must not burn nodes 1 and 2
    assert cluster.route_prefill("b").node_id == 1


def test_route_prefill_round_robin_matches_the_exact_documented_sequence():
    """[ROBUST] Catches: always-node-0, random, off-by-one, non-round-robin.

    The `len(assigned_node_ids) > 1` assertion cannot distinguish
    [0,0,0,1,1,1,1] from true round-robin. Only an exact sequence does.
    """
    cluster = DisaggregatedClusterManager(3, 3, 4, 8)
    assigned_node_ids = [
        cluster.route_prefill(f"request_{i}").node_id for i in range(7)
    ]
    assert assigned_node_ids == [0, 1, 2, 0, 1, 2, 0]


def test_route_decode_round_robin_starts_at_the_first_decode_node():
    """[ROBUST] Catches: decode cursor seeded from prefill pool size, or
    the decode pool reused for prefill routing."""
    cluster = DisaggregatedClusterManager(3, 2, 4, 8)
    assigned_node_ids = [cluster.route_decode(f"request_{i}").node_id for i in range(5)]
    assert assigned_node_ids == [3, 4, 3, 4, 3]


def test_prefill_and_decode_cursors_advance_independently():
    """[ROBUST] Catches: one shared counter drives both pools.

    Under that bug the 2nd prefill call lands on node 2, not node 1.
    """
    cluster = DisaggregatedClusterManager(3, 3, 4, 8)
    assert cluster.route_prefill("a").node_id == 0
    assert cluster.route_decode("a").node_id == 3
    assert cluster.route_prefill("b").node_id == 1
    assert cluster.route_decode("b").node_id == 4


def test_prefill_and_decode_pools_are_disjoint_node_instances():
    """[ROBUST] Catches: both roles resolve to the same node objects."""
    cluster = DisaggregatedClusterManager(2, 2, 4, 8)
    prefill_node = cluster.route_prefill("a")
    decode_node = cluster.route_decode("a")
    assert prefill_node is not decode_node
    assert (prefill_node.role, decode_node.role) == ("prefill", "decode")
    assert all(decode_node is not n for n in cluster.prefill_nodes)


def test_node_ids_are_unique_across_both_pools():
    """[ROBUST] Catches: decode IDs restart at 0 and collide with prefill IDs."""
    cluster = DisaggregatedClusterManager(4, 4, 4, 8)
    all_node_ids = [n.node_id for n in cluster.prefill_nodes] + [
        n.node_id for n in cluster.decode_nodes
    ]
    assert len(all_node_ids) == len(set(all_node_ids)) == 8


# ===========================================================================
# Constructor / node / id contracts
# ===========================================================================


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_prefill_nodes": True,
            "num_decode_nodes": 1,
            "block_size": 4,
            "d_model": 8,
        },
        {
            "num_prefill_nodes": 1,
            "num_decode_nodes": True,
            "block_size": 4,
            "d_model": 8,
        },
        {
            "num_prefill_nodes": 1,
            "num_decode_nodes": 1,
            "block_size": True,
            "d_model": 8,
        },
        {
            "num_prefill_nodes": 1,
            "num_decode_nodes": 1,
            "block_size": 4,
            "d_model": True,
        },
    ],
)
def test_constructor_rejects_bool_for_every_positive_int_param(kwargs):
    """[ROBUST] Catches: `isinstance(x, int)` without bool exclusion, on any
    of the four positive-int params (an earlier version only tested prefill).
    """
    with pytest.raises(TypeError):
        DisaggregatedClusterManager(**kwargs)


def test_constructor_rejects_non_positive_block_size_and_d_model():
    """[ROBUST] Catches: zero/negative geometry accepted (would later raise
    ZeroDivisionError deep inside partitioning instead of at construction)."""
    with pytest.raises(ValueError):
        DisaggregatedClusterManager(1, 1, 0, 8)
    with pytest.raises(ValueError):
        DisaggregatedClusterManager(1, 1, 4, -16)


def test_constructor_rejects_non_int_block_size_and_d_model():
    """[ROBUST] Catches: float or string geometry silently accepted."""
    with pytest.raises(TypeError):
        DisaggregatedClusterManager(1, 1, 4.0, 16)
    with pytest.raises(TypeError):
        DisaggregatedClusterManager(1, 1, "4", 16)
    with pytest.raises(TypeError):
        DisaggregatedClusterManager(1, 1, 4, 16.0)


def test_remote_kv_node_rejects_invalid_role():
    """[ROBUST] Catches: role accepted without validation."""
    with pytest.raises(ValueError, match="role"):
        RemoteKVNode(0, "encoder", 4, 16)


def test_remote_kv_node_rejects_negative_node_id():
    """[ROBUST] Catches: negative node_id accepted."""
    with pytest.raises(ValueError):
        RemoteKVNode(-1, "prefill", 4, 16)


def test_node_configuration_is_immutable_after_construction():
    """[ROBUST] Catches: post-construction mutation of node config.

    Mutating block_size changes how `_copy_token_prefix` partitions stored
    blocks; mutating node_id invalidates previously issued BlockIds;
    mutating role bypasses the prefill-only write guard. All three are
    prevented by read-only properties + __slots__.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    for attribute_name, bad_value in [
        ("node_id", 99),
        ("role", "decode"),
        ("block_size", 8),
        ("d_model", 32),
    ]:
        with pytest.raises(AttributeError):
            setattr(node, attribute_name, bad_value)


def test_route_rejects_empty_or_non_string_sequence_id():
    """[ROBUST] Catches: '' or None silently shared as a routing key."""
    cluster = DisaggregatedClusterManager(2, 2, 4, 8)
    with pytest.raises(ValueError):
        cluster.route_prefill("")
    with pytest.raises(TypeError):
        cluster.route_decode(None)


def test_write_rejects_invalid_direct_seq_id():
    """[ROBUST] Catches: write accepts '' or non-str seq_id directly, bypassing
    the routing-layer validation."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(4, 16)
    with pytest.raises(ValueError):
        node.write_prefill_blocks("", k_data, v_data)
    with pytest.raises(TypeError):
        node.write_prefill_blocks(None, k_data, v_data)


def test_read_rejects_invalid_direct_seq_id():
    """[ROBUST] Catches: read accepts '' or non-str seq_id directly."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    with pytest.raises(ValueError):
        node.read_sequence("")
    with pytest.raises(TypeError):
        node.read_sequence(None)


# ===========================================================================
# Write
# ===========================================================================


def test_write_rejects_mismatched_k_and_v_token_counts():
    """[ROBUST] Catches: K and V validated independently, stored unpaired.

    A 7-token V stored against an 8-token K would only explode at decode
    time; the assertion checks that nothing was stored either.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    with pytest.raises(ValueError):
        node.write_prefill_blocks(
            "s",
            np.zeros((8, 16), np.float32),
            np.zeros((7, 16), np.float32),
        )
    with pytest.raises(KeyError):
        node.read_sequence("s")


def test_write_rejects_mismatched_k_and_v_dtypes():
    """[ROBUST] Catches: K/V dtype divergence; without the rejection an empty
    sequence would be read back with a single stored dtype on both sides."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 8)
    node = cluster.route_prefill("s")
    k_data = np.zeros((4, 8), dtype=np.float32)
    v_data = np.zeros((4, 8), dtype=np.float64)
    with pytest.raises(TypeError):
        node.write_prefill_blocks("s", k_data, v_data)
    with pytest.raises(KeyError):
        node.read_sequence("s")


def test_decode_node_rejects_prefill_write():
    """[ROBUST] Catches: architectural boundary bypassed by direct write to a
    decode node without going through transfer."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 8)
    decode_node = cluster.route_decode("s")
    k_data, v_data = _make_arange_kv_data(4, 8)
    with pytest.raises(ValueError, match="prefill"):
        decode_node.write_prefill_blocks("s", k_data, v_data)


def test_write_rejects_rank_three_tensors():
    """[ROBUST] Catches: validation checks only the final dimension."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    bad_tensor = np.zeros((2, 4, 16), np.float32)
    with pytest.raises(ValueError):
        node.write_prefill_blocks("s", bad_tensor, bad_tensor)


def test_write_rejects_wrong_d_model_width():
    """[ROBUST] Catches: d_model inferred from caller instead of node config."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    bad_tensor = np.zeros((4, 15), np.float32)
    with pytest.raises(ValueError):
        node.write_prefill_blocks("s", bad_tensor, bad_tensor)


def test_write_rejects_object_dtype_without_coercing_it():
    """[ROBUST] Catches: lenient `np.asarray(x, dtype=float)` silently converting.

    Silent conversion is how a precision-loss bug reaches production
    undetected. The node must reject, not coerce.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_tensor = np.empty((4, 16), dtype=object)
    k_tensor[:] = 1.0
    v_tensor = np.empty((4, 16), dtype=object)
    v_tensor[:] = 1.0
    with pytest.raises(TypeError):
        node.write_prefill_blocks("s", k_tensor, v_tensor)


def test_write_rejects_bool_dtype():
    """[ROBUST] Catches: dtype kind 'b' passing a loose numeric check."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    bad_tensor = np.zeros((4, 16), dtype=bool)
    with pytest.raises(TypeError):
        node.write_prefill_blocks("s", bad_tensor, bad_tensor)


def test_write_preserves_float32_without_silent_promotion():
    """[ROBUST] Catches: storage via default-dtype np.asarray (memory double).

    Doubling the memory of the payload this whole exercise is about.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16, dtype=np.float32)
    node.write_prefill_blocks("s", k_data, v_data)
    k_returned, v_returned = node.read_sequence("s")
    assert k_returned.dtype == np.float32
    assert v_returned.dtype == np.float32


def test_write_returns_ceil_tokens_over_block_size_block_ids():
    """[ROBUST] Catches: floor division dropping the final partial block.

    10 tokens / block_size=4 must produce 3 block IDs, not 2.
    """
    cluster = DisaggregatedClusterManager(1, 1, block_size=4, d_model=16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(10, 16)
    blocks = node.write_prefill_blocks("s", k_data, v_data)
    assert len(blocks) == 3
    assert [block.block_index for block in blocks] == [0, 1, 2]


def test_partial_final_block_is_not_padded_or_exposed_as_real_tokens():
    """[ROBUST] Catches: zero-padding the tail block, phantom tokens on read.

    Zero-padding would turn a 10-token sequence into a 12-token one and feed
    two phantom tokens to the decoder.
    """
    cluster = DisaggregatedClusterManager(1, 1, block_size=4, d_model=16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(10, 16)
    node.write_prefill_blocks("s", k_data, v_data)

    k_all, v_all = node.read_sequence("s")
    assert k_all.shape == (10, 16)
    np.testing.assert_array_equal(k_all, k_data)
    np.testing.assert_array_equal(v_all, v_data)
    with pytest.raises(ValueError):
        node.read_sequence("s", 12)  # 12 is only readable if padding leaked


def test_write_defensive_copy_isolates_cache_from_caller_mutation():
    """[ROBUST] Catches: node retains references to the caller's arrays."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16)
    expected_values = k_data.copy()
    node.write_prefill_blocks("s", k_data, v_data)

    k_data[:] = -1.0
    v_data[:] = -1.0

    k_returned, _ = node.read_sequence("s")
    np.testing.assert_array_equal(k_returned, expected_values)


def test_write_zero_tokens_produces_no_blocks_and_no_phantom_read():
    """[ROBUST] Catches: ceil(0/B) computed as 1, phantom empty block."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    empty_k = np.zeros((0, 16), np.float32)
    empty_v = np.zeros((0, 16), np.float32)
    assert node.write_prefill_blocks("s", empty_k, empty_v) == []
    k_returned, v_returned = node.read_sequence("s", 0)
    assert k_returned.shape == (0, 16) and v_returned.shape == (0, 16)
    with pytest.raises(ValueError):
        node.read_sequence("s", 1)


def test_write_replaces_existing_sequence_for_the_same_seq_id():
    """[ROBUST] Catches: append-on-rewrite (block count doubles)."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data_v1, v_data_v1 = _make_arange_kv_data(8, 16)
    k_data_v2, v_data_v2 = _make_arange_kv_data(8, 16)
    k_data_v2 = k_data_v2 + 1000.0

    node.write_prefill_blocks("s", k_data_v1, v_data_v1)
    node.write_prefill_blocks("s", k_data_v2, v_data_v2)

    k_returned, _ = node.read_sequence("s", 8)
    np.testing.assert_array_equal(k_returned, k_data_v2)


# ===========================================================================
# Read
# ===========================================================================


def test_read_sequence_returns_exactly_the_requested_prefix_length():
    """[ROBUST] Catches: bound derived from block count, not token count."""
    cluster = DisaggregatedClusterManager(1, 1, block_size=4, d_model=16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(10, 16)
    node.write_prefill_blocks("s", k_data, v_data)

    k_returned, v_returned = node.read_sequence("s", 5)
    assert k_returned.shape == (5, 16) and v_returned.shape == (5, 16)
    np.testing.assert_array_equal(k_returned, k_data[:5])
    np.testing.assert_array_equal(v_returned, v_data[:5])


def test_read_prefix_does_not_concatenate_more_blocks_than_needed(monkeypatch):
    """[ROBUST] Catches: read_sequence concatenates all stored blocks, then slices.

    With 5 blocks stored and 5 tokens requested (block_size=4), only 2 blocks
    are touched (1 full + 1 partial). The original implementation
    concatenated all 5. The spy intercepts np.concatenate calls and asserts
    that the maximum block count passed to any single call is <= 2.

    Only catches np.concatenate-based mutations. A mutated implementation
    that used np.stack or np.vstack would not trigger the spy; verify those
    separately if the implementation ever changes.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(20, 16)  # 5 blocks
    node.write_prefill_blocks("s", k_data, v_data)

    concatenated_block_counts: list[int] = []
    original_concatenate = np.concatenate

    def spy_concatenate(arrays, axis=0):
        if isinstance(arrays, (list, tuple)):
            concatenated_block_counts.append(len(arrays))
        return original_concatenate(arrays, axis=axis)

    monkeypatch.setattr(np, "concatenate", spy_concatenate)

    k_returned, _ = node.read_sequence("s", 5)
    np.testing.assert_array_equal(k_returned, k_data[:5])
    assert max(concatenated_block_counts) <= 2, (
        f"concatenated {max(concatenated_block_counts)} blocks for a 5-token prefix"
    )


def test_read_sequence_with_zero_tokens_on_nonempty_returns_empty_arrays():
    """[ROBUST] Catches: `k_all[:0]` returning the whole array (slice bug)."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16)
    node.write_prefill_blocks("s", k_data, v_data)

    k_returned, v_returned = node.read_sequence("s", 0)
    assert k_returned.shape == (0, 16) and v_returned.shape == (0, 16)


def test_read_sequence_with_none_returns_full_sequence():
    """[ROBUST] Catches: `None` treated as 0 instead of full length."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(10, 16)
    node.write_prefill_blocks("s", k_data, v_data)

    k_returned, v_returned = node.read_sequence("s")  # default num_tokens=None
    np.testing.assert_array_equal(k_returned, k_data)
    np.testing.assert_array_equal(v_returned, v_data)


def test_read_sequence_preserves_token_order_across_block_boundaries():
    """[ROBUST] Catches: reverse/set-order block concatenation.

    10 tokens at block_size=3 produces blocks of 3/3/3/1; equality over the
    full sequence pins the order.
    """
    cluster = DisaggregatedClusterManager(1, 1, block_size=3, d_model=4)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(10, 4)
    node.write_prefill_blocks("s", k_data, v_data)

    k_returned, v_returned = node.read_sequence("s")
    np.testing.assert_array_equal(k_returned, k_data)
    np.testing.assert_array_equal(v_returned, v_data)


def test_read_sequence_rejects_negative_token_count_instead_of_tail_slicing():
    """[ROBUST] Catches: num_tokens passed straight into `k_all[:num_tokens]`.

    `k_all[-3]` silently returns the *last* three tokens — a valid-looking
    array of the wrong content.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16)
    node.write_prefill_blocks("s", k_data, v_data)
    with pytest.raises(ValueError):
        node.read_sequence("s", -3)


def test_read_sequence_rejects_bool_token_count():
    """[ROBUST] Catches: `True` accepted as 1 token (bool subclasses int)."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16)
    node.write_prefill_blocks("s", k_data, v_data)
    with pytest.raises(TypeError):
        node.read_sequence("s", True)


def test_read_sequence_rejects_oversized_and_float_token_counts():
    """[ROBUST] Catches: unbounded count, float silently truncated."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(8, 16)
    node.write_prefill_blocks("s", k_data, v_data)
    with pytest.raises(ValueError):
        node.read_sequence("s", 9)  # stored length is 8
    with pytest.raises(TypeError):
        node.read_sequence("s", 4.0)


def test_read_sequence_returns_arrays_that_do_not_alias_stored_blocks():
    """[ROBUST] Catches: single-block fast path returning a view.

    A decode-time in-place op on a returned view would corrupt the cache.
    """
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    node = cluster.route_prefill("s")
    k_data, v_data = _make_arange_kv_data(4, 16)  # exactly one block
    node.write_prefill_blocks("s", k_data, v_data)

    k_returned, _ = node.read_sequence("s", 4)
    k_returned[0, 0] = -999.0
    k_again, _ = node.read_sequence("s", 4)
    assert k_again[0, 0] == k_data[0, 0]


# ===========================================================================
# Transfer
# ===========================================================================


def test_transfer_makes_destination_independently_readable_with_matching_payload():
    """[ROBUST] Catches: K/V swapped in _snapshot, blocks reordered, block dropped."""
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    k_returned, v_returned = decode_node.read_sequence(seq_id, num_tokens=8)
    np.testing.assert_array_equal(k_returned, k_data)
    np.testing.assert_array_equal(v_returned, v_data)
    assert k_returned.dtype == k_data.dtype
    assert v_returned.dtype == v_data.dtype
    assert k_returned.shape == k_data.shape
    assert v_returned.shape == v_data.shape


def test_transfer_snapshot_copies_internal_block_storage():
    """[ROBUST] Catches: `_snapshot` returns source block references
    (`entry.k_blocks[i]`, `.view()`, or `np.asarray`) instead of copies,
    which would alias destination storage to source storage.

    White-box because `read_sequence` copies, masking internal aliasing at
    the API layer.
    """
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    source_block = prefill_node._sequences[seq_id].k_blocks[0]
    destination_block = decode_node._sequences[seq_id].k_blocks[0]
    assert not np.shares_memory(source_block, destination_block)


def test_snapshot_failure_leaves_destination_unchanged(monkeypatch):
    """[ROBUST] Catches: publish-before-copy, or rollback that clears the old
    destination before the new payload is fully built.

    Validation-phase failure is covered by
    test_failed_replacement_preserves_the_previous_destination_value. This
    test exercises the snapshot-phase failure path: an exception raised
    mid-copy of the payload must still leave the destination's previous
    value intact.
    """
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    k_data_v2, v_data_v2 = _make_arange_kv_data(8, 16)
    k_data_v2 = k_data_v2 + 1000.0
    blocks_v2 = prefill_node.write_prefill_blocks(seq_id, k_data_v2, v_data_v2)

    def failing_snapshot(_self, _seq_id, _block_count):
        raise MemoryError("simulated allocation failure during snapshot")

    monkeypatch.setattr(RemoteKVNode, "_snapshot", failing_snapshot)

    with pytest.raises(MemoryError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks_v2)

    k_returned, _ = decode_node.read_sequence(seq_id, 8)
    np.testing.assert_array_equal(k_returned, k_data)


def test_transfer_leaves_the_prefill_source_intact():
    """[ROBUST] Catches: transfer pops or deletes source blocks."""
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    k_source, v_source = prefill_node.read_sequence(seq_id, 8)
    np.testing.assert_array_equal(k_source, k_data)
    np.testing.assert_array_equal(v_source, v_data)


def test_transfer_accepts_reordered_block_ids_and_reconstructs_token_order():
    """[ROBUST] Catches: copy in caller order, unsorted."""
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, [blocks[1], blocks[0]])

    k_returned, _ = decode_node.read_sequence(seq_id, 8)
    np.testing.assert_array_equal(k_returned, k_data)


def test_transfer_rejects_missing_block_without_publishing_a_partial_sequence():
    """[ROBUST] Catches: append-as-you-go without a completeness check.

    The destination then serves a sequence that is silently half-length.
    """
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks[:1])
    with pytest.raises(KeyError):
        decode_node.read_sequence(seq_id)


def test_transfer_rejects_empty_block_list_for_a_nonempty_sequence():
    """[ROBUST] Catches: `set([]) == expected` comparison skipped for empty input."""
    cluster, seq_id, prefill_node, decode_node, _blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, [])
    with pytest.raises(KeyError):
        decode_node.read_sequence(seq_id)


def test_transfer_rejects_unknown_sequence_on_source():
    """[ROBUST] Catches: `_block_count` KeyError leaking, or `[] == []` passing."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 16)
    prefill_node = cluster.route_prefill("ghost")
    decode_node = cluster.route_decode("ghost")
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache("ghost", prefill_node, decode_node, [])
    with pytest.raises(KeyError):
        decode_node.read_sequence("ghost")


def test_transfer_rejects_non_sequence_blocks_arg():
    """[ROBUST] Catches: dict / set / string silently accepted as a block list."""
    cluster, seq_id, prefill_node, decode_node, _blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(TypeError):
        cluster.transfer_kv_cache(
            seq_id, prefill_node, decode_node, {"0": "not a block"}
        )
    with pytest.raises(TypeError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, "abc")


def test_transfer_rejects_duplicated_block_ids():
    """[ROBUST] Catches: `set(blocks) == expected` hiding a duplicate."""
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(
            seq_id, prefill_node, decode_node, blocks + [blocks[0]]
        )
    with pytest.raises(KeyError):
        decode_node.read_sequence(seq_id)


def test_transfer_rejects_block_ids_minted_for_another_node():
    """[ROBUST] Catches: ownership checked by index only.

    A hand-forged BlockId with the right seq_id and index but the wrong
    node_id must be refused: block identity is (seq, node, index), not
    (index).
    """
    cluster, seq_id, prefill_node, decode_node, _blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    forged_blocks = [
        BlockId(
            seq_id=seq_id,
            node_id=prefill_node.node_id + 1,
            block_index=block_index,
        )
        for block_index in range(2)
    ]
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, forged_blocks)


def test_transfer_rejects_block_ids_belonging_to_another_sequence():
    """[ROBUST] Catches: blocks accepted because valid *somewhere* on the node
    rather than for this exact sequence."""
    cluster, seq_id, prefill_node, decode_node, _blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    k_data_other, v_data_other = _make_arange_kv_data(8, 16)
    other_blocks = prefill_node.write_prefill_blocks(
        "other_req", k_data_other, v_data_other
    )
    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, other_blocks)
    with pytest.raises(KeyError):
        decode_node.read_sequence(seq_id)


def test_transfer_rejects_raw_integer_indices_as_block_ids():
    """[ROBUST] Catches: bare int accepted as block identity (guessable)."""
    cluster, seq_id, prefill_node, decode_node, _blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(TypeError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, [0, 1])


def test_transfer_rejects_a_prefill_node_from_another_cluster():
    """[ROBUST] Catches: membership by node_id instead of identity.

    Both clusters number their first prefill node 0, so an id-based check
    passes a foreign node straight through. The frozenset membership uses
    identity (RemoteKVNode does not override __eq__/__hash__).
    """
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    other_cluster = DisaggregatedClusterManager(2, 2, 4, 16)
    foreign_prefill_node = other_cluster.route_prefill(seq_id)
    assert foreign_prefill_node.node_id == prefill_node.node_id

    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, foreign_prefill_node, decode_node, blocks)


def test_transfer_rejects_a_decode_node_from_another_cluster():
    """[ROBUST] Catches: only the source node is membership-checked."""
    cluster, seq_id, prefill_node, _decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    other_cluster = DisaggregatedClusterManager(2, 2, 4, 16)
    foreign_decode_node = other_cluster.route_decode(seq_id)

    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, foreign_decode_node, blocks)


def test_transfer_rejects_role_swapped_source_and_destination():
    """[ROBUST] Catches: role inferred from node_id ordering, not checked."""
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    with pytest.raises(ValueError):
        # Decode node as source:
        cluster.transfer_kv_cache(seq_id, decode_node, prefill_node, blocks)
    with pytest.raises(ValueError):
        # Prefill node as destination:
        cluster.transfer_kv_cache(seq_id, prefill_node, prefill_node, blocks)


def test_failed_replacement_preserves_the_previous_destination_value():
    """[ROBUST] Catches: destination cleared before validation completes.

    The prior valid entry must survive a failed replacement byte-for-byte.
    """
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    k_data_v2, v_data_v2 = _make_arange_kv_data(8, 16)
    k_data_v2 = k_data_v2 + 1000.0
    blocks_v2 = prefill_node.write_prefill_blocks(seq_id, k_data_v2, v_data_v2)

    with pytest.raises(ValueError):
        cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks_v2[:1])

    k_returned, _ = decode_node.read_sequence(seq_id, 8)
    np.testing.assert_array_equal(k_returned, k_data)  # still the OLD value


def test_duplicate_transfer_is_idempotent_replacement():
    """[CONTRACT] Catches: append-on-rewrite (shape doubles) or reject.

    The documented policy is idempotent replacement; a second transfer of the
    same block set must leave one entry with identical content.
    """
    cluster, seq_id, prefill_node, decode_node, blocks, k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)

    k_returned, _ = decode_node.read_sequence(seq_id)
    assert k_returned.shape == (8, 16)  # not (16, 16)
    np.testing.assert_array_equal(k_returned, k_data)


def test_zero_token_sequence_transfers_without_phantom_blocks():
    """[ROBUST] Catches: empty payload rejected, ceil(0/B)=1 phantom block."""
    cluster = DisaggregatedClusterManager(1, 1, 4, 8)
    prefill_node = cluster.route_prefill("e")
    decode_node = cluster.route_decode("e")
    empty_tensor = np.zeros((0, 8), np.float32)
    blocks = prefill_node.write_prefill_blocks("e", empty_tensor, empty_tensor)
    assert blocks == []

    cluster.transfer_kv_cache("e", prefill_node, decode_node, blocks)

    k_returned, v_returned = decode_node.read_sequence("e", 0)
    assert k_returned.shape == (0, 8) and v_returned.shape == (0, 8)


# ===========================================================================
# Lifecycle
# ===========================================================================


def test_release_request_clears_routing_so_next_assignment_advances():
    """[ROBUST] Catches: release_request leaves sticky entry, blocking re-route."""
    cluster = DisaggregatedClusterManager(2, 2, 4, 8)
    first_assignment = cluster.route_prefill("a")
    cluster.release_request("a")
    second_assignment = cluster.route_prefill("a")
    assert second_assignment is not first_assignment
    assert second_assignment.node_id == 1  # cursor advanced


def test_release_request_frees_node_storage():
    """[ROBUST] Catches: release_request clears routing but leaks KV data."""
    cluster, seq_id, prefill_node, decode_node, blocks, _k_data, _v_data = (
        _make_prefilled_cluster()
    )
    cluster.transfer_kv_cache(seq_id, prefill_node, decode_node, blocks)
    cluster.release_request(seq_id)
    with pytest.raises(KeyError):
        prefill_node.read_sequence(seq_id)
    with pytest.raises(KeyError):
        decode_node.read_sequence(seq_id)


def test_release_request_is_idempotent():
    """[ROBUST] Catches: second release on the same id raises KeyError."""
    cluster = DisaggregatedClusterManager(2, 2, 4, 8)
    cluster.route_prefill("a")
    cluster.route_decode("a")
    cluster.release_request("a")
    cluster.release_request("a")  # no error


def test_release_request_accepts_unknown_seq_id():
    """[ROBUST] Catches: KeyError on cleanup of a never-seen id."""
    cluster = DisaggregatedClusterManager(2, 2, 4, 8)
    cluster.release_request("never_seen")  # no-op


# ===========================================================================
# Latency
# ===========================================================================


def test_kv_cache_bytes_counts_k_and_v_and_honours_element_size():
    """[ROBUST] Catches: missing factor 2 (K+V) or missing element-size factor.

    The commonly quoted `2*B*d_model*L` is a byte count only when `s == 1`;
    with fp16 it understates by 2x. Each scaling axis is checked separately.
    """
    base = kv_cache_bytes(num_tokens=4, num_layers=2, d_model=16, element_size_bytes=2)
    assert base == 2 * 4 * 2 * 16 * 2
    assert kv_cache_bytes(8, 2, 16, 2) == 2 * base  # linear in tokens
    assert kv_cache_bytes(4, 4, 16, 2) == 2 * base  # linear in layers
    assert kv_cache_bytes(4, 2, 16, 4) == 2 * base  # linear in element size
    assert kv_cache_bytes(4, 2, 16, 1) == base // 2  # fp16 -> int8 halves


def test_kv_cache_bytes_rejects_bool_and_non_positive_args():
    """[ROBUST] Catches: bool silently accepted as int; zero/negative geometry."""
    with pytest.raises(TypeError):
        kv_cache_bytes(True, 2, 16, 4)
    with pytest.raises(TypeError):
        kv_cache_bytes(4, True, 16, 4)
    with pytest.raises(ValueError):
        kv_cache_bytes(-1, 2, 16, 4)
    with pytest.raises(ValueError):
        kv_cache_bytes(4, 0, 16, 4)
    with pytest.raises(ValueError):
        kv_cache_bytes(4, 2, 0, 4)
    with pytest.raises(ValueError):
        kv_cache_bytes(4, 2, 16, 0)


def test_ideal_transfer_seconds_is_linear_in_payload_and_inverse_in_bandwidth():
    """[ROBUST] Catches: inverted division (bandwidth / payload).

    The single most common way this formula is mis-transcribed.
    """
    assert ideal_transfer_seconds(2e6, 1e9) == pytest.approx(2e-3)
    assert ideal_transfer_seconds(4e6, 1e9) == pytest.approx(4e-3)  # linear in payload
    assert ideal_transfer_seconds(2e6, 2e9) == pytest.approx(
        1e-3
    )  # inverse in bandwidth


def test_ideal_transfer_seconds_of_zero_payload_is_zero():
    """[ROBUST] Catches: fixed overhead smuggled into a payload-only function."""
    assert ideal_transfer_seconds(0, 1e9) == 0.0


@pytest.mark.parametrize("bad_bandwidth", [0.0, -1.0, float("nan"), float("inf")])
def test_ideal_transfer_seconds_rejects_unusable_bandwidth(bad_bandwidth):
    """[ROBUST] Catches: eps-clamp turning 0 B/s into a plausible-looking latency.

    A clamp would produce a finite, wrong number where an error is correct.
    """
    with pytest.raises(ValueError):
        ideal_transfer_seconds(1024.0, bad_bandwidth)


@pytest.mark.parametrize("bad_payload", [-1.0, float("nan"), float("inf"), True])
def test_ideal_transfer_seconds_rejects_unusable_payload(bad_payload):
    """[ROBUST] Catches: negative/NaN/inf/bool payload accepted as valid input."""
    with pytest.raises((TypeError, ValueError)):
        ideal_transfer_seconds(bad_payload, 1e9)


# ===========================================================================
# Public interface
# ===========================================================================


def test_public_operations_keep_their_documented_signatures():
    """[CONTRACT] Catches: renamed/reordered/dropped parameter."""
    assert list(inspect.signature(DisaggregatedClusterManager.__init__).parameters)[
        1:
    ] == [
        "num_prefill_nodes",
        "num_decode_nodes",
        "block_size",
        "d_model",
    ]
    assert list(
        inspect.signature(DisaggregatedClusterManager.route_prefill).parameters
    )[1:] == ["seq_id"]
    assert list(inspect.signature(DisaggregatedClusterManager.route_decode).parameters)[
        1:
    ] == ["seq_id"]
    assert list(
        inspect.signature(DisaggregatedClusterManager.transfer_kv_cache).parameters
    )[1:] == ["seq_id", "prefill_node", "decode_node", "blocks"]
    assert list(
        inspect.signature(DisaggregatedClusterManager.release_request).parameters
    )[1:] == ["seq_id"]
    assert list(inspect.signature(RemoteKVNode.write_prefill_blocks).parameters)[
        1:
    ] == ["seq_id", "k_data", "v_data"]
    assert list(inspect.signature(RemoteKVNode.read_sequence).parameters)[1:] == [
        "seq_id",
        "num_tokens",
    ]
    assert list(inspect.signature(RemoteKVNode.release_sequence).parameters)[1:] == [
        "seq_id"
    ]


# ===========================================================================
# Property-test strategy (comments; Hypothesis not required)
# ===========================================================================
#
# PROPERTY 1 — round-robin exactness + stickiness.
#   @given(n_pre=integers(1, 8), n_dec=integers(1, 8), n_req=integers(1, 50))
#   For any pool sizes and distinct seq_ids, i-th fresh seq -> prefill[i % n_pre],
#   decode[i % n_dec]; re-routing returns same object and does not advance cursor.
#
# PROPERTY 2 — partition/reconstruction round-trip.
#   @given(block_size=integers(1, 8), d_model=integers(1, 8),
#          num_tokens=integers(0, 40),
#          dtype=sampled_from([np.float16, np.float32, np.float64, np.int32]))
#   block count == ceil(n/B); read_sequence preserves dtype, shape, order.
#   num_tokens generated to include 0, exact multiples, ±1 at each boundary.
#
# PROPERTY 3 — permutation invariance of a valid transfer.
#   @given(perm=permutations(range(k))) for a random valid k-block partition.
#   Transfer succeeds and reconstructs the original sequence exactly.
#
# PROPERTY 4 — no partial publication on invalid block set.
#   @given(blocks=lists_of_block_ids(unique=False), dropped=integers(0, k))
#   For any block list that is missing an index, duplicated, or foreign,
#   transfer raises and destination observable state is unchanged.
#
# PROPERTY 5 — latency linearity.
#   @given(payload=floats(0, 1e12), bandwidth=floats(1e3, 1e12),
#          factor=floats(1e-6, 1e6))
#   ideal_transfer_seconds is linear in payload, inverse in bandwidth, finite
#   for finite inputs.
#
# PROPERTY 6 — prefix-read block-access bound (paired with the spy test).


# ===========================================================================
# Not tested (and why):
# - RDMA / InfiniBand / RoCE transport: no network stack exists; a test would
#   assert a mock, not the code.
# - GPU HBM residency, eviction, memory pressure: RemoteKVNode holds host
#   numpy arrays; a "GPU memory test" would be a test of a variable name.
# - Queueing, batching, contention, TTFT: no scheduler or clock modelled.
# - Cross-process / distributed failure modes: a single-process simulator
#   cannot produce them without being rebuilt as a fault-injection framework.
# - Cluster throughput uplift (2x-3x): requires hardware baselines.
# - NaN/inf sanitisation of stored payloads: byte-exact cache by design.
# - Concurrency / thread safety: documented single-threaded.
# - Capacity / remote-memory budgets: out of scope.
# - Request-level read authorization: out of scope (single-process simulator,
#   no caller identity). read_sequence trusts the seq_id. Adding auth would
#   require threading a caller token through every API - disproportionate.
#   Documented in the module docstring.
# - BlockId minting provenance: BlockId is a validation token, not an
#   authentication token. A forged-but-consistent BlockId is accepted; the
#   caller had to guess seq+node+index, which is already outside the sim's
#   trust boundary.
# - Direct mutation via `node._node_id = ...`: prevented by convention, not
#   enforced. Making the `_`-prefixed slot read-only would require a
#   __setattr__ override; not worth it.
# - np.stack / np.vstack variants of the read-prefix mutation: the spy only
#   intercepts np.concatenate. Documented in the spy test docstring.
# - Read isolation between requests on the same node: same as "Request-level
#   read authorization" — out of scope, documented.
