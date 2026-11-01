import hashlib
import io
import json
import threading
from pathlib import Path

import day57_checkpoint as ckpt
import numpy as np
import pytest
from day57_checkpoint import (
    CheckpointConfig,
    CheckpointConflictError,
    CheckpointCorruptionError,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointValidationError,
    DistributedCheckpointManager,
)


def _manager(tmp_path, rank=0, world_size=1, **config_kwargs):
    return DistributedCheckpointManager(
        str(tmp_path),
        rank=rank,
        world_size=world_size,
        config=CheckpointConfig(**config_kwargs),
    )


# ---------------------------------------------------------------------------
# Round-trip and manifest truthfulness
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_values_shape_and_dtype(tmp_path):
    """[ROBUST] Fails if load casts dtype, reshapes arrays, or drops scalar/empty arrays.

    Does NOT test reference aliasing; that is covered by
    test_snapshot_is_isolated_during_serialization and
    test_async_snapshot_taken_before_worker_starts.
    """
    original = {
        "weight": np.arange(8, dtype=np.float32).reshape(4, 2),
        "bias": np.array([1, 2, 3], dtype=np.int64),
        "scalar": np.array(3.5, dtype=np.float64),
        "empty": np.zeros((0, 3), dtype=np.float32),
    }
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=100, state_dict=original)
        restored, metadata = manager.load_checkpoint(step=100)

    assert metadata["step"] == 100
    assert metadata["world_size"] == 1
    for name, arr in original.items():
        assert restored[name].shape == arr.shape
        assert restored[name].dtype == arr.dtype
        np.testing.assert_array_equal(restored[name], arr)


def test_manifest_records_step_world_size_rank_mapping_and_array_metadata(tmp_path):
    """[ROBUST] Fails if manifest omits step/world_size/rank shard mapping or array dtype/shape."""
    weights = {"weight": np.arange(8, dtype=np.float32).reshape(4, 2)}
    with _manager(tmp_path, rank=0, world_size=2) as manager:
        manager.save_checkpoint(step=100, state_dict=weights)
        _, metadata = manager.load_checkpoint(step=100)

    manifest = json.loads((tmp_path / "step_100" / "manifest.json").read_text())
    assert manifest["step"] == 100
    assert manifest["world_size"] == 2
    assert manifest["ranks"]["0"]["shard"] == "rank_0.npz"
    assert manifest["ranks"]["0"]["arrays"]["weight"]["shape"] == [4, 2]
    assert manifest["ranks"]["0"]["arrays"]["weight"]["dtype"] == "float32"
    assert metadata["ranks"]["0"]["shard"] == "rank_0.npz"


def test_two_rank_entries_merge_into_one_step_manifest(tmp_path):
    """[ROBUST] Fails if a rank save overwrites the whole manifest instead of merging its rank entry."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"a": np.ones(2, dtype=np.float32)})
    with _manager(tmp_path, rank=1, world_size=2) as m1:
        m1.save_checkpoint(step=1, state_dict={"b": np.ones(3, dtype=np.int64)})

    manifest = json.loads((tmp_path / "step_1" / "manifest.json").read_text())
    assert set(manifest["ranks"]) == {"0", "1"}

    with _manager(tmp_path, rank=0, world_size=2) as m0:
        state0, _ = m0.load_checkpoint(step=1)
    with _manager(tmp_path, rank=1, world_size=2) as m1:
        state1, _ = m1.load_checkpoint(step=1)

    np.testing.assert_array_equal(state0["a"], np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(state1["b"], np.ones(3, dtype=np.int64))


def test_manifest_offsets_only_when_supplied(tmp_path):
    """[CONTRACT] Fails if manifest fabricates global offsets for local arrays or drops supplied offsets."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(
            step=14,
            state_dict={"w": np.zeros((2, 3), dtype=np.float32)},
            offsets={"w": [0, 0]},
        )
        manager.save_checkpoint(
            step=15, state_dict={"w": np.zeros((2, 3), dtype=np.float32)}
        )

    m14 = json.loads((tmp_path / "step_14" / "manifest.json").read_text())
    m15 = json.loads((tmp_path / "step_15" / "manifest.json").read_text())
    assert m14["ranks"]["0"]["offsets"]["w"] == [0, 0]
    assert "offsets" not in m15["ranks"]["0"]


# ---------------------------------------------------------------------------
# Atomicity, commit ordering, cleanup
# ---------------------------------------------------------------------------


def test_successful_save_leaves_no_tmp_files(tmp_path):
    """[ROBUST] Fails if shard or manifest temp files are left after successful publication."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(
            step=10, state_dict={"data": np.zeros(4, dtype=np.float32)}
        )
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_commit_ordering_via_os_replace_spy(tmp_path, monkeypatch):
    """[ROBUST] Fails if the manifest is published before the shard it references has been moved to its final path (shard os.replace must precede manifest os.replace, and the shard must exist when the manifest is replaced)."""
    events: list[tuple] = []
    original_replace = ckpt.os.replace

    def spy_replace(src, dst, *args, **kwargs):
        dst_p = Path(dst)
        try:
            dst_p.relative_to(tmp_path)
            under_test = True
        except ValueError:
            under_test = False
        if under_test and dst_p.suffix == ".npz":
            events.append(("shard_replace", dst_p.name))
        elif under_test and dst_p.name == "manifest.json":
            sibling = dst_p.parent / "rank_0.npz"
            events.append(("manifest_replace", dst_p.name, sibling.exists()))
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ckpt.os, "replace", spy_replace)
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2, dtype=np.float32)})

    kinds = [e[0] for e in events]
    assert "shard_replace" in kinds, f"no shard publication observed: {events}"
    assert "manifest_replace" in kinds, f"no manifest publication observed: {events}"
    shard_idx = kinds.index("shard_replace")
    manifest_idx = kinds.index("manifest_replace")
    assert shard_idx < manifest_idx, f"manifest published before shard: {events}"
    manifest_event = events[manifest_idx]
    assert manifest_event[2] is True, (
        f"manifest was replaced while its referenced shard was absent: {events}"
    )


def test_forced_serialization_failure_commits_no_manifest_and_cleans_tmp(
    tmp_path, monkeypatch
):
    """[ROBUST] Fails if manifest is committed before shard serialization succeeds or temp cleanup is skipped on failure."""

    def boom(*args, **kwargs):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(ckpt.np, "savez", boom)
    with _manager(tmp_path) as manager, pytest.raises(RuntimeError, match="injected"):
        manager.save_checkpoint(
            step=5, state_dict={"data": np.zeros(4, dtype=np.float32)}
        )

    assert not (tmp_path / "step_5" / "manifest.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_write_failure_after_shard_write_unlinks_shard(tmp_path, monkeypatch):
    """[ROBUST] Fails if a manifest commit failure leaves the freshly-written shard on disk (orphan) or a partial manifest."""

    def boom(self, manifest_path, manifest):
        raise RuntimeError("injected manifest write failure")

    monkeypatch.setattr(DistributedCheckpointManager, "_write_manifest_atomic", boom)
    with (
        _manager(tmp_path) as manager,
        pytest.raises(RuntimeError, match="injected manifest write failure"),
    ):
        manager.save_checkpoint(step=6, state_dict={"x": np.zeros(2, dtype=np.float32)})

    assert not (tmp_path / "step_6" / "manifest.json").exists()
    assert not (tmp_path / "step_6" / "rank_0.npz").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_shard_rename_failure_cleans_tmp_and_raises(tmp_path, monkeypatch):
    """[ROBUST] Fails if os.replace failure on the shard leaves a .tmp behind or lets the manifest proceed."""
    original_replace = ckpt.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith(".npz"):
            raise OSError("injected shard rename failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ckpt.os, "replace", failing_replace)
    try:
        with (
            _manager(tmp_path) as manager,
            pytest.raises(OSError, match="injected shard rename failure"),
        ):
            manager.save_checkpoint(
                step=7, state_dict={"x": np.zeros(2, dtype=np.float32)}
            )
    finally:
        monkeypatch.setattr(ckpt.os, "replace", original_replace)

    assert not (tmp_path / "step_7" / "manifest.json").exists()
    assert not (tmp_path / "step_7" / "rank_0.npz").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_rename_failure_preserves_prior_committed_manifest(
    tmp_path, monkeypatch
):
    """[ROBUST] Fails if a manifest-rename failure while a second rank updates an existing step's manifest removes or corrupts the previously committed rank-0 manifest, or orphans the new rank-1 shard."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"x": np.zeros(2, dtype=np.float32)})

    original_replace = ckpt.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("manifest.json"):
            raise OSError("injected manifest rename failure")
        return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ckpt.os, "replace", failing_replace)
    try:
        with (
            _manager(tmp_path, rank=1, world_size=2) as m1,
            pytest.raises(OSError, match="injected manifest rename"),
        ):
            m1.save_checkpoint(step=1, state_dict={"y": np.ones(3, dtype=np.float32)})
    finally:
        monkeypatch.setattr(ckpt.os, "replace", original_replace)

    manifest = json.loads((tmp_path / "step_1" / "manifest.json").read_text())
    assert set(manifest["ranks"].keys()) == {"0"}

    with _manager(tmp_path, rank=0, world_size=2) as m0:
        restored, _ = m0.load_checkpoint(step=1)
    np.testing.assert_array_equal(restored["x"], np.zeros(2, dtype=np.float32))

    assert not (tmp_path / "step_1" / "rank_1.npz").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_stale_tmp_file_cleaned_on_next_save(tmp_path):
    """[ROBUST] Fails if a stale rank-specific .tmp from a previous aborted save is left in place forever."""
    step_dir = tmp_path / "step_5"
    step_dir.mkdir(parents=True)
    stale = step_dir / ".rank_0.npz.stale-uuid.tmp"
    stale.write_bytes(b"garbage from aborted save")

    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=5, state_dict={"x": np.zeros(2)})

    assert not stale.exists()
    assert (step_dir / "manifest.json").exists()


def test_unrelated_user_tmp_file_is_preserved(tmp_path):
    """[ROBUST] Fails if stale-tmp cleanup is overzealous and deletes arbitrary *.tmp files in the step directory."""
    step_dir = tmp_path / "step_6"
    step_dir.mkdir(parents=True)
    user_tmp = step_dir / "user_data.tmp"
    user_tmp.write_bytes(b"user-owned bytes")

    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=6, state_dict={"x": np.zeros(2)})

    assert user_tmp.exists()
    assert user_tmp.read_bytes() == b"user-owned bytes"


def test_existing_checkpoint_survives_failed_later_save(tmp_path, monkeypatch):
    """[ROBUST] Fails if a failed save deletes or corrupts a previously committed checkpoint."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(
            step=1, state_dict={"stable": np.arange(3, dtype=np.float32)}
        )

        def boom(*args, **kwargs):
            raise RuntimeError("injected later failure")

        monkeypatch.setattr(ckpt.np, "savez", boom)
        with pytest.raises(RuntimeError, match="injected later failure"):
            manager.save_checkpoint(
                step=2, state_dict={"bad": np.zeros(2, dtype=np.float32)}
            )

        restored, _ = manager.load_checkpoint(step=1)

    np.testing.assert_array_equal(restored["stable"], np.arange(3, dtype=np.float32))
    assert not (tmp_path / "step_2" / "manifest.json").exists()


def test_failed_second_rank_save_preserves_first_rank_entry(tmp_path, monkeypatch):
    """[ROBUST] Fails if a failed save for rank 1 at the same step removes or corrupts rank 0's previously committed entry."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=5, state_dict={"x": np.ones(2, dtype=np.float32)})

    def boom(*args, **kwargs):
        raise RuntimeError("injected failure for rank 1")

    monkeypatch.setattr(ckpt.np, "savez", boom)
    with (
        _manager(tmp_path, rank=1, world_size=2) as m1,
        pytest.raises(RuntimeError, match="injected"),
    ):
        m1.save_checkpoint(step=5, state_dict={"y": np.zeros(3)})

    with _manager(tmp_path, rank=0, world_size=2) as m0:
        restored, _ = m0.load_checkpoint(step=5)
    np.testing.assert_array_equal(restored["x"], np.ones(2, dtype=np.float32))
    manifest = json.loads((tmp_path / "step_5" / "manifest.json").read_text())
    assert set(manifest["ranks"].keys()) == {"0"}


# ---------------------------------------------------------------------------
# Integrity / corruption detection
# ---------------------------------------------------------------------------


def test_corrupted_shard_checksum_is_rejected(tmp_path):
    """[ROBUST] Fails if load trusts shard bytes without verifying the manifest checksum."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=7, state_dict={"w": np.ones(4, dtype=np.float32)})
    shard = tmp_path / "step_7" / "rank_0.npz"
    shard.write_bytes(shard.read_bytes() + b"corruption")

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=7)


def test_truncated_shard_is_rejected(tmp_path):
    """[ROBUST] Fails if load silently accepts a truncated shard instead of detecting checksum mismatch."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(
            step=8, state_dict={"x": np.arange(100, dtype=np.float32)}
        )
    shard = tmp_path / "step_8" / "rank_0.npz"
    raw = shard.read_bytes()
    shard.write_bytes(raw[: len(raw) // 2])

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=8)


def test_missing_shard_file_is_rejected(tmp_path):
    """[ROBUST] Fails if load returns partial state when the shard referenced by the manifest is absent."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=9, state_dict={"x": np.zeros(2)})
    (tmp_path / "step_9" / "rank_0.npz").unlink()

    with _manager(tmp_path) as manager, pytest.raises(CheckpointNotFoundError):
        manager.load_checkpoint(step=9)


def test_missing_manifest_fails_loudly(tmp_path):
    """[ROBUST] Fails if load returns empty state when the manifest is absent."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointNotFoundError):
        manager.load_checkpoint(step=999)


def test_malformed_json_manifest_is_rejected(tmp_path):
    """[ROBUST] Fails if load lets a raw JSONDecodeError escape or silently returns empty state."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=3, state_dict={"x": np.ones(2)})
    (tmp_path / "step_3" / "manifest.json").write_text("{not valid json")

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=3)


def test_manifest_with_invalid_utf8_is_rejected(tmp_path):
    """[ROBUST] Fails if invalid UTF-8 in the manifest propagates as UnicodeDecodeError instead of CheckpointCorruptionError."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    (tmp_path / "step_1" / "manifest.json").write_bytes(b"\xff\xfe\xfd")

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=1)


def test_malformed_manifest_schema_is_rejected(tmp_path):
    """[ROBUST] Fails if load accepts a manifest missing the required 'step' field."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=3, state_dict={"x": np.ones(2)})
    manifest_path = tmp_path / "step_3" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    del data["step"]
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.load_checkpoint(step=3)


def test_manifest_with_boolean_format_version_is_rejected(tmp_path):
    """[ROBUST] Fails if a manifest with "format_version": true is accepted because Python's True == 1."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["format_version"] = True
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.load_checkpoint(step=1)


def test_shard_array_name_mismatch_is_rejected(tmp_path):
    """[ROBUST] Fails if load silently returns a subset when manifest lists arrays absent from the shard."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2), "y": np.ones(3)})
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    del data["ranks"]["0"]["arrays"]["y"]
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=1)


def test_shape_mismatch_between_manifest_and_shard_is_rejected(tmp_path):
    """[ROBUST] Fails if load only trusts manifest shape metadata without comparing against the shard."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros((2, 3))})
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["ranks"]["0"]["arrays"]["x"]["shape"] = [3, 2]
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=1)


def test_dtype_mismatch_between_manifest_and_shard_is_rejected(tmp_path):
    """[ROBUST] Fails if load only trusts manifest dtype metadata without comparing against the shard."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2, dtype=np.float32)})
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["ranks"]["0"]["arrays"]["x"]["dtype"] = "float64"
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=1)


def test_manifest_rejects_wrong_rank_shard_reference(tmp_path):
    """[ROBUST] Fails if a manifest whose rank-1 entry points to rank 0's shard (with matching checksum) is accepted."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    with _manager(tmp_path, rank=1, world_size=2) as m1:
        m1.save_checkpoint(step=1, state_dict={"y": np.zeros(3)})

    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["ranks"]["1"]["shard"] = "rank_0.npz"
    data["ranks"]["1"]["sha256"] = data["ranks"]["0"]["sha256"]
    data["ranks"]["1"]["arrays"] = dict(data["ranks"]["0"]["arrays"])
    manifest_path.write_text(json.dumps(data))

    with (
        _manager(tmp_path, rank=1, world_size=2) as m1,
        pytest.raises(CheckpointValidationError),
    ):
        m1.load_checkpoint(step=1)


def test_manifest_rejects_non_canonical_rank_keys(tmp_path):
    """[ROBUST] Fails if load accepts an alias rank key such as "00" for rank 0."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["ranks"]["00"] = data["ranks"]["0"]
    del data["ranks"]["0"]
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.load_checkpoint(step=1)


def test_object_array_shard_is_rejected_at_load(tmp_path):
    """[ROBUST] Fails if np.load is called with pickle enabled (allow_pickle=True) or a manifest with object dtype is silently accepted."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})

    shard = tmp_path / "step_1" / "rank_0.npz"
    buf = io.BytesIO()
    np.savez(buf, x=np.array([{"evil": 1}], dtype=object))
    shard.write_bytes(buf.getvalue())

    new_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_path = tmp_path / "step_1" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["ranks"]["0"]["sha256"] = new_sha
    data["ranks"]["0"]["arrays"]["x"] = {"shape": [1], "dtype": "object"}
    manifest_path.write_text(json.dumps(data))

    with _manager(tmp_path) as manager, pytest.raises(CheckpointCorruptionError):
        manager.load_checkpoint(step=1)


def test_missing_rank_entry_for_requested_rank_is_rejected(tmp_path):
    """[ROBUST] Fails if load falls back to another rank's shard when this rank has no manifest entry."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=4, state_dict={"x": np.ones(2)})

    with (
        _manager(tmp_path, rank=1, world_size=2) as m1,
        pytest.raises(CheckpointNotFoundError),
    ):
        m1.load_checkpoint(step=4)


def test_world_size_mismatch_on_load_is_rejected(tmp_path):
    """[ROBUST] Fails if load ignores manifest world_size and uses a shard from a differently sized job."""
    with _manager(tmp_path, rank=0, world_size=2) as manager:
        manager.save_checkpoint(step=8, state_dict={"x": np.ones(2)})

    with (
        _manager(tmp_path, rank=0, world_size=3) as manager,
        pytest.raises(CheckpointValidationError),
    ):
        manager.load_checkpoint(step=8)


# ---------------------------------------------------------------------------
# Snapshot isolation — deterministic against no-copy and copy-in-worker mutants
# ---------------------------------------------------------------------------


def test_snapshot_is_isolated_during_serialization(tmp_path, monkeypatch):
    """[ROBUST] Fails if save serializes a live caller reference instead of the snapshotted copy: mutating the source array from inside np.savez (after the snapshot boundary but during serialization) must not alter the persisted values."""
    original_savez = ckpt.np.savez
    source_arr = np.arange(4, dtype=np.float32)
    expected = source_arr.copy()

    def mutating_savez(handle, **arrays):
        source_arr[:] = -99
        return original_savez(handle, **arrays)

    monkeypatch.setattr(ckpt.np, "savez", mutating_savez)
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=20, state_dict={"x": source_arr})
        restored, _ = manager.load_checkpoint(step=20)

    np.testing.assert_array_equal(restored["x"], expected)


def test_async_snapshot_taken_before_worker_starts(tmp_path, monkeypatch):
    """[ROBUST] Fails if async save copies arrays inside the worker (instead of before submit): a caller mutation that occurs after submit but before the worker's first read of the arrays must not leak into the checkpoint.

    Deterministic barrier: _save_impl is wrapped to signal and block on a
    threading.Event before delegating to the real _save_impl. Mutating the
    source array between the signal and release guarantees that a
    copy-in-worker mutant observes the mutated value, while a snapshot-before-
    submit implementation is unaffected."""
    original_save_impl = DistributedCheckpointManager._save_impl
    worker_entered = threading.Event()
    allow_worker = threading.Event()

    def controlled_save_impl(self, step, snapshot, offsets):
        worker_entered.set()
        if not allow_worker.wait(timeout=5):
            raise AssertionError("test failed to release worker")
        return original_save_impl(self, step, snapshot, offsets)

    monkeypatch.setattr(
        DistributedCheckpointManager, "_save_impl", controlled_save_impl
    )

    source_arr = np.arange(6, dtype=np.float32)
    expected = source_arr.copy()

    with _manager(tmp_path, async_save=True) as manager:
        handle = manager.save_checkpoint(step=21, state_dict={"x": source_arr})
        assert worker_entered.wait(timeout=5), "worker never entered _save_impl"
        source_arr[:] = -7
        allow_worker.set()
        handle.wait(timeout=5)
        restored, _ = manager.load_checkpoint(step=21)

    np.testing.assert_array_equal(restored["x"], expected)


# ---------------------------------------------------------------------------
# Concurrency / conflict policy
# ---------------------------------------------------------------------------


def test_same_step_rank_second_save_is_rejected(tmp_path):
    """[ROBUST] Fails if a duplicate save for the same step/rank silently overwrites a committed shard."""
    with _manager(tmp_path) as manager:
        manager.save_checkpoint(step=10, state_dict={"x": np.ones(2, dtype=np.float32)})
        with pytest.raises(CheckpointConflictError):
            manager.save_checkpoint(
                step=10, state_dict={"x": np.zeros(2, dtype=np.float32)}
            )
        restored, _ = manager.load_checkpoint(step=10)
    np.testing.assert_array_equal(restored["x"], np.ones(2, dtype=np.float32))


def test_two_managers_same_step_rank_second_is_rejected(tmp_path):
    """[ROBUST] Fails if two DistributedCheckpointManager instances targeting the same dir/step/rank can silently overwrite each other (sequentially, in one process)."""
    with _manager(tmp_path) as m1:
        m1.save_checkpoint(step=11, state_dict={"x": np.ones(2, dtype=np.float32)})
    with _manager(tmp_path) as m2:
        with pytest.raises(CheckpointConflictError):
            m2.save_checkpoint(step=11, state_dict={"x": np.zeros(2, dtype=np.float32)})
        restored, _ = m2.load_checkpoint(step=11)
    np.testing.assert_array_equal(restored["x"], np.ones(2, dtype=np.float32))


def test_serialized_async_saves_on_same_step_rank_yield_single_commit(tmp_path):
    """[ROBUST] Fails if two saves submitted to the same single-worker executor both commit at the same step/rank.

    This proves queue-order semantics: with max_workers=1, submissions are
    serialized, so the first completes and the second raises
    CheckpointConflictError. It does NOT prove cross-thread or cross-process
    race safety; the process-local lock and single-worker executor rule those
    out by design, not by test."""
    with _manager(tmp_path, async_save=True) as manager:
        h1 = manager.save_checkpoint(
            step=12, state_dict={"x": np.ones(2, dtype=np.float32)}
        )
        h2 = manager.save_checkpoint(
            step=12, state_dict={"x": np.zeros(2, dtype=np.float32)}
        )
        h1.wait(timeout=5)
        with pytest.raises(CheckpointConflictError):
            h2.result(timeout=5)
        restored, _ = manager.load_checkpoint(step=12)
    np.testing.assert_array_equal(restored["x"], np.ones(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Async lifecycle
# ---------------------------------------------------------------------------


def test_background_save_completes_and_roundtrips(tmp_path):
    """[ROBUST] Fails if async save never commits or its wait/result handle does not observe completion."""
    with _manager(tmp_path, async_save=True) as manager:
        handle = manager.save_checkpoint(
            step=13, state_dict={"x": np.arange(5, dtype=np.int64)}
        )
        handle.wait(timeout=5)
        restored, _ = manager.load_checkpoint(step=13)
    np.testing.assert_array_equal(restored["x"], np.arange(5, dtype=np.int64))


def test_background_save_propagates_exception(tmp_path, monkeypatch):
    """[ROBUST] Fails if background save swallows serialization exceptions or leaves a manifest."""
    with _manager(tmp_path, async_save=True) as manager:

        def boom(*args, **kwargs):
            raise RuntimeError("async injected")

        monkeypatch.setattr(ckpt.np, "savez", boom)
        handle = manager.save_checkpoint(
            step=14, state_dict={"x": np.zeros(2, dtype=np.float32)}
        )
        with pytest.raises(RuntimeError, match="async injected"):
            handle.result(timeout=5)
        assert not (tmp_path / "step_14" / "manifest.json").exists()
        assert not list(tmp_path.rglob("*.tmp"))


def test_save_after_close_raises(tmp_path):
    """[GUARD] Fails if a closed manager can silently still schedule saves or crash with an unrelated error."""
    manager = _manager(tmp_path)
    manager.close()
    with pytest.raises(CheckpointError, match="closed"):
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})


def test_save_after_close_raises_for_async_manager(tmp_path):
    """[GUARD] Fails if a closed async manager can silently still schedule saves."""
    manager = _manager(tmp_path, async_save=True)
    manager.close()
    with pytest.raises(CheckpointError, match="closed"):
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})


def test_double_close_is_idempotent(tmp_path):
    """[GUARD] Fails if close() is non-idempotent or raises on the second call."""
    manager = _manager(tmp_path, async_save=True)
    manager.close()
    manager.close()


# ---------------------------------------------------------------------------
# Completeness helper
# ---------------------------------------------------------------------------


def test_is_complete_false_for_absent_manifest(tmp_path):
    """[ROBUST] Fails if is_complete() returns True for a step with no manifest or reports errors by raising."""
    with _manager(tmp_path, rank=0, world_size=2) as manager:
        assert manager.is_complete(step=1) is False


def test_is_complete_reflects_global_rank_coverage(tmp_path):
    """[ROBUST] Fails if is_complete() reports True for a partial-rank manifest or False when every rank is present."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
        assert m0.is_complete(step=1) is False

    with _manager(tmp_path, rank=1, world_size=2) as m1:
        m1.save_checkpoint(step=1, state_dict={"y": np.zeros(3)})

    with _manager(tmp_path, rank=0, world_size=2) as m0:
        assert m0.is_complete(step=1) is True


def test_is_complete_false_for_malformed_manifest(tmp_path):
    """[ROBUST] Fails if is_complete() raises on a corrupted manifest instead of returning False."""
    with _manager(tmp_path, rank=0, world_size=1) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
        (tmp_path / "step_1" / "manifest.json").write_text("{not json")

    with _manager(tmp_path, rank=0, world_size=1) as manager:
        assert manager.is_complete(step=1) is False


def test_is_complete_false_when_shard_file_is_deleted(tmp_path):
    """[ROBUST] Fails if is_complete() reports True for a manifest whose referenced shard has been deleted (manifest-coverage-only semantics)."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    with _manager(tmp_path, rank=1, world_size=2) as m1:
        m1.save_checkpoint(step=1, state_dict={"y": np.zeros(3)})

    (tmp_path / "step_1" / "rank_1.npz").unlink()

    with _manager(tmp_path, rank=0, world_size=2) as manager:
        assert manager.is_complete(step=1) is False


def test_is_complete_false_on_checksum_mismatch(tmp_path):
    """[ROBUST] Fails if is_complete() reports True when a referenced shard's bytes no longer match its recorded checksum."""
    with _manager(tmp_path, rank=0, world_size=1) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})

    shard = tmp_path / "step_1" / "rank_0.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with _manager(tmp_path, rank=0, world_size=1) as manager:
        assert manager.is_complete(step=1) is False


def test_is_complete_cheap_mode_skips_checksum_hashing(tmp_path, monkeypatch):
    """[ROBUST] Fails if verify_checksums=False still calls _sha256_file on shards (defeating the cheap-mode contract): instrument the hasher and assert zero calls in cheap mode, at least one in default mode."""
    with _manager(tmp_path, rank=0, world_size=2) as m0:
        m0.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    with _manager(tmp_path, rank=1, world_size=2) as m1:
        m1.save_checkpoint(step=1, state_dict={"y": np.zeros(3)})

    call_count = {"n": 0}
    original_sha = ckpt._sha256_file

    def counting_sha(path):
        call_count["n"] += 1
        return original_sha(path)

    monkeypatch.setattr(ckpt, "_sha256_file", counting_sha)

    with _manager(tmp_path, rank=0, world_size=2) as manager:
        assert manager.is_complete(step=1, verify_checksums=False) is True
        assert call_count["n"] == 0, (
            f"cheap mode hashed {call_count['n']} shards; expected 0"
        )

        assert manager.is_complete(step=1, verify_checksums=True) is True
        assert call_count["n"] >= 1, "default mode did not hash any shard"


def test_is_complete_cheap_mode_requires_shard_existence(tmp_path):
    """[ROBUST] Fails if verify_checksums=False ignores shard existence and returns True for a manifest-only view."""
    with _manager(tmp_path, rank=0, world_size=1) as manager:
        manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})

    (tmp_path / "step_1" / "rank_0.npz").unlink()

    with _manager(tmp_path, rank=0, world_size=1) as manager:
        assert manager.is_complete(step=1, verify_checksums=False) is False


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_object_dtype_arrays_are_rejected(tmp_path):
    """[GUARD] Fails if object dtype arrays are allowed into a pickle-free serialization path."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.save_checkpoint(
            step=1, state_dict={"bad": np.array([{"a": 1}], dtype=object)}
        )


def test_non_ndarray_state_values_are_rejected(tmp_path):
    """[GUARD] Fails if non-ndarray values are silently serialized."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.save_checkpoint(step=1, state_dict={"bad": [1, 2, 3]})


def test_nested_state_values_are_rejected(tmp_path):
    """[GUARD] Fails if nested mappings are silently serialized."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.save_checkpoint(
            step=1,
            state_dict={"nested": {"x": np.zeros(1, dtype=np.float32)}},
        )


@pytest.mark.parametrize("rank,world_size", [(-1, 2), (2, 2), (0, 0)])
def test_constructor_rejects_invalid_rank_and_world_size(tmp_path, rank, world_size):
    """[GUARD] Fails if rank/world_size validation is missing or off-by-one."""
    with pytest.raises(CheckpointValidationError):
        DistributedCheckpointManager(str(tmp_path), rank=rank, world_size=world_size)


def test_constructor_rejects_manifest_name_colliding_with_shard(tmp_path):
    """[GUARD] Fails if a manifest filename that collides with a generated shard filename is accepted, producing an unrestorable checkpoint."""
    with pytest.raises(CheckpointValidationError, match="collides"):
        DistributedCheckpointManager(
            str(tmp_path),
            rank=0,
            world_size=1,
            config=CheckpointConfig(manifest_name="rank_0.npz"),
        )


@pytest.mark.parametrize("bad_prefix", ["*", "rank?", "r[a]nk", "r*"])
def test_constructor_rejects_shard_prefix_with_glob_metacharacters(
    tmp_path, bad_prefix
):
    """[GUARD] Fails if a shard_prefix containing glob metacharacters is accepted, allowing stale-temp cleanup (which interpolates the prefix into a glob pattern) to accidentally match unrelated files."""
    with pytest.raises(CheckpointValidationError, match="glob metacharacters"):
        DistributedCheckpointManager(
            str(tmp_path),
            rank=0,
            world_size=1,
            config=CheckpointConfig(shard_prefix=bad_prefix),
        )


def test_constructor_allows_custom_shard_prefix_with_non_colliding_manifest(
    tmp_path,
):
    """[CONTRACT] Fails if the collision check is overzealous and rejects valid custom configurations."""
    manager = DistributedCheckpointManager(
        str(tmp_path),
        rank=0,
        world_size=1,
        config=CheckpointConfig(shard_prefix="ckpt", manifest_name="manifest.json"),
    )
    manager.save_checkpoint(step=1, state_dict={"x": np.zeros(2)})
    restored, _ = manager.load_checkpoint(step=1)
    np.testing.assert_array_equal(restored["x"], np.zeros(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"async_save": 1},
        {"async_save": "yes"},
        {"fsync": 0},
        {"fsync": "no"},
        {"format_version": True},
        {"format_version": 0},
        {"format_version": -1},
        {"format_version": 1.5},
    ],
)
def test_constructor_rejects_invalid_config_fields(tmp_path, kwargs):
    """[GUARD] Fails if CheckpointConfig accepts a non-bool async/fsync flag or a non-positive/non-int format_version."""
    with pytest.raises(CheckpointValidationError):
        DistributedCheckpointManager(
            str(tmp_path),
            rank=0,
            world_size=1,
            config=CheckpointConfig(**kwargs),
        )


def test_save_rejects_invalid_step(tmp_path):
    """[GUARD] Fails if negative or boolean step values are accepted on save."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.save_checkpoint(step=-1, state_dict={"x": np.zeros(1)})


def test_save_rejects_boolean_step(tmp_path):
    """[GUARD] Fails if boolean step values are accepted on save (True == 1)."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.save_checkpoint(step=True, state_dict={"x": np.zeros(1)})


def test_load_rejects_invalid_step(tmp_path):
    """[GUARD] Fails if negative step values are accepted on load."""
    with _manager(tmp_path) as manager, pytest.raises(CheckpointValidationError):
        manager.load_checkpoint(step=-1)


# ---------------------------------------------------------------------------
# Starter-API contract
# ---------------------------------------------------------------------------


def test_public_starter_usage_roundtrip(tmp_path):
    """[CONTRACT] Fails if the documented starter import/call sequence changes."""
    world_size = 2
    rank = 0
    local_weights = {"weight": np.arange(8, dtype=np.float32).reshape(4, 2)}

    manager = DistributedCheckpointManager(
        checkpoint_dir=str(tmp_path), rank=rank, world_size=world_size
    )
    manager.save_checkpoint(step=100, state_dict=local_weights)
    restored_state, metadata = manager.load_checkpoint(step=100)

    assert metadata["step"] == 100
    assert metadata["world_size"] == world_size
    np.testing.assert_array_equal(restored_state["weight"], local_weights["weight"])


# PROPERTY: for any flat mapping of supported non-object NumPy arrays, saving at
# step s for rank r and then loading at step s for rank r returns arrays with
# identical shape, dtype, and values. Hypothesis strategy: generate dictionaries
# of safe-name keys to arrays with numeric/bool dtypes, including 0-d and
# zero-sized shapes. Not executed here because Hypothesis is not a dependency.
#
# Not tested (and why):
# - Real multi-process/multi-node barriers and collective manifest commit:
#   explicitly a local single-process simulator.
# - Concurrent same-step saves from two threads in one process with *different*
#   ranks: covered only sequentially by
#   test_two_rank_entries_merge_into_one_step_manifest; the class-level step
#   lock is believed to serialize but is not exercised across ranks.
# - Cross-thread / cross-process race against the process-local step lock:
#   test_serialized_async_saves_on_same_step_rank_yield_single_commit
#   exercises the single-worker executor's queue order, not a critical-section
#   race. A real race test would require more than one worker or more than one
#   process.
# - Crash durability after os.replace under power loss: no directory fsync
#   and no filesystem fault injection is implemented.
# - Inter-process concurrency on shared filesystems: only process-local locks
#   serialize same-step saves; two OS processes can still race on manifests.
# - Network-filesystem rename/atomicity semantics: os.replace behavior is
#   filesystem-dependent and not emulated.
# - Symlink-alias step-lock collision: _get_step_lock keys on abspath, not
#   realpath; documented but not tested.
# - Repair of corrupted shards: checksums detect corruption but do not recover.
# - Optimizer/scheduler non-array state: supported contract is flat
#   str -> numpy.ndarray only.
# - Concurrent mutation of source arrays *during* the snapshot copy:
#   documented as a caller responsibility; not deterministically testable
#   without instruments inside np.array(copy=True).
# - Full distributed restart from saved shards: needs real multi-process
#   orchestration which this simulator does not implement.
# - is_complete() performance characteristics on very large clusters: no
#   benchmark; the default verify_checksums=True reads each shard fully.
