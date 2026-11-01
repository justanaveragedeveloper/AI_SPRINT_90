"""Distributed checkpointing and atomic state-publication reference simulator.

Problem solved
--------------
Small, auditable, single-process simulator of rank-local checkpoint sharding,
atomic shard/manifest publication, checksummed integrity verification, snapshot
isolation, and optional background saving.

Complexity
----------
For rank r with local state size P_r bytes and A_r arrays:
- snapshot copy: O(P_r) data movement plus per-array overhead
- shard serialization/write: O(P_r)
- shard sha256: O(P_r)
- manifest update: O(A_r + R) where R is existing rank entries
- load: O(P_r) checksum + O(P_r) decode

Background saving can overlap I/O with caller work, but it does not remove I/O
cost and does not reduce peak memory. Peak memory during a save is at least
O(P_r) for the private snapshot plus O(P_r) for the serialization buffer. No
zero-copy claim is made and no benchmark is included.

What this is NOT
----------------
Not a real multi-node distributed-training checkpoint framework. No inter-
process coordination, no shared-filesystem rename guarantees, no directory
fsync, no crash durability, no distributed barriers, no cluster fault
tolerance. The class-level step lock is process-local; two OS processes can
still race on manifests. The lock registry is process-lifetime and grows by
one RLock per distinct (checkpoint_dir, step); a long-lived daemon would need
bounded eviction.

Reproducibility
---------------
Checkpoint content and manifest bytes are deterministic: array metadata is
sorted by name, JSON is written with sort_keys=True, no RNG is used in
checkpoint content. Temp-file names use uuid4 only for collision avoidance and
are never part of persisted state. No seed parameter is required.

Snapshot contract
-----------------
save_checkpoint copies caller arrays before scheduling work. Mutating the
originals after save_checkpoint returns cannot change the submitted checkpoint.
The copy is not transactional across arrays: concurrent mutation of the source
arrays while save_checkpoint is running can produce a mixed-moment snapshot.
Callers that need cross-array consistency must coordinate their own
quiescence / lock / version protocol.

Public API compatibility
------------------------
The starter constructor and calls are unchanged:
    DistributedCheckpointManager(checkpoint_dir=..., rank=..., world_size=...)
    manager.save_checkpoint(step=..., state_dict=...)
    restored, metadata = manager.load_checkpoint(step=...)

CheckpointConfig is optional; it exists so sync/async and fsync behaviour can
be selected without touching the constructor signature. is_complete(step,
verify_checksums=True) is an additional read-only helper that returns True
only when every configured rank has a committed manifest entry *and* its shard
exists on disk (checksums verified by default). It is the fail-closed "is this
step wholly restorable?" query.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, cast

import numpy as np

__all__ = [
    "CheckpointConfig",
    "CheckpointConflictError",
    "CheckpointCorruptionError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointValidationError",
    "DistributedCheckpointManager",
    "SaveHandle",
]


class CheckpointError(Exception):
    """Base error for checkpoint operations."""


class CheckpointValidationError(CheckpointError):
    """Raised when public inputs or checkpoint metadata are invalid."""


class CheckpointConflictError(CheckpointError):
    """Raised when a save would overwrite a committed step/rank entry."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested manifest or shard is absent."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when checksum or manifest/shard consistency verification fails."""


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for DistributedCheckpointManager.

    async_save: if True, save_checkpoint returns a SaveHandle backed by a
        single-worker ThreadPoolExecutor. The caller must observe the handle;
        dropping it silently hides background exceptions.
    fsync: if True, fsync temporary shard/manifest files before os.replace.
        This still does not perform directory fsync and is not a crash
        durability guarantee.
    manifest_name: basename of the per-step manifest file. Must not collide
        with the shard naming pattern {shard_prefix}_<rank>.npz.
    shard_prefix: prefix for rank shard filenames. Must not contain glob
        metacharacters because it is interpolated into the stale-temp cleanup
        glob pattern.
    format_version: manifest schema version.
    """

    async_save: bool = False
    fsync: bool = False
    manifest_name: str = "manifest.json"
    shard_prefix: str = "rank"
    format_version: int = 1


class SaveHandle:
    """Wrapper around concurrent.futures.Future for save completion.

    Callers must call wait() or result() to observe completion and propagate
    any background exception. Silently dropping the handle discards the error.
    """

    def __init__(self, future: Future) -> None:
        self._future = future

    def wait(self, timeout: float | None = None) -> None:
        self._future.result(timeout=timeout)

    def result(self, timeout: float | None = None) -> dict[str, object]:
        return cast(dict[str, object], self._future.result(timeout=timeout))


_ARRAY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_GLOB_METACHARS = frozenset("*?[]")


def _validate_world_size(world_size: int) -> int:
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size < 1
    ):
        raise CheckpointValidationError("world_size must be an integer >= 1")
    return world_size


def _validate_rank(rank: int, world_size: int) -> int:
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not (0 <= rank < world_size)
    ):
        raise CheckpointValidationError(
            f"rank must be an integer in [0, {world_size - 1}], got {rank!r}"
        )
    return rank


def _validate_step(step: int) -> int:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointValidationError("step must be an integer >= 0")
    return step


def _validate_array_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(".")
        or _ARRAY_NAME_RE.match(name) is None
    ):
        raise CheckpointValidationError(
            "array names must be non-empty strings matching [A-Za-z0-9_.-]+ "
            "and must not start with '.'"
        )
    return name


def _snapshot_state_dict(
    state_dict: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not isinstance(state_dict, Mapping):
        raise CheckpointValidationError(
            "state_dict must be a mapping of str to numpy.ndarray"
        )
    snapshot: dict[str, np.ndarray] = {}
    for name, value in state_dict.items():
        _validate_array_name(name)
        if not isinstance(value, np.ndarray):
            raise CheckpointValidationError(
                f"state_dict[{name!r}] must be a numpy.ndarray, got {type(value).__name__}"
            )
        if value.dtype.hasobject:
            raise CheckpointValidationError(
                f"state_dict[{name!r}] has object dtype; unsafe for pickle-free "
                "serialization"
            )
        snapshot[name] = np.array(value, copy=True, order="C")
    return snapshot


def _snapshot_offsets(
    offsets: Mapping[str, Sequence[int]] | None,
    snapshot: Mapping[str, np.ndarray],
) -> dict[str, list[int]] | None:
    if offsets is None:
        return None
    if not isinstance(offsets, Mapping):
        raise CheckpointValidationError(
            "offsets must be a mapping of array name to integer sequence"
        )
    result: dict[str, list[int]] = {}
    for name, value in offsets.items():
        _validate_array_name(name)
        if name not in snapshot:
            raise CheckpointValidationError(
                f"offsets provided for unknown array {name!r}"
            )
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise CheckpointValidationError(
                f"offsets[{name!r}] must be a sequence of integers"
            )
        dims: list[int] = []
        for dim in value:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise CheckpointValidationError(
                    f"offsets[{name!r}] must contain non-negative integers"
                )
            dims.append(dim)
        if len(dims) != snapshot[name].ndim:
            raise CheckpointValidationError(
                f"offsets[{name!r}] length {len(dims)} does not match ndim "
                f"{snapshot[name].ndim}"
            )
        result[name] = dims
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise CheckpointValidationError(f"config.{name} must be a bool")
    return value


def _validate_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CheckpointValidationError(
            f"config.{name} must be an int >= 1 (bool excluded), got {value!r}"
        )
    return value


def _validate_safe_basename(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise CheckpointValidationError(f"{name} must be a safe basename")
    return value


def _validate_shard_prefix(value: object) -> str:
    _validate_safe_basename("shard_prefix", value)
    assert isinstance(value, str)
    bad = sorted(set(value) & _GLOB_METACHARS)
    if bad:
        raise CheckpointValidationError(
            f"shard_prefix {value!r} must not contain glob metacharacters "
            f"{bad!r}; it is interpolated into the stale-temp cleanup pattern"
        )
    return value


class DistributedCheckpointManager:
    """Rank-local checkpoint simulator with atomic shard and manifest publication.

    Chose a per-step, merged manifest over one manifest per rank so loaders can
    validate step/world-size/rank mappings from a single file. Matters when
    several simulated ranks in one process target the same step.

    Chose os.replace over direct overwrite because it publishes a complete file
    atomically on supported local filesystems. Matters when a process dies
    during a write: the manifest never points at a partially written shard.
    This does not guarantee crash durability; directory fsync is not performed
    and os.replace atomicity is filesystem-dependent.

    Chose np.savez(handle, ...) + np.load(..., allow_pickle=False) over pickle
    because checkpoint bytes must never be able to execute code on load.
    """

    _locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _step_locks: ClassVar[dict[tuple[str, int], threading.RLock]] = {}

    def __init__(
        self,
        checkpoint_dir: str,
        rank: int,
        world_size: int,
        config: CheckpointConfig | None = None,
    ) -> None:
        if not isinstance(checkpoint_dir, (str, os.PathLike)) or not os.fspath(
            checkpoint_dir
        ):
            raise CheckpointValidationError(
                "checkpoint_dir must be a non-empty path-like"
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.world_size = _validate_world_size(world_size)
        self.rank = _validate_rank(rank, self.world_size)

        if config is None:
            config = CheckpointConfig()
        if not isinstance(config, CheckpointConfig):
            raise CheckpointValidationError(
                "config must be a CheckpointConfig instance"
            )
        _validate_bool("async_save", config.async_save)
        _validate_bool("fsync", config.fsync)
        _validate_positive_int("format_version", config.format_version)
        _validate_safe_basename("manifest_name", config.manifest_name)
        _validate_shard_prefix(config.shard_prefix)

        shard_pattern = re.compile(rf"^{re.escape(config.shard_prefix)}_[0-9]+\.npz$")
        if shard_pattern.match(config.manifest_name):
            raise CheckpointValidationError(
                f"manifest_name {config.manifest_name!r} collides with shard "
                f"filename pattern {config.shard_prefix}_<rank>.npz"
            )

        self.config = config
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        if self.config.async_save:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"ckpt-rank-{self.rank}"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the background executor if present.

        Idempotent. Blocks until any in-flight background save finishes. To
        observe errors from an in-flight save, the caller must have called
        SaveHandle.wait() or SaveHandle.result() before close() returns —
        close() itself does not surface background exceptions. Calling
        close() from within a worker thread would deadlock; that pattern is
        out of scope.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._closed = True

    @classmethod
    def _get_step_lock(cls, checkpoint_dir: Path, step: int) -> threading.RLock:
        """Return the process-local lock keyed by (abspath(checkpoint_dir), step).

        Keyed by abspath, not realpath: two symlink aliases that resolve to
        the same directory receive distinct locks and are NOT serialized
        against each other. Callers sharing a checkpoint directory must use
        a consistent path spelling if they rely on the lock.
        """
        key = (os.path.abspath(os.fspath(checkpoint_dir)), step)
        with cls._locks_guard:
            lock = cls._step_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._step_locks[key] = lock
            return lock

    def _expected_shard_name(self, rank: int) -> str:
        return f"{self.config.shard_prefix}_{rank}.npz"

    def save_checkpoint(
        self,
        step: int,
        state_dict: Mapping[str, np.ndarray],
        *,
        offsets: Mapping[str, Sequence[int]] | None = None,
    ) -> SaveHandle:
        """Save this rank's local state for a step.

        The caller-visible arrays are snapshotted before submission. Mutating
        the originals after this returns cannot alter the checkpoint that was
        submitted for saving, synchronous or asynchronous. The snapshot is not
        transactional across arrays: callers must not mutate the source arrays
        while save_checkpoint is running.
        """
        step = _validate_step(step)
        if self._closed or (self.config.async_save and self._executor is None):
            raise CheckpointError("checkpoint manager has been closed")
        snapshot = _snapshot_state_dict(state_dict)
        offsets_snapshot = _snapshot_offsets(offsets, snapshot)

        if self.config.async_save:
            executor = self._executor
            assert executor is not None
            future = executor.submit(self._save_impl, step, snapshot, offsets_snapshot)
            return SaveHandle(future)

        future: Future = Future()
        future.set_result(self._save_impl(step, snapshot, offsets_snapshot))
        return SaveHandle(future)

    def _save_impl(
        self,
        step: int,
        snapshot: Mapping[str, np.ndarray],
        offsets: Mapping[str, Sequence[int]] | None,
    ) -> dict[str, object]:
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = step_dir / self.config.manifest_name
        shard_filename = self._expected_shard_name(self.rank)
        shard_path = step_dir / shard_filename

        lock = self._get_step_lock(self.checkpoint_dir, step)
        with lock:
            for stale in step_dir.glob(f".{shard_filename}.*.tmp"):
                try:
                    stale.unlink()
                except OSError:
                    pass

            if manifest_path.exists():
                manifest = self._read_manifest(manifest_path, step)
                ranks = cast(dict[str, object], manifest["ranks"])
            else:
                manifest = {
                    "format_version": self.config.format_version,
                    "step": step,
                    "world_size": self.world_size,
                    "ranks": {},
                }
                ranks = cast(dict[str, object], manifest["ranks"])

            rank_key = str(self.rank)
            if rank_key in ranks:
                raise CheckpointConflictError(
                    f"rank {self.rank} already has a committed checkpoint at "
                    f"step {step}"
                )

            sha256 = self._write_shard_atomic(shard_path, snapshot)

            arrays_meta: dict[str, object] = {}
            for name in sorted(snapshot):
                arr = snapshot[name]
                arrays_meta[name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }

            entry: dict[str, object] = {
                "shard": shard_filename,
                "sha256": sha256,
                "arrays": arrays_meta,
            }
            if offsets is not None:
                entry["offsets"] = {
                    name: list(value) for name, value in sorted(offsets.items())
                }

            ranks[rank_key] = entry
            manifest["ranks"] = ranks

            try:
                self._write_manifest_atomic(manifest_path, manifest)
            except BaseException:
                try:
                    shard_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

            return manifest

    def _write_shard_atomic(
        self,
        shard_path: Path,
        snapshot: Mapping[str, np.ndarray],
    ) -> str:
        tmp_path = shard_path.parent / f".{shard_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "wb") as handle:
                np.savez(handle, **snapshot)
                handle.flush()
                if self.config.fsync:
                    os.fsync(handle.fileno())
            sha256 = _sha256_file(tmp_path)
            os.replace(tmp_path, shard_path)
            return sha256
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_manifest_atomic(
        self,
        manifest_path: Path,
        manifest: Mapping[str, object],
    ) -> None:
        tmp_path = (
            manifest_path.parent / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                if self.config.fsync:
                    os.fsync(handle.fileno())
            os.replace(tmp_path, manifest_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_manifest(
        self, manifest_path: Path, expected_step: int
    ) -> dict[str, object]:
        if not manifest_path.exists():
            raise CheckpointNotFoundError(f"manifest not found: {manifest_path}")
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except UnicodeDecodeError as exc:
            raise CheckpointCorruptionError(
                f"manifest is not valid UTF-8: {manifest_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptionError(
                f"manifest is not valid JSON: {manifest_path}"
            ) from exc
        if not isinstance(data, dict):
            raise CheckpointValidationError("manifest root must be a JSON object")
        self._validate_manifest(data, expected_step)
        return data

    def _validate_manifest(
        self, manifest: Mapping[str, object], expected_step: int
    ) -> None:
        fv = manifest.get("format_version")
        if not isinstance(fv, int) or isinstance(fv, bool):
            raise CheckpointValidationError(
                f"manifest format_version must be an integer, got {type(fv).__name__}"
            )
        if fv != self.config.format_version:
            raise CheckpointValidationError(
                f"manifest format_version must be {self.config.format_version}"
            )

        step = manifest.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step != expected_step:
            raise CheckpointValidationError(
                f"manifest step must be integer {expected_step}, got {step!r}"
            )
        world_size = manifest.get("world_size")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size != self.world_size
        ):
            raise CheckpointValidationError(
                f"manifest world_size must be {self.world_size}, got {world_size!r}"
            )
        ranks = manifest.get("ranks")
        if not isinstance(ranks, dict):
            raise CheckpointValidationError("manifest ranks must be an object")

        for rank_key, entry in ranks.items():
            if (
                not isinstance(rank_key, str)
                or not rank_key.isdigit()
                or str(int(rank_key)) != rank_key
            ):
                raise CheckpointValidationError(
                    f"non-canonical rank key in manifest: {rank_key!r}"
                )
            rank_int = int(rank_key)
            if not (0 <= rank_int < self.world_size):
                raise CheckpointValidationError(
                    f"manifest rank key {rank_int} outside world_size {self.world_size}"
                )
            if not isinstance(entry, dict):
                raise CheckpointValidationError(
                    f"manifest entry for rank {rank_int} must be an object"
                )
            shard = entry.get("shard")
            expected_shard = self._expected_shard_name(rank_int)
            if shard != expected_shard:
                raise CheckpointValidationError(
                    f"manifest shard {shard!r} for rank {rank_int} does not "
                    f"match expected {expected_shard!r}"
                )
            sha256 = entry.get("sha256")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(ch not in string.hexdigits for ch in sha256)
            ):
                raise CheckpointValidationError(
                    f"manifest entry for rank {rank_int} has invalid sha256"
                )
            arrays = entry.get("arrays")
            if not isinstance(arrays, dict):
                raise CheckpointValidationError(
                    f"manifest entry for rank {rank_int} arrays must be an object"
                )
            for name, meta in arrays.items():
                if not isinstance(name, str) or not name:
                    raise CheckpointValidationError(
                        "manifest array names must be non-empty strings"
                    )
                if not isinstance(meta, dict):
                    raise CheckpointValidationError(
                        f"array metadata for {name!r} must be an object"
                    )
                shape = meta.get("shape")
                if not isinstance(shape, list) or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                    for dim in shape
                ):
                    raise CheckpointValidationError(
                        f"array metadata for {name!r} has invalid shape"
                    )
                dtype = meta.get("dtype")
                if not isinstance(dtype, str) or not dtype:
                    raise CheckpointValidationError(
                        f"array metadata for {name!r} has invalid dtype"
                    )
            offsets = entry.get("offsets")
            if offsets is not None:
                if not isinstance(offsets, dict):
                    raise CheckpointValidationError(
                        f"manifest entry for rank {rank_int} offsets must be an object"
                    )
                for name, value in offsets.items():
                    if not isinstance(name, str) or name not in arrays:
                        raise CheckpointValidationError(
                            f"offsets key {name!r} has no matching array"
                        )
                    if not isinstance(value, list) or any(
                        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                        for dim in value
                    ):
                        raise CheckpointValidationError(
                            f"offsets for {name!r} must be non-negative integers"
                        )
                    meta = arrays[name]
                    if not isinstance(meta, dict):
                        raise CheckpointValidationError(
                            f"array metadata for {name!r} must be an object"
                        )
                    shape = meta.get("shape")
                    if not isinstance(shape, list) or len(value) != len(shape):
                        raise CheckpointValidationError(
                            f"offsets for {name!r} length does not match shape rank"
                        )

    def is_complete(self, step: int, *, verify_checksums: bool = True) -> bool:
        """Return True only if this step is fully restorable across all ranks.

        Specifically, returns True iff:
          1. The manifest exists and validates against this manager's schema.
          2. Every configured rank (0 .. world_size-1) has an entry in the
             manifest's `ranks` mapping.
          3. Each referenced shard file exists on disk.
          4. When verify_checksums is True (default), each referenced shard's
             sha256 matches the manifest entry.

        Returns False -- never raises -- for: absent manifest, malformed or
        schema-invalid manifest, missing rank entry, missing shard file, or
        (when verify_checksums=True) checksum mismatch. Rank-local loading via
        load_checkpoint() still raises precise exceptions; this helper is the
        fail-closed "is this step wholly present?" query.

        Chose a strong default (verify_checksums=True) over manifest-only
        coverage because a caller asking "is this step fully restorable?" should
        not receive True when shards are gone or corrupt. The cheap variant
        remains available via verify_checksums=False for existence-only checks
        when the O(P) hash cost is unacceptable; in that mode _sha256_file is
        not called on any shard.
        """
        step = _validate_step(step)
        step_dir = self.checkpoint_dir / f"step_{step}"
        manifest_path = step_dir / self.config.manifest_name
        if not manifest_path.exists():
            return False
        try:
            manifest = self._read_manifest(manifest_path, step)
        except CheckpointError:
            return False

        ranks = manifest.get("ranks")
        if not isinstance(ranks, dict):
            return False
        expected_keys = {str(r) for r in range(self.world_size)}
        if set(ranks.keys()) != expected_keys:
            return False

        for rank_key in sorted(ranks.keys()):
            entry = ranks[rank_key]
            if not isinstance(entry, dict):
                return False
            shard_name = entry.get("shard")
            if not isinstance(shard_name, str):
                return False
            shard_path = step_dir / shard_name
            if not shard_path.exists():
                return False
            if verify_checksums:
                expected_sha = entry.get("sha256")
                try:
                    actual_sha = _sha256_file(shard_path)
                except OSError:
                    return False
                if actual_sha != expected_sha:
                    return False
        return True

    def load_checkpoint(
        self, step: int
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        """Load and verify this rank's shard for the requested step."""
        step = _validate_step(step)
        step_dir = self.checkpoint_dir / f"step_{step}"
        manifest_path = step_dir / self.config.manifest_name
        manifest = self._read_manifest(manifest_path, step)
        ranks = cast(dict[str, object], manifest["ranks"])
        rank_key = str(self.rank)

        if rank_key not in ranks:
            raise CheckpointNotFoundError(
                f"manifest at step {step} has no entry for rank {self.rank}"
            )
        entry = ranks[rank_key]
        if not isinstance(entry, dict):
            raise CheckpointValidationError(
                f"manifest entry for rank {self.rank} must be an object"
            )
        shard_name = entry.get("shard")
        if not isinstance(shard_name, str):
            raise CheckpointValidationError(
                f"manifest entry for rank {self.rank} has invalid shard"
            )
        shard_path = step_dir / shard_name
        if not shard_path.exists():
            raise CheckpointNotFoundError(f"shard not found: {shard_path}")

        expected_sha = entry.get("sha256")
        actual_sha = _sha256_file(shard_path)
        if actual_sha != expected_sha:
            raise CheckpointCorruptionError(
                f"checksum mismatch for {shard_path}: expected {expected_sha}, "
                f"got {actual_sha}"
            )

        try:
            with np.load(shard_path, allow_pickle=False) as npz:
                arrays = {name: np.array(npz[name], copy=True) for name in npz.files}
        except Exception as exc:
            raise CheckpointCorruptionError(
                f"failed to load shard {shard_path}: {exc}"
            ) from exc

        arrays_meta = entry.get("arrays")
        if not isinstance(arrays_meta, dict):
            raise CheckpointValidationError(
                f"manifest entry for rank {self.rank} arrays must be an object"
            )
        if set(arrays.keys()) != set(arrays_meta.keys()):
            raise CheckpointCorruptionError(
                f"shard array names {sorted(arrays)} do not match manifest "
                f"{sorted(arrays_meta)}"
            )
        for name, meta in arrays_meta.items():
            if not isinstance(meta, dict):
                raise CheckpointValidationError(
                    f"array metadata for {name!r} must be an object"
                )
            expected_shape = meta.get("shape")
            expected_dtype = meta.get("dtype")
            arr = arrays[name]
            if list(arr.shape) != expected_shape:
                raise CheckpointCorruptionError(
                    f"shape mismatch for {name!r}: expected {expected_shape}, "
                    f"got {list(arr.shape)}"
                )
            if str(arr.dtype) != expected_dtype:
                raise CheckpointCorruptionError(
                    f"dtype mismatch for {name!r}: expected {expected_dtype}, "
                    f"got {arr.dtype}"
                )
        return arrays, manifest
