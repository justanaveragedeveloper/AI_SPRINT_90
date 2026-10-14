"""
Pipeline-parallel training scheduler (1F1B).

What this file does
-------------------
Pipeline parallelism splits a neural network across several stages (in
practice, several GPUs, each holding a slice of the model). One big
training mini-batch is broken into M smaller pieces called micro-batches
so the stages can work in parallel.

This file decides the order in which each stage should run its two kinds
of work:
    FORWARD  -- push a micro-batch through the stage.
    BACKWARD -- send gradients back through the stage.

The scheduling pattern is called 1F1B: one forward, one backward. After
a short warm-up, each stage alternates between the two so activation
memory stays bounded.

What this file does NOT do
--------------------------
It only plans the work. It does not run any neural network, move data
between GPUs, compute gradients, or measure memory.

Conventions
-----------
- Stage 0 is the first (input) stage.
- Stage S-1 is the last (output / loss) stage.
- A task is identified by (stage_id, micro_batch_id, task_type).

Warm-up
-------
Stage s starts by running a few forwards before it can begin any
backward work. The count of these preliminary forwards is:

    warmup(s) = min(M, S - 1 - s)

Stage 0 warms up the most; the last stage warms up zero times. Warm-up
is capped at M because a stage cannot forward more micro-batches than
exist.

A note on what "zero" means here. When warmup(s) is 0 the stage still
emits one forward before its first backward, because the alternation
loop starts with a forward. So the leading run of forwards in the queue
is:

    warmup(s)      if warmup(s) > 0
    1              otherwise (i.e. for the last stage, since M >= 1)

Steady state
------------
After warm-up, each stage alternates:
    backward on the oldest outstanding micro-batch,
    then forward on the next new one.
Backwards always go in increasing micro-batch order (oldest first).

Cool-down
---------
Once a stage has forwarded every micro-batch, it just finishes the
remaining backwards in increasing micro-batch order.

Complexity
----------
- Generation: O(S * M). Each stage emits exactly 2*M tasks with O(1)
  work per task; with S stages that is 2*S*M tasks total.
- Storage: O(S * M). Every emitted task is kept.
- Validation: O(S * M) to build a dependency graph and check it.
- Idealized pipeline steps under balanced stages: O(S + M).
- Idealized efficiency: M / (M + S - 1). This assumes every stage takes
  the same time and there is no communication cost. Real runs are
  usually less efficient than this.

Determinism and numerical notes
-------------------------------
This file has no randomness. Two calls with the same settings return
schedules with the same values (but fresh objects, so mutating one does
not affect the other). There are no floats here, so floating-point
stability rules do not apply.

Honest limitations
------------------
- Not a GPU runtime.
- Not an autograd engine.
- Not a memory profiler. The O(S) activation-memory figure from the
  literature is a scheduling intuition; this file does not measure it.
- Proving a topological order exists is not the same as proving that
  real hardware can run the queues without resource conflicts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

TaskKey = tuple[int, int, str]


@dataclass(frozen=True)
class MicroBatchTask:
    """One FORWARD or BACKWARD task for a single micro-batch on one stage.

    Frozen so that two calls to generate_schedule() cannot accidentally
    share or mutate the same task objects.
    """

    stage_id: int
    micro_batch_id: int
    task_type: str  # "FORWARD" or "BACKWARD"

    def __post_init__(self) -> None:
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(self.stage_id, bool) or not isinstance(self.stage_id, int):
            raise TypeError("stage_id must be an int (not bool)")
        if isinstance(self.micro_batch_id, bool) or not isinstance(
            self.micro_batch_id, int
        ):
            raise TypeError("micro_batch_id must be an int (not bool)")
        if self.stage_id < 0:
            raise ValueError("stage_id must be non-negative")
        if self.micro_batch_id < 0:
            raise ValueError("micro_batch_id must be non-negative")
        if self.task_type not in ("FORWARD", "BACKWARD"):
            raise ValueError("task_type must be 'FORWARD' or 'BACKWARD'")

    def key(self) -> TaskKey:
        """Unique identifier used by the dependency graph."""
        return (self.stage_id, self.micro_batch_id, self.task_type)

    def __repr__(self) -> str:
        return f"Stage {self.stage_id} | MB {self.micro_batch_id} | {self.task_type}"


class ScheduleValidationError(RuntimeError):
    """Raised when a schedule or the structure passed to validate() is invalid."""


class Pipeline1F1BScheduler:
    """Build and check 1F1B schedules for S pipeline stages and M micro-batches."""

    def __init__(self, num_stages: int, num_micro_batches: int) -> None:
        _require_positive_int("num_stages", num_stages)
        _require_positive_int("num_micro_batches", num_micro_batches)
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches

    def generate_schedule(self) -> dict[int, list[MicroBatchTask]]:
        """Return a fresh schedule: one ordered task list per stage."""
        S, M = self.num_stages, self.num_micro_batches
        return {stage: _schedule_stage(stage, S, M) for stage in range(S)}

    def validate(self, schedules: dict[int, list[MicroBatchTask]]) -> None:
        """Check a schedule, raising ScheduleValidationError on any problem.

        Checks performed, in order:
          1. The top-level object is a dict keyed by 0..S-1.
          2. Each stage value is a list of MicroBatchTask objects.
          3. Each stage has exactly M forwards and M backwards, in
             ascending order, with forward before backward per
             micro-batch.
          4. Cross-stage dependencies do not form a cycle.
        """
        S, M = self.num_stages, self.num_micro_batches

        # Check 1: outer container.
        if not isinstance(schedules, dict):
            raise ScheduleValidationError(
                f"schedules must be a dict, got {type(schedules).__name__}"
            )
        if set(schedules.keys()) != set(range(S)):
            raise ScheduleValidationError("stage keys must be exactly range(S)")

        # Check 2: each stage's queue.
        for stage in range(S):
            queue = schedules[stage]
            if not isinstance(queue, list):
                raise ScheduleValidationError(
                    f"stage {stage} queue must be a list, got {type(queue).__name__}"
                )
            for idx, task in enumerate(queue):
                if not isinstance(task, MicroBatchTask):
                    raise ScheduleValidationError(
                        f"stage {stage} entry {idx} must be MicroBatchTask, "
                        f"got {type(task).__name__}"
                    )

        _check_counts_and_identity(schedules, S, M)
        _check_dependencies(schedules, S, M)


# --------------------------------------------------------------- generation


def _schedule_stage(stage: int, S: int, M: int) -> list[MicroBatchTask]:
    """Emit the task list for one stage: warm-up, steady state, cool-down."""
    warmup = min(M, max(0, S - 1 - stage))
    tasks: list[MicroBatchTask] = []
    fwd, bwd = 0, 0

    # Warm-up: a block of forwards before any backwards.
    for _ in range(warmup):
        tasks.append(MicroBatchTask(stage, fwd, "FORWARD"))
        fwd += 1

    # Steady state + cool-down: alternate forward and backward until both
    # counters reach M. The last few backwards are the cool-down.
    #
    # When warmup == 0 the loop's first iteration emits a forward (because
    # next_is_backward starts as False). That is why the last stage shows a
    # leading forward run of 1 even though its warm-up parameter is 0.
    next_is_backward = warmup > 0
    while fwd < M or bwd < M:
        if next_is_backward and bwd < M:
            tasks.append(MicroBatchTask(stage, bwd, "BACKWARD"))
            bwd += 1
        elif not next_is_backward and fwd < M:
            tasks.append(MicroBatchTask(stage, fwd, "FORWARD"))
            fwd += 1
        elif bwd < M:
            # Forward side exhausted; finish the remaining backwards.
            tasks.append(MicroBatchTask(stage, bwd, "BACKWARD"))
            bwd += 1
        else:
            # Backward side exhausted; finish the remaining forwards.
            tasks.append(MicroBatchTask(stage, fwd, "FORWARD"))
            fwd += 1
        next_is_backward = not next_is_backward

    return tasks


# --------------------------------------------------------------- validation


def _require_positive_int(name: str, value: int) -> None:
    """Reject anything that is not a positive int (bool is rejected too)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (not bool), got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_counts_and_identity(
    schedules: dict[int, list[MicroBatchTask]], S: int, M: int
) -> None:
    """Per-stage checks: counts, id ranges, ordering, forward-before-backward.

    Duplicate task identities are impossible given these checks. Every
    stage must hold exactly 2*M tasks, and its forward ids must equal
    [0, 1, ..., M-1]. If any forward id were duplicated, some other
    forward id would be missing, and the list-equality test below would
    fail. No separate "seen" set is needed.
    """
    for stage in range(S):
        tasks = schedules[stage]
        if len(tasks) != 2 * M:
            raise ScheduleValidationError(
                f"stage {stage} must have exactly 2*M={2 * M} tasks, got {len(tasks)}"
            )

        fwd_ids: list[int] = []
        bwd_ids: list[int] = []
        positions: dict[tuple[int, str], int] = {}

        for idx, task in enumerate(tasks):
            if task.stage_id != stage:
                raise ScheduleValidationError(
                    f"stage {stage} queue contains task with stage_id={task.stage_id}"
                )
            if not 0 <= task.micro_batch_id < M:
                raise ScheduleValidationError(
                    f"micro-batch id {task.micro_batch_id} outside [0, {M})"
                )
            if task.task_type not in ("FORWARD", "BACKWARD"):
                raise ScheduleValidationError(f"invalid task_type {task.task_type!r}")

            (fwd_ids if task.task_type == "FORWARD" else bwd_ids).append(
                task.micro_batch_id
            )
            positions[(task.micro_batch_id, task.task_type)] = idx

        # Forwards must appear as 0, 1, ..., M-1 in that order.
        if fwd_ids != list(range(M)):
            raise ScheduleValidationError(
                f"stage {stage} forward order must be 0..M-1, got {fwd_ids}"
            )
        # Backwards must appear as 0, 1, ..., M-1 in that order (oldest first).
        if bwd_ids != list(range(M)):
            raise ScheduleValidationError(
                f"stage {stage} backward order must be oldest-first 0..M-1, got {bwd_ids}"
            )
        # The forward for a micro-batch must come before its backward.
        for mb in range(M):
            if positions[(mb, "FORWARD")] > positions[(mb, "BACKWARD")]:
                raise ScheduleValidationError(
                    f"stage {stage}: B({mb}) scheduled before F({mb}) in queue"
                )


def _check_dependencies(
    schedules: dict[int, list[MicroBatchTask]], S: int, M: int
) -> None:
    """Check cross-stage dependencies by building a graph and sorting it.

    Two kinds of check:
      (a) The graph is acyclic -- a topological order exists.
      (b) That order respects the three declared dependency directions:
            F(s, m)  before  F(s+1, m)       (forward flows downstream)
            B(s+1, m) before B(s, m)          (backwards flow upstream)
            F(s, m)  before  B(s, m)          (on the same stage)

    The direction check (b) is redundant with the edges added in (a) by
    construction. It exists so that a source-level change to this function
    that reverses an edge direction is caught, even when the resulting
    graph is still acyclic. No runtime input can reach this check by
    itself; see TEST_STRATEGY.md section 6.
    """
    adj: dict[TaskKey, list[TaskKey]] = {}
    indeg: dict[TaskKey, int] = {}

    def touch(node: TaskKey) -> None:
        adj.setdefault(node, [])
        indeg.setdefault(node, 0)

    def add(u: TaskKey, v: TaskKey) -> None:
        """Add an edge meaning 'u must happen before v'."""
        touch(u)
        touch(v)
        adj[u].append(v)
        indeg[v] += 1

    # Same-stage ordering: each task precedes the next in that stage's queue.
    for stage in range(S):
        tasks = schedules[stage]
        for i in range(len(tasks) - 1):
            add(tasks[i].key(), tasks[i + 1].key())

    # Cross-stage ordering.
    for stage in range(S - 1):
        for mb in range(M):
            # Forward flows from stage s to stage s+1.
            add((stage, mb, "FORWARD"), (stage + 1, mb, "FORWARD"))
            # Backward flows from stage s+1 back to stage s.
            add((stage + 1, mb, "BACKWARD"), (stage, mb, "BACKWARD"))

    # Same-stage: forward before backward for the same micro-batch.
    for stage in range(S):
        for mb in range(M):
            add((stage, mb, "FORWARD"), (stage, mb, "BACKWARD"))

    # Topological sort (Kahn's algorithm).
    ready = deque(n for n, d in indeg.items() if d == 0)
    topo: list[TaskKey] = []
    while ready:
        node = ready.popleft()
        topo.append(node)
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    if len(topo) != len(adj):
        raise ScheduleValidationError("dependency graph contains a cycle")

    # Direction check: verify each declared edge type appears in the
    # expected order in the sorted list.
    pos = {node: i for i, node in enumerate(topo)}
    for s in range(S - 1):
        for mb in range(M):
            if pos[(s, mb, "FORWARD")] >= pos[(s + 1, mb, "FORWARD")]:
                raise ScheduleValidationError(
                    f"forward propagation direction violated: "
                    f"F({s},{mb}) must precede F({s + 1},{mb})"
                )
            if pos[(s + 1, mb, "BACKWARD")] >= pos[(s, mb, "BACKWARD")]:
                raise ScheduleValidationError(
                    f"backward propagation direction violated: "
                    f"B({s + 1},{mb}) must precede B({s},{mb})"
                )
    for s in range(S):
        for mb in range(M):
            if pos[(s, mb, "FORWARD")] >= pos[(s, mb, "BACKWARD")]:
                raise ScheduleValidationError(
                    f"same-stage order violated: F({s},{mb}) must precede B({s},{mb})"
                )
