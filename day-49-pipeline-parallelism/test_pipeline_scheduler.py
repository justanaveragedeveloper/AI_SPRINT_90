"""
Tests for day49_pipeline_scheduler.py.

Each test has a one-line comment above it explaining what wrong behaviour
the test would catch. The [ROBUST] / [CONTRACT] / [GUARD] tag describes
the category of test:

  [ROBUST]   fails if a specific wrong implementation of the scheduler
             or validator is used.
  [CONTRACT] fails if the public interface (field names, exception
             types) changes.
  [GUARD]    fails if input validation stops rejecting bad input.

This file does not run in a GPU environment and does not attempt to test
any real distributed execution. It only checks the schedule generator and
the schedule validator.
"""

import pytest
from day49_pipeline_scheduler import (
    MicroBatchTask,
    Pipeline1F1BScheduler,
    ScheduleValidationError,
)

# ---------------------------------------------------------------------------
# Hand-derived reference schedules.
#
# These expected sequences were written out by hand from the schedule rules
# in the module docstring, not by running generate_schedule(). They are the
# main correctness check for the generator: if the generator ever produces
# a different sequence, one of these rows will fail.
#
# Format: (S, M) -> {stage_id: ["F0", "B0", ...]}
# F<n> means FORWARD on micro-batch n; B<n> means BACKWARD on micro-batch n.
#
# Note on stage 0 for S=2: warm-up is min(M, S-1-0) = min(3, 1) = 1, so
# stage 0 emits one forward before its first backward.
# ---------------------------------------------------------------------------
REFERENCE = {
    (1, 1): {0: ["F0", "B0"]},
    (1, 3): {0: ["F0", "B0", "F1", "B1", "F2", "B2"]},
    (3, 1): {0: ["F0", "B0"], 1: ["F0", "B0"], 2: ["F0", "B0"]},
    (2, 3): {
        0: ["F0", "B0", "F1", "B1", "F2", "B2"],
        1: ["F0", "B0", "F1", "B1", "F2", "B2"],
    },
    (3, 4): {
        0: ["F0", "F1", "B0", "F2", "B1", "F3", "B2", "B3"],
        1: ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3"],
        2: ["F0", "B0", "F1", "B1", "F2", "B2", "F3", "B3"],
    },
    (4, 2): {
        0: ["F0", "F1", "B0", "B1"],
        1: ["F0", "F1", "B0", "B1"],
        2: ["F0", "B0", "F1", "B1"],
        3: ["F0", "B0", "F1", "B1"],
    },
}


def _stringify(schedules):
    """Turn a schedule into plain strings so it is easy to compare by eye."""
    return {
        s: [
            f"{'F' if t.task_type == 'FORWARD' else 'B'}{t.micro_batch_id}"
            for t in tasks
        ]
        for s, tasks in schedules.items()
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


# [GUARD] Catches a constructor that accepts zero stages.
def test_rejects_zero_stages():
    with pytest.raises(ValueError):
        Pipeline1F1BScheduler(0, 4)


# [GUARD] Catches a constructor that accepts zero micro-batches.
def test_rejects_zero_micro_batches():
    with pytest.raises(ValueError):
        Pipeline1F1BScheduler(3, 0)


# [GUARD] Catches a constructor that accepts negative counts.
@pytest.mark.parametrize("s,m", [(-1, 4), (3, -2), (-1, -1)])
def test_rejects_negative_values(s, m):
    with pytest.raises(ValueError):
        Pipeline1F1BScheduler(s, m)


# [GUARD] Catches a constructor that treats True/False as 1/0.
# bool is a subclass of int in Python, so this check has to be explicit.
@pytest.mark.parametrize("s,m", [(True, 4), (3, False), (True, False)])
def test_rejects_bool_values(s, m):
    with pytest.raises(TypeError):
        Pipeline1F1BScheduler(s, m)


# [GUARD] Catches a constructor that accepts floats.
@pytest.mark.parametrize("s,m", [(1.5, 4), (3, 2.0), (1.0, 1.0)])
def test_rejects_float_values(s, m):
    with pytest.raises(TypeError):
        Pipeline1F1BScheduler(s, m)


# [GUARD] Catches MicroBatchTask accepting a task type that is neither
# "FORWARD" nor "BACKWARD".
def test_micro_batch_task_rejects_invalid_type():
    with pytest.raises(ValueError):
        MicroBatchTask(0, 0, "INVALID")


# [GUARD] Catches MicroBatchTask accepting a bool as a stage or micro-batch id.
@pytest.mark.parametrize("s,m", [(True, 0), (0, True)])
def test_micro_batch_task_rejects_bool_ids(s, m):
    with pytest.raises(TypeError):
        MicroBatchTask(s, m, "FORWARD")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


# [CONTRACT] Catches renaming or reordering the public fields of
# MicroBatchTask, or changing what .key() returns.
def test_micro_batch_task_public_fields():
    task = MicroBatchTask(2, 3, "FORWARD")
    assert task.stage_id == 2
    assert task.micro_batch_id == 3
    assert task.task_type == "FORWARD"
    assert task.key() == (2, 3, "FORWARD")


# ---------------------------------------------------------------------------
# Correctness of the generated schedule
# ---------------------------------------------------------------------------


# [ROBUST] Catches any change to the task sequence that would break the
# hand-derived reference. Also runs validate() on the generated schedule so
# the validator is exercised on valid input.
@pytest.mark.parametrize("S,M", list(REFERENCE.keys()))
def test_matches_hand_written_reference(S, M):
    sched = Pipeline1F1BScheduler(S, M)
    schedules = sched.generate_schedule()
    sched.validate(schedules)
    got = _stringify(schedules)
    assert got == REFERENCE[(S, M)]


# [ROBUST] Catches wrong task counts, duplicated task identities, tasks
# assigned to the wrong stage, and out-of-range micro-batch ids.
@pytest.mark.parametrize("S,M", [(1, 1), (1, 5), (2, 3), (3, 4), (4, 2), (5, 5)])
def test_identity_and_counts(S, M):
    schedules = Pipeline1F1BScheduler(S, M).generate_schedule()
    assert set(schedules.keys()) == set(range(S))
    seen = set()
    for stage, tasks in schedules.items():
        fwd = [t for t in tasks if t.task_type == "FORWARD"]
        bwd = [t for t in tasks if t.task_type == "BACKWARD"]
        assert len(fwd) == M
        assert len(bwd) == M
        for t in tasks:
            assert t.stage_id == stage
            assert 0 <= t.micro_batch_id < M
            key = t.key()
            assert key not in seen
            seen.add(key)


# [ROBUST] Catches a warm-up formula that gives stage 0 fewer leading
# forwards than stage S-1. Note: the last stage's warm-up parameter is 0,
# but the alternation loop still emits one forward before its first
# backward, so the last stage's leading forward run is 1.
@pytest.mark.parametrize("S,M", [(2, 1), (2, 3), (3, 4), (4, 2), (5, 3)])
def test_warmup_direction(S, M):
    schedules = Pipeline1F1BScheduler(S, M).generate_schedule()

    def leading_forwards(tasks):
        """Count leading FORWARD tasks before the first BACKWARD."""
        n = 0
        for t in tasks:
            if t.task_type == "FORWARD":
                n += 1
            else:
                break
        return n

    w0 = leading_forwards(schedules[0])
    wlast = leading_forwards(schedules[S - 1])

    # Stage 0 warms up the most: min(M, S-1) forwards before any backward.
    assert w0 == min(M, S - 1)

    # The last stage's warm-up parameter is 0, but the alternation loop
    # starts with a forward, so its leading forward run is exactly 1.
    assert wlast == 1

    # Stage 0 must not warm up less than the last stage.
    assert w0 >= wlast


# [ROBUST] Catches a scheduler that emits all forwards first and all
# backwards afterwards ("block schedule") instead of alternating.
# Looks at the region between the end of warm-up and the last forward:
# in that window, adjacent tasks must have different types.
@pytest.mark.parametrize("S,M", [(2, 3), (3, 4), (4, 2), (5, 3)])
def test_steady_state_alternates(S, M):
    schedules = Pipeline1F1BScheduler(S, M).generate_schedule()
    for stage, tasks in schedules.items():
        warmup = min(M, max(0, S - 1 - stage))
        last_f = max(i for i, t in enumerate(tasks) if t.task_type == "FORWARD")
        active = [t.task_type for t in tasks[warmup : last_f + 1]]
        for i in range(len(active) - 1):
            assert active[i] != active[i + 1], (
                f"stage {stage}: non-alternating at index {i}: {active}"
            )


# [ROBUST] Catches a scheduler that runs backwards newest-first instead of
# oldest-first.
@pytest.mark.parametrize("S,M", list(REFERENCE.keys()))
def test_backward_is_oldest_first(S, M):
    for tasks in Pipeline1F1BScheduler(S, M).generate_schedule().values():
        bwd_ids = [t.micro_batch_id for t in tasks if t.task_type == "BACKWARD"]
        assert bwd_ids == list(range(M))


# [ROBUST] Catches a scheduler that puts a micro-batch's backward before
# its own forward within the same stage.
@pytest.mark.parametrize("S,M", list(REFERENCE.keys()))
def test_forward_precedes_backward_same_stage(S, M):
    for tasks in Pipeline1F1BScheduler(S, M).generate_schedule().values():
        positions = {}
        for i, t in enumerate(tasks):
            positions[(t.micro_batch_id, t.task_type)] = i
        for mb in range(M):
            assert positions[(mb, "FORWARD")] < positions[(mb, "BACKWARD")]


# [ROBUST] Catches a scheduler that returns the same list or the same task
# objects across calls, which would leak mutations between callers.
def test_repeated_calls_are_independent():
    scheduler = Pipeline1F1BScheduler(3, 4)
    first = scheduler.generate_schedule()
    second = scheduler.generate_schedule()
    assert first == second
    assert first is not second
    assert first[0] is not second[0]
    first[0].append(MicroBatchTask(0, 0, "FORWARD"))
    assert second[0] != first[0]


# ---------------------------------------------------------------------------
# The validator must reject broken schedules
# ---------------------------------------------------------------------------


# [ROBUST] Catches a validator that accepts a schedule missing a task.
def test_validator_rejects_missing_task():
    sched = Pipeline1F1BScheduler(3, 4)
    schedules = sched.generate_schedule()
    schedules[0] = [
        t
        for t in schedules[0]
        if not (t.task_type == "BACKWARD" and t.micro_batch_id == 0)
    ]
    with pytest.raises(ScheduleValidationError):
        sched.validate(schedules)


# [ROBUST] Catches a validator that allows backward before forward on the
# same stage.
def test_validator_rejects_same_stage_backward_before_forward():
    sched = Pipeline1F1BScheduler(1, 1)
    corrupted = {0: [MicroBatchTask(0, 0, "BACKWARD"), MicroBatchTask(0, 0, "FORWARD")]}
    with pytest.raises(ScheduleValidationError):
        sched.validate(corrupted)


# [ROBUST] Catches a validator whose cross-stage dependency edges are
# missing, reversed, or simply not checked. Every per-stage rule passes
# in the corrupted input; the conflict exists only across stages:
#
#     B(0,0) -> F(0,1) -> F(1,1) -> B(1,0) -> B(0,0)
#
# That is a cycle. The validator must reject it.
def test_validator_rejects_cross_stage_dependency_cycle():
    sched = Pipeline1F1BScheduler(2, 2)
    corrupted = {
        0: [
            MicroBatchTask(0, 0, "FORWARD"),
            MicroBatchTask(0, 0, "BACKWARD"),
            MicroBatchTask(0, 1, "FORWARD"),
            MicroBatchTask(0, 1, "BACKWARD"),
        ],
        1: [
            MicroBatchTask(1, 0, "FORWARD"),
            MicroBatchTask(1, 1, "FORWARD"),
            MicroBatchTask(1, 0, "BACKWARD"),
            MicroBatchTask(1, 1, "BACKWARD"),
        ],
    }
    with pytest.raises(ScheduleValidationError, match="cycle"):
        sched.validate(corrupted)


# [ROBUST] Catches a validator that ignores micro-batch ids outside [0, M).
def test_validator_rejects_out_of_range_micro_batch_id():
    sched = Pipeline1F1BScheduler(1, 1)
    corrupted = {0: [MicroBatchTask(0, 5, "FORWARD"), MicroBatchTask(0, 0, "BACKWARD")]}
    with pytest.raises(ScheduleValidationError):
        sched.validate(corrupted)


# [ROBUST] Catches a validator that ignores a task placed in the wrong
# stage's queue.
def test_validator_rejects_wrong_stage_id():
    sched = Pipeline1F1BScheduler(2, 1)
    corrupted = {
        0: [MicroBatchTask(1, 0, "FORWARD"), MicroBatchTask(0, 0, "BACKWARD")],
        1: [MicroBatchTask(1, 0, "FORWARD"), MicroBatchTask(1, 0, "BACKWARD")],
    }
    with pytest.raises(ScheduleValidationError):
        sched.validate(corrupted)


# ---------------------------------------------------------------------------
# The validator must reject malformed input
# ---------------------------------------------------------------------------


# [ROBUST] Catches a validator that lets an AttributeError leak out when
# the top-level argument is not a dict.
def test_validator_rejects_non_dict():
    sched = Pipeline1F1BScheduler(1, 1)
    with pytest.raises(ScheduleValidationError):
        sched.validate([])  # type: ignore[arg-type]


# [ROBUST] Catches a validator that accepts a stage queue that is not a list.
def test_validator_rejects_non_list_queue():
    sched = Pipeline1F1BScheduler(1, 1)
    corrupted = (MicroBatchTask(0, 0, "FORWARD"), MicroBatchTask(0, 0, "BACKWARD"))
    with pytest.raises(ScheduleValidationError):
        sched.validate({0: corrupted})  # type: ignore[dict-item]


# [ROBUST] Catches a validator that dereferences arbitrary objects in a
# queue instead of checking the type first.
def test_validator_rejects_non_task_entry():
    sched = Pipeline1F1BScheduler(1, 1)
    with pytest.raises(ScheduleValidationError):
        sched.validate({0: ["not a task", "also not a task"]})  # type: ignore[list-item]


# [ROBUST] Catches a validator that accepts a dict with missing stage keys.
def test_validator_rejects_missing_stage_key():
    sched = Pipeline1F1BScheduler(2, 1)
    with pytest.raises(ScheduleValidationError):
        sched.validate(
            {0: [MicroBatchTask(0, 0, "FORWARD"), MicroBatchTask(0, 0, "BACKWARD")]}
        )


# ---------------------------------------------------------------------------
# Property test (Hypothesis) -- commented out because Hypothesis is not a
# required dependency for this exercise.
# ---------------------------------------------------------------------------
#
# The property: for any positive S and M (up to a small bound), the
# generated schedule must have the following shape.
#
#   - The stage keys are exactly 0 .. S-1.
#   - Every stage contains exactly M forwards and M backwards.
#   - Every (stage, micro_batch, task_type) tuple appears only once.
#   - Every micro-batch id is in range(M).
#   - Forward ids and backward ids per stage are each in ascending order.
#   - validate() raises nothing on the generated schedule.
#   - A second call returns an equal schedule but not the same objects.
#
# from hypothesis import given, strategies as st
#
# @given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
# def test_schedule_properties(S, M):
#     sched = Pipeline1F1BScheduler(S, M)
#     s1 = sched.generate_schedule()
#     s2 = sched.generate_schedule()
#     sched.validate(s1)
#     assert s1 == s2 and s1 is not s2
#     for stage, tasks in s1.items():
#         fwd = [t.micro_batch_id for t in tasks if t.task_type == "FORWARD"]
#         bwd = [t.micro_batch_id for t in tasks if t.task_type == "BACKWARD"]
#         assert fwd == list(range(M))
#         assert bwd == list(range(M))


# Not tested, and why:
#
# - Actual GPU execution, tensor movement, or communication timing:
#   this module only emits a plan; there is no hardware model here.
# - Autograd or gradient correctness: no gradients are computed.
# - Activation-memory measurement: the generator does not allocate
#   tensors, so it cannot demonstrate any memory bound.
# - Real cross-stage concurrency: validate() proves a topological order
#   exists for the declared dependencies; it does not prove that real
#   hardware can run the queues without resource conflicts.
# - The direction check inside _check_dependencies, after the topological
#   sort: no runtime input reaches it. It is a defense against a source
#   level change to the DAG builder that reverses an edge, and its only
#   observable effect on valid input is that it does not raise.
# - Reversing a single dependency edge in the source of
#   _check_dependencies: no test mutates the source. This would require a
#   source-level mutation harness, which is out of scope for this file.
# - Very large S and M: the hand-written reference table cannot be
#   enumerated.
# - TEST_STRATEGY.md consistency: only the module docstring is checked
#   here.
