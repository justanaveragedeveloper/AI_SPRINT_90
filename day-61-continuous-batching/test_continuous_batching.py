"""Test suite for the Day 61 continuous-batching scheduler.

Every test names a specific wrong implementation it catches.

Tags (reviewer taxonomy):
  [ROBUST]   catches a specific mutated implementation
  [CONTRACT] catches a changed public interface
  [THEATER]  asserts only the happy path or restates the docstring

Property strategies are described as comment blocks; the `hypothesis`
dependency is optional.
"""

import pytest
from day61_continuous_batching import (
    IterationBatch,
    IterationScheduler,
    PrefillChunk,
    SequenceGroup,
    State,
)

# =========================================================================
# Backfill and retirement
# =========================================================================


def test_backfill_queued_request_enters_after_decode_slot_frees():
    # [ROBUST]
    # Fails if the scheduler removes a finished request but does NOT admit a
    # waiting request into the freed slot at the next schedule_iteration.
    scheduler = IterationScheduler(
        max_batch_size=2, max_tokens_per_batch=100, prefill_chunk_size=8
    )
    first = SequenceGroup("first", [1, 2], max_new_tokens=2)
    second = SequenceGroup("second", [3], max_new_tokens=1)
    third = SequenceGroup("third", [4, 5], max_new_tokens=2)
    for request in (first, second, third):
        scheduler.add_request(request)

    batch1 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch1.active_requests] == ["first", "second"]
    assert third.state is State.WAITING

    batch2 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch2.decode_requests] == ["first", "second"]
    second.step(token=99)
    assert second.is_finished()

    batch3 = scheduler.schedule_iteration()
    scheduled_ids = [request.req_id for request in batch3.active_requests]
    assert "second" not in scheduled_ids, "finished request reappeared in active batch"
    assert "third" in scheduled_ids, "waiting request was NOT backfilled after retire"


def test_free_finished_requests_not_required_for_backfill():
    # [ROBUST]
    # Fails if backfill requires an explicit free_finished_requests() call.
    scheduler = IterationScheduler(max_batch_size=1, max_tokens_per_batch=10)
    first = SequenceGroup("first", [1], max_new_tokens=1)
    second = SequenceGroup("second", [2], max_new_tokens=1)
    scheduler.add_request(first)
    scheduler.add_request(second)

    scheduler.schedule_iteration()
    scheduler.schedule_iteration()
    first.step(token=0)
    assert first.is_finished()
    # Deliberately not calling free_finished_requests().
    batch = scheduler.schedule_iteration()
    assert [request.req_id for request in batch.active_requests] == ["second"]


def test_free_finished_requests_is_idempotent():
    # [ROBUST]
    # Fails if free_finished_requests raises on a second call or resurrects
    # a removed request.
    scheduler = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    request = SequenceGroup("request", [1], max_new_tokens=1)
    scheduler.add_request(request)
    scheduler.schedule_iteration()
    request.step(token=0)
    assert request.is_finished()

    scheduler.free_finished_requests()
    scheduler.free_finished_requests()

    for _ in range(3):
        batch = scheduler.schedule_iteration()
        assert request not in batch.active_requests


# =========================================================================
# Ordering and precedence
# =========================================================================


def test_decode_precedence_over_prefill_under_tight_budget():
    # [ROBUST]
    # Fails if prefill work is admitted before decode work, starving decodes.
    scheduler = IterationScheduler(
        max_batch_size=2, max_tokens_per_batch=2, prefill_chunk_size=10
    )
    first = SequenceGroup("first", [1], max_new_tokens=2)
    second = SequenceGroup("second", [2], max_new_tokens=2)
    third = SequenceGroup("third", [3, 4, 5], max_new_tokens=2)
    for request in (first, second, third):
        scheduler.add_request(request)

    batch1 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch1.active_requests] == ["first", "second"]

    batch2 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch2.decode_requests] == ["first", "second"]
    assert len(batch2.prefill_chunks) == 0
    assert third.state is State.WAITING


def test_fifo_admission_order_among_waiting_requests():
    # [ROBUST]
    # Fails if waiting requests are reordered by length, priority, or by a
    # set/dict iteration artifact.
    #
    # NOTE: this test explicitly steps the first two requests to FINISHED
    # before asserting that `third` enters. An earlier version of this test
    # omitted those steps and produced a false failure against a correct
    # implementation: the first two requests were still in DECODING and
    # correctly occupied both decode slots in iteration 4.
    scheduler = IterationScheduler(
        max_batch_size=2, max_tokens_per_batch=100, prefill_chunk_size=1
    )
    first = SequenceGroup("first", [1, 2], max_new_tokens=1)
    second = SequenceGroup("second", [3, 4], max_new_tokens=1)
    third = SequenceGroup("third", [5, 6], max_new_tokens=1)
    for request in (first, second, third):
        scheduler.add_request(request)

    # Iteration 1: first half of prefill for first and second; third waits.
    batch1 = scheduler.schedule_iteration()
    assert [chunk.req_id for chunk in batch1.prefill_chunks] == ["first", "second"]
    assert third.state is State.WAITING

    # Iteration 2: second half of prefill; both transition to DECODING.
    batch2 = scheduler.schedule_iteration()
    assert [chunk.req_id for chunk in batch2.prefill_chunks] == ["first", "second"]
    assert first.state is State.DECODING
    assert second.state is State.DECODING

    # Iteration 3: decodes for first and second; third still waits.
    batch3 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch3.decode_requests] == ["first", "second"]
    first.step(token=10)
    second.step(token=11)
    assert first.is_finished()
    assert second.is_finished()

    # Iteration 4: first and second removed; third enters prefill.
    batch4 = scheduler.schedule_iteration()
    assert [chunk.req_id for chunk in batch4.prefill_chunks] == ["third"]


def test_prefill_chunks_emitted_in_fifo_admission_order():
    # [ROBUST]
    # Fails if prefill chunks from different requests are emitted in any
    # order other than admission order.
    scheduler = IterationScheduler(
        max_batch_size=3, max_tokens_per_batch=100, prefill_chunk_size=1
    )
    for index in range(3):
        scheduler.add_request(
            SequenceGroup(f"request_{index}", [index], max_new_tokens=1)
        )
    batch = scheduler.schedule_iteration()
    assert [chunk.req_id for chunk in batch.prefill_chunks] == [
        "request_0",
        "request_1",
        "request_2",
    ]


def test_decode_order_follows_admission_not_activation():
    # [ROBUST]
    # Fails if the sort by _admission_order in schedule_iteration is
    # removed: an empty-prompt request admitted second would decode before
    # a long-prompt request admitted first, which only activates in a later
    # iteration.
    scheduler = IterationScheduler(2, 100, prefill_chunk_size=1)
    long_prompt = SequenceGroup("long_prompt", [1, 2], 5)
    empty_prompt = SequenceGroup("empty_prompt", [], 5)
    scheduler.add_request(long_prompt)
    scheduler.add_request(empty_prompt)
    scheduler.schedule_iteration()  # empty_prompt activates; long_prompt chunk 1
    scheduler.schedule_iteration()  # long_prompt chunk 2 -> DECODING
    batch = scheduler.schedule_iteration()
    assert [request.req_id for request in batch.decode_requests] == [
        "long_prompt",
        "empty_prompt",
    ]


def test_same_process_repeated_operations_produce_identical_trace():
    # [ROBUST]
    # Checks repeatability for identical operations in this process.
    def run():
        scheduler = IterationScheduler(
            max_batch_size=3, max_tokens_per_batch=20, prefill_chunk_size=2
        )
        for index, (prompt, limit) in enumerate(
            [([1, 2, 3], 2), ([4, 5], 3), ([6], 1), ([7, 8, 9], 2)]
        ):
            scheduler.add_request(
                SequenceGroup(f"request_{index}", prompt, max_new_tokens=limit)
            )
        trace = []
        for _ in range(20):
            batch = scheduler.schedule_iteration()
            trace.append(
                (
                    tuple(request.req_id for request in batch.decode_requests),
                    tuple(
                        (chunk.req_id, chunk.start, chunk.end)
                        for chunk in batch.prefill_chunks
                    ),
                )
            )
            for request in batch.decode_requests:
                if not request.is_finished():
                    request.step(token=0)
        return trace

    assert run() == run()


# =========================================================================
# Chunked prefill correctness
# =========================================================================


def test_prefill_chunks_cover_prompt_exactly_once():
    # [ROBUST]
    # Fails if prefill chunks skip, duplicate, or reorder prompt tokens; or
    # if progress is not retained across iterations.
    prompt = list(range(1, 11))
    request = SequenceGroup("request", prompt, max_new_tokens=1)
    scheduler = IterationScheduler(
        max_batch_size=10, max_tokens_per_batch=100, prefill_chunk_size=3
    )
    scheduler.add_request(request)

    collected = []
    for _ in range(20):
        if request.state is State.DECODING:
            break
        batch = scheduler.schedule_iteration()
        for chunk in batch.prefill_chunks:
            if chunk.request is request:
                collected.extend(chunk.tokens)
    assert collected == prompt


def test_prefill_chunk_start_end_indices_are_contiguous_and_ordered():
    # [ROBUST]
    # Fails if start/end indices do not match the actual tokens (off-by-one,
    # overlapping, or non-contiguous coverage).
    prompt = [10, 20, 30, 40, 50, 60, 70]
    request = SequenceGroup("request", prompt, max_new_tokens=1)
    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=100, prefill_chunk_size=2
    )
    scheduler.add_request(request)

    collected_chunks = []
    for _ in range(10):
        if request.state is State.DECODING:
            break
        batch = scheduler.schedule_iteration()
        collected_chunks.extend(batch.prefill_chunks)

    expected_start = 0
    for chunk in collected_chunks:
        assert chunk.start == expected_start
        assert chunk.end > chunk.start
        assert chunk.tokens == tuple(prompt[chunk.start : chunk.end])
        expected_start = chunk.end
    assert expected_start == len(prompt)


def test_state_is_prefilling_until_final_chunk():
    # [ROBUST]
    # Fails if a request transitions to DECODING before the final prefill
    # chunk is scheduled.
    request = SequenceGroup("request", [1, 2, 3, 4], max_new_tokens=1)
    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=100, prefill_chunk_size=2
    )
    scheduler.add_request(request)
    scheduler.schedule_iteration()
    assert request.state is State.PREFILLING
    assert request.prefilled_token_count == 2

    scheduler.schedule_iteration()
    assert request.state is State.DECODING
    assert request.prefilled_token_count == 4


def test_prefill_chunk_size_can_be_configured_independently():
    # [ROBUST]
    # Fails if prefill_chunk_size is ignored and the prompt is always
    # prefilled in one shot.
    prompt = [1, 2, 3, 4]
    request = SequenceGroup("request", prompt, max_new_tokens=1)
    scheduler = IterationScheduler(
        max_batch_size=10, max_tokens_per_batch=100, prefill_chunk_size=1
    )
    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert len(batch.prefill_chunks) == 1
    assert batch.prefill_chunks[0].length == 1
    assert request.prefilled_token_count == 1
    assert request.state is State.PREFILLING


# =========================================================================
# Budgets
# =========================================================================


def test_batch_item_budget_selects_exactly_first_n_by_admission():
    # [ROBUST]
    # Fails if the scheduler schedules MORE than max_batch_size, FEWER than
    # max_batch_size when more are available, or selects items out of
    # admission order.
    scheduler = IterationScheduler(
        max_batch_size=2, max_tokens_per_batch=1000, prefill_chunk_size=1
    )
    for index in range(4):
        scheduler.add_request(
            SequenceGroup(f"request_{index}", [index], max_new_tokens=3)
        )
    batch = scheduler.schedule_iteration()
    assert batch.num_items == 2
    assert [request.req_id for request in batch.active_requests] == [
        "request_0",
        "request_1",
    ]


def test_batch_size_cap_with_more_waiting_requests_than_slots():
    # [ROBUST]
    # Fails if item budget is off-by-one: either allowing max+1 or scheduling
    # fewer than max when more than max items are available. This case uses
    # prefill-only work; the mixed case is covered by
    # `test_mixed_decode_and_prefill_respect_both_budgets`.
    scheduler = IterationScheduler(
        max_batch_size=3, max_tokens_per_batch=1000, prefill_chunk_size=1
    )
    for index in range(5):
        scheduler.add_request(
            SequenceGroup(f"request_{index}", [index], max_new_tokens=1)
        )
    batch = scheduler.schedule_iteration()
    assert batch.num_items == 3
    assert [request.req_id for request in batch.active_requests] == [
        "request_0",
        "request_1",
        "request_2",
    ]


def test_token_budget_counts_prefill_tokens_not_chunks():
    # [ROBUST]
    # Fails if prefill tokens are counted as one per chunk (a common
    # mutation: subtracting len(prefill_chunks) instead of the chunk length).
    scheduler = IterationScheduler(
        max_batch_size=10, max_tokens_per_batch=5, prefill_chunk_size=10
    )
    request = SequenceGroup("request", [1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=1)
    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert batch.token_count <= 5
    assert len(batch.prefill_chunks) == 1
    assert batch.prefill_chunks[0].length == 5
    assert batch.prefill_chunks[0].tokens == (1, 2, 3, 4, 5)


def test_decode_tokens_are_counted_against_token_budget():
    # [ROBUST]
    # Fails if decode tokens are not decremented from max_tokens_per_batch
    # (a common mutation: only prefill tokens touch the budget).
    scheduler = IterationScheduler(
        max_batch_size=10, max_tokens_per_batch=3, prefill_chunk_size=10
    )
    for index in range(4):
        scheduler.add_request(SequenceGroup(f"request_{index}", [], max_new_tokens=2))
    batch = scheduler.schedule_iteration()
    assert len(batch.decode_requests) == 3
    assert batch.token_count == 3


def test_mixed_decode_and_prefill_respect_both_budgets():
    # [ROBUST]
    # Fails if (a) the item budget is exceeded when a decode and a prefill
    # share an iteration, (b) the token budget is exceeded, or (c) the
    # prefill is not shortened to fit the remaining budget after the decode
    # consumes one token.
    scheduler = IterationScheduler(
        max_batch_size=2, max_tokens_per_batch=3, prefill_chunk_size=10
    )
    decode_request = SequenceGroup("decode_request", [], max_new_tokens=5)
    prefill_request = SequenceGroup("prefill_request", [1, 2, 3, 4], max_new_tokens=5)
    scheduler.add_request(decode_request)
    scheduler.add_request(prefill_request)

    batch1 = scheduler.schedule_iteration()
    assert [request.req_id for request in batch1.decode_requests] == ["decode_request"]
    assert [chunk.req_id for chunk in batch1.prefill_chunks] == ["prefill_request"]
    assert batch1.num_items == 2
    assert batch1.token_count == 3
    assert batch1.prefill_chunks[0].length == 2
    assert batch1.num_items <= scheduler.max_batch_size
    assert batch1.token_count <= scheduler.max_tokens_per_batch


def test_prefill_chunk_shortened_when_token_budget_smaller_than_chunk_size():
    # [ROBUST]
    # Fails if a chunk larger than the remaining token budget is emitted
    # (budget overrun) or is deferred intact (starvation when the budget
    # never grows). We require shorten-to-fit.
    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=3, prefill_chunk_size=10
    )
    request = SequenceGroup("request", [10, 20, 30, 40, 50], max_new_tokens=1)
    scheduler.add_request(request)

    batch1 = scheduler.schedule_iteration()
    assert len(batch1.prefill_chunks) == 1
    assert batch1.prefill_chunks[0].length == 3
    assert batch1.prefill_chunks[0].tokens == (10, 20, 30)
    assert request.prefilled_token_count == 3

    batch2 = scheduler.schedule_iteration()
    assert len(batch2.prefill_chunks) == 1
    assert batch2.prefill_chunks[0].length == 2
    assert batch2.prefill_chunks[0].tokens == (40, 50)
    assert request.state is State.DECODING


def test_exact_boundary_prefill_chunk_fills_budget():
    # [ROBUST]
    # Fails if the scheduler refuses to schedule a chunk that EXACTLY
    # consumes the token budget (off-by-one: strict `<` instead of `<=`).
    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=4, prefill_chunk_size=4
    )
    request = SequenceGroup("request", [1, 2, 3, 4, 5], max_new_tokens=1)
    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert len(batch.prefill_chunks) == 1
    assert batch.prefill_chunks[0].length == 4
    assert batch.token_count == 4


# =========================================================================
# Edge cases
# =========================================================================


def test_zero_max_new_tokens_finishes_without_decode_slot():
    # [ROBUST]
    # Fails if a max_new_tokens=0 request ever takes a decode slot or
    # generates a token.
    scheduler = IterationScheduler(max_batch_size=4, max_tokens_per_batch=100)
    request = SequenceGroup("request", [1, 2, 3], max_new_tokens=0)
    scheduler.add_request(request)
    assert request.is_finished()
    assert request.num_generated == 0
    for _ in range(3):
        batch = scheduler.schedule_iteration()
        assert request not in batch.decode_requests
        assert request not in batch.active_requests
    assert request.num_generated == 0


def test_empty_prompt_enters_decoding_without_prefill_chunk():
    # [ROBUST]
    # Fails if an empty prompt generates a zero-length prefill chunk (which
    # would waste a batch item) or is rejected / never scheduled.
    scheduler = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    request = SequenceGroup("request", [], max_new_tokens=2)
    scheduler.add_request(request)
    assert request.state is State.WAITING

    batch1 = scheduler.schedule_iteration()
    assert request.state is State.DECODING
    assert request in batch1.decode_requests
    assert len(batch1.prefill_chunks) == 0

    request.step(token=7)
    request.step(token=8)
    assert request.is_finished()


def test_step_rejects_when_not_decoding_and_when_finished():
    # [ROBUST]
    # Fails if step() silently accepts a WAITING or PREFILLING request, or
    # allows over-generation past max_new_tokens.
    request = SequenceGroup("request", [1, 2], max_new_tokens=5)
    with pytest.raises(RuntimeError):
        request.step(token=7)  # WAITING

    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=10, prefill_chunk_size=1
    )
    scheduler.add_request(request)
    scheduler.schedule_iteration()
    with pytest.raises(RuntimeError):
        request.step(token=7)  # PREFILLING
    scheduler.schedule_iteration()
    for index in range(5):
        request.step(token=100 + index)
    assert request.is_finished()
    with pytest.raises(RuntimeError):
        request.step(token=999)  # FINISHED


def test_request_finishes_on_exact_max_new_tokens_th_step():
    # [ROBUST]
    # Fails if the request finishes one step too early or one step too late.
    request = SequenceGroup("request", [1], max_new_tokens=3)
    scheduler = IterationScheduler(max_batch_size=1, max_tokens_per_batch=10)
    scheduler.add_request(request)
    scheduler.schedule_iteration()

    for index in range(3):
        batch = scheduler.schedule_iteration()
        assert request in batch.decode_requests
        assert not request.is_finished(), f"premature finish before step {index + 1}"
        request.step(token=200 + index)
        if index < 2:
            assert not request.is_finished(), f"premature finish after step {index + 1}"
    assert request.is_finished()
    assert request.num_generated == 3


def test_caller_mutation_of_prompt_after_construction_does_not_affect_scheduling():
    # [ROBUST]
    # Fails if prompt_tokens is stored by reference and later caller
    # mutations silently alter already-admitted prefill work.
    prompt = [1, 2, 3]
    request = SequenceGroup("request", prompt, max_new_tokens=1)
    prompt.append(999)
    prompt[0] = -1

    scheduler = IterationScheduler(
        max_batch_size=1, max_tokens_per_batch=100, prefill_chunk_size=10
    )
    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert batch.prefill_chunks[0].tokens == (1, 2, 3)
    assert request.prompt_tokens == (1, 2, 3)


def test_eos_token_finishes_request_immediately():
    # [ROBUST]
    # Fails if EOS handling is missing, uses a magic default, or fails to
    # terminate the request after the configured token is generated.
    request = SequenceGroup("eos_request", [], max_new_tokens=10, eos_token=99)
    scheduler = IterationScheduler(max_batch_size=1, max_tokens_per_batch=10)
    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert request in batch.decode_requests
    request.step(token=10)
    assert not request.is_finished()
    request.step(token=99)
    assert request.is_finished()
    assert request.num_generated == 2


# =========================================================================
# Contract tests
# =========================================================================


def test_public_api_names_and_shapes_are_stable():
    # [CONTRACT]
    # Fails if required public names, attribute types, or return contracts
    # change.
    scheduler = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    request = SequenceGroup("request", [1, 2], max_new_tokens=1)
    assert request.req_id == "request"
    assert request.max_new_tokens == 1
    assert isinstance(request.prompt_tokens, tuple)

    scheduler.add_request(request)
    batch = scheduler.schedule_iteration()
    assert isinstance(batch, IterationBatch)
    assert isinstance(batch.active_requests, tuple)
    assert isinstance(batch.decode_requests, tuple)
    assert isinstance(batch.prefill_chunks, tuple)
    assert all(isinstance(chunk, PrefillChunk) for chunk in batch.prefill_chunks)
    assert scheduler.free_finished_requests() is None


def test_prompt_tokens_accepts_any_sequence_of_ints():
    # [CONTRACT]
    # Fails if the constructor claims `Sequence[int]` but rejects `range`
    # (a legitimate Sequence[int]) with a type error.
    request = SequenceGroup("request", range(5), max_new_tokens=1)  # type: ignore[arg-type]
    assert request.prompt_tokens == (0, 1, 2, 3, 4)


# =========================================================================
# Validation tests
# =========================================================================


def test_duplicate_live_req_id_rejected():
    # [ROBUST]
    # Fails if add_request accepts a duplicate ID (as distinct objects) while
    # the first is live.
    scheduler = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    scheduler.add_request(SequenceGroup("duplicate_id", [1], max_new_tokens=1))
    with pytest.raises(ValueError):
        scheduler.add_request(SequenceGroup("duplicate_id", [2], max_new_tokens=1))


def test_request_object_cannot_be_admitted_twice():
    # [ROBUST]
    # Fails if the same SequenceGroup object can be admitted into the same
    # scheduler twice, or into two schedulers, sharing mutable state.
    request = SequenceGroup("request", [1, 2], max_new_tokens=2)
    scheduler_a = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    scheduler_b = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)

    scheduler_a.add_request(request)
    scheduler_a.schedule_iteration()  # advances the shared request's prefill
    with pytest.raises(ValueError):
        scheduler_a.add_request(request)
    with pytest.raises(ValueError):
        scheduler_b.add_request(request)


def test_lifecycle_state_is_read_only_to_callers():
    # [ROBUST]
    # Fails if outside code can force a request into an arbitrary state by
    # assigning `request.state = ...`, bypassing the lifecycle.
    request = SequenceGroup("request", [1], max_new_tokens=1)
    with pytest.raises(AttributeError):
        request.state = State.DECODING  # type: ignore[misc]


def test_invalid_scheduler_params_rejected():
    # [ROBUST]
    # Fails if zero/negative limits, bool limits, or non-int limits are
    # accepted.
    with pytest.raises(ValueError):
        IterationScheduler(max_batch_size=0, max_tokens_per_batch=10)
    with pytest.raises(ValueError):
        IterationScheduler(max_batch_size=2, max_tokens_per_batch=0)
    with pytest.raises(ValueError):
        IterationScheduler(
            max_batch_size=2, max_tokens_per_batch=10, prefill_chunk_size=0
        )
    with pytest.raises(TypeError):
        IterationScheduler(max_batch_size=True, max_tokens_per_batch=10)
    with pytest.raises(TypeError):
        IterationScheduler(max_batch_size=2, max_tokens_per_batch=True)
    with pytest.raises(TypeError):
        IterationScheduler(
            max_batch_size=2,
            max_tokens_per_batch=10,
            prefill_chunk_size=True,
        )
    with pytest.raises(TypeError):
        IterationScheduler(max_batch_size=2.5, max_tokens_per_batch=10)


def test_invalid_request_params_rejected():
    # [ROBUST]
    # Fails if empty req_id, non-sequence prompt, non-int prompt token, or
    # negative/bool max_new_tokens is accepted.
    with pytest.raises(ValueError):
        SequenceGroup("", [1], max_new_tokens=1)
    with pytest.raises(TypeError):
        SequenceGroup("request", "not a list", max_new_tokens=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SequenceGroup("request", [1, "two"], max_new_tokens=1)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        SequenceGroup("request", [1], max_new_tokens=-1)
    with pytest.raises(TypeError):
        SequenceGroup("request", [1], max_new_tokens=True)


def test_boolean_token_ids_are_rejected():
    # [ROBUST]
    # Fails if bool is silently accepted as an int for prompt tokens or EOS
    # (Python: isinstance(True, int) is True).
    with pytest.raises(TypeError):
        SequenceGroup("request", [True], max_new_tokens=1)
    with pytest.raises(TypeError):
        SequenceGroup("request", [1, False, 3], max_new_tokens=1)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        SequenceGroup("request", [1], max_new_tokens=1, eos_token=True)  # type: ignore[arg-type]


def test_step_rejects_non_integer_token():
    # [ROBUST]
    # Fails if step() accepts non-int tokens (strings, floats, bools).
    request = SequenceGroup("request", [1], max_new_tokens=3)
    scheduler = IterationScheduler(max_batch_size=1, max_tokens_per_batch=10)
    scheduler.add_request(request)
    scheduler.schedule_iteration()  # DECODING

    with pytest.raises(TypeError):
        request.step(token="hello")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        request.step(token=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        request.step(token=True)  # type: ignore[arg-type]


def test_admitting_request_in_impossible_state_rejected():
    # [ROBUST]
    # Fails if add_request accepts a request whose state was bypassed via
    # `_state` mutation (defence-in-depth behind the read-only property).
    request = SequenceGroup("request", [1], max_new_tokens=2)
    request._state = State.DECODING  # bypass the property deliberately
    scheduler = IterationScheduler(max_batch_size=2, max_tokens_per_batch=10)
    with pytest.raises(ValueError):
        scheduler.add_request(request)


# =========================================================================
# Hypothesis-style property tests (comment blocks; dependency optional)
# =========================================================================
#
# PROPERTY 1 (prefill coverage):
#   For any prompt P = list[int] with len(P) >= 1, any prefill_chunk_size
#   c >= 1, any max_tokens_per_batch tb >= 1, any max_batch_size mb >= 1,
#   and enough schedule_iteration calls to reach DECODING:
#       concat([chunk.tokens for chunk in emitted_chunks_for(request)]) == P
#   and each chunk satisfies 1 <= chunk.length <= c and
#   chunk.tokens == P[chunk.start:chunk.end].
#
#   @given(
#       prompt=st.lists(
#           st.integers(min_value=0, max_value=10**6),
#           min_size=1, max_size=200,
#       ),
#       chunk_size=st.integers(min_value=1, max_value=50),
#       token_budget=st.integers(min_value=1, max_value=500),
#       batch_size=st.integers(min_value=1, max_value=50),
#   )
#
# PROPERTY 2 (budget safety):
#   For any sequence of add_request / step / schedule_iteration operations,
#   every returned IterationBatch satisfies:
#       batch.num_items   <= max_batch_size
#       batch.token_count <= max_tokens_per_batch
#
#   @given(
#       ops=st.lists(
#           st.one_of(
#               st.just(("add", st.integers(0, 5))),
#               st.just(("step", st.integers(0, 5))),
#               st.just("schedule"),
#           ),
#           min_size=1, max_size=100,
#       ),
#       mb=st.integers(1, 8),
#       tb=st.integers(1, 64),
#       cs=st.integers(1, 16),
#   )
#
# PROPERTY 3 (finished requests never reappear):
#   For any sequence of operations, once `request.is_finished()` returns
#   True, that request never appears in a later batch's `decode_requests`
#   or `prefill_chunks[*].request`.
#
# PROPERTY 4 (lifecycle monotonicity):
#   For any request, the sequence of states observed over its lifetime is a
#   prefix of
#       WAITING -> PREFILLING -> DECODING -> FINISHED
#   (with PREFILLING skipped for empty prompts, and DECODING skipped for
#   max_new_tokens == 0).


# =========================================================================
# Not tested (and why):
# - True GPU / async execution: this is a CPU scheduler simulator.
# - EOS with a magic/silent default token id: the spec forbids assuming
#   one. We test only the explicit per-request `eos_token` path.
# - Starvation freedom / fairness: not proven. FIFO admission order is
#   deterministic, but pathological prefill chunk sizes and tight decode
#   budgets can delay some requests. Out of scope per spec.
# - KV-cache allocation, PagedAttention block tables, memory pressure:
#   covered by Days 54-55; this scheduler models requests, not memory
#   blocks.
# - Wall-clock throughput and GPU utilisation: no benchmarks are run. The
#   throughput formula in the brief is a metric definition, not a
#   measurement.
# - Multi-tenant isolation, cancellation, deadlines, retries, priorities:
#   out of scope per spec.
# - Concurrency / thread safety: the scheduler is single-threaded.
# - Cross-process determinism: the repeated-trace test runs within one
#   process. Hash randomisation across processes could in principle affect
#   a broken implementation, but this code path uses no sets for ordering;
#   a subprocess test is deferred to future work.
# - Executable Hypothesis property tests: the four properties above are
#   documented as comment blocks because `hypothesis` is an optional
#   dependency. Implementing them as executable tests is tracked as future
#   work.
# - The prefill loop assumes `_remove_finished_requests` and
#   `_move_empty_prompts_to_decoding` have already removed FINISHED and
#   empty-prompt requests from `_waiting_queue`. A violation is not tested
#   (it would require fabricating an impossible state); the invariant is
#   maintained by construction.
# - Stepping a DECODING request *before* `schedule_iteration()` returns it
#   bypasses token-budget accounting. The API does not prevent this; it is
#   documented as caller misuse.
# =========================================================================
