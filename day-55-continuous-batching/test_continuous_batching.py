"""
test_continuous_batching.py — Mutation-sensitive tests for Day 55.

Tags: [ROBUST] catches a specific mutated implementation
      [CONTRACT] catches a changed public interface
      [GUARD] catches a missing defensive validation (subset of ROBUST)

Adversarial-review revision adds:
    test_configuration_is_read_only         — catches writable capacity limits
    test_mixed_decode_prefill_respects_budget — catches ignoring existing
        decode cost when admitting a new prefill (an implementation could
        pass prompt-only budget tests while mis-accounting decode)
    test_waiting_and_active_disjoint        — catches a slot added to
        _active without being removed from _waiting_queue
    test_negative_max_new_tokens_rejected   — catches accepting < 0
And strengthens:
    test_seen_ids_preserved_against_req_id_bypass — now asserts the
        original id 2 remains reserved after a bypass of r2's mirror to 1
    test_deterministic_ordering_and_tokens  — asserts exact expected order,
        not just equality between two runs

Command:
    cd day-55-continuous-batching && pytest -v test_continuous_batching.py
"""

import pytest
from day55_continuous_batching import (
    ContinuousBatchingEngine,
    RequestState,
    SequenceRequest,
    _placeholder_token,
)


def _req(req_id, prompt, max_new):
    return SequenceRequest(
        req_id=req_id, prompt_tokens=list(prompt), max_new_tokens=max_new
    )


# --------------------------------------------------------------------------- #
# Lifecycle                                                                   #
# --------------------------------------------------------------------------- #

# [ROBUST] catches add_request eagerly admitting.
def test_add_request_enqueues_without_activating():
    engine = ContinuousBatchingEngine(max_batch_size=4, max_tokens_per_batch=16)
    r = _req(1, [1, 2], 3)
    engine.add_request(r)
    assert len(engine.waiting_queue) == 1
    assert len(engine.active_requests) == 0
    assert r.state == RequestState.WAITING
    assert r.generated_tokens == ()


# [ROBUST] catches LIFO admission.
def test_admission_is_fifo():
    engine = ContinuousBatchingEngine(max_batch_size=3, max_tokens_per_batch=16)
    for i in (1, 2, 3):
        engine.add_request(_req(i, [0], 1))
    engine.step()
    assert [r.req_id for r in engine.active_requests] == [1, 2, 3]
    assert engine.last_admitted == (1, 2, 3)


# [ROBUST] catches missing max_batch_size check.
def test_max_batch_size_enforced():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=64)
    for i in range(5):
        engine.add_request(_req(i, [0], 2))
    engine.step()
    assert engine.last_admitted == (0, 1)
    assert len(engine.waiting_queue) == 3


# [ROBUST] catches missing token-budget check.
def test_token_budget_enforced_on_admission():
    engine = ContinuousBatchingEngine(max_batch_size=4, max_tokens_per_batch=6)
    for i in range(3):
        engine.add_request(_req(i, [0, 0, 0], 1))
    engine.step()
    assert engine.last_admitted == (0, 1)
    assert len(engine.waiting_queue) == 1


# [ROBUST] catches admission logic that treats each active request as cost 1
# regardless of whether it is prefilling or decoding. On step 2 both active
# slots cost 1 each (decode), so a 2-token prefill cannot fit under budget 3.
# A mutation that ignores decode cost would report cost=0 and admit request 3.
def test_mixed_decode_prefill_respects_budget():
    engine = ContinuousBatchingEngine(max_batch_size=3, max_tokens_per_batch=3)
    engine.add_request(_req(1, [0], 3))        # prefill cost 1
    engine.add_request(_req(2, [0], 3))        # prefill cost 1
    engine.add_request(_req(3, [0, 0], 1))     # prefill cost 2
    engine.step()
    assert engine.last_admitted == (1, 2)
    assert [r.req_id for r in engine.waiting_queue] == [3]
    # Step 2: 1 and 2 are decoding (cost 2). Request 3 prefill (2) + decode 2
    # = 4 > 3, so admission must be blocked.
    engine.step()
    assert engine.last_admitted == ()
    assert [r.req_id for r in engine.waiting_queue] == [3]


# --------------------------------------------------------------------------- #
# Generation and completion                                                   #
# --------------------------------------------------------------------------- #

# [ROBUST] catches off-by-one in max_new_tokens and position drift.
def test_max_new_tokens_two_emits_exactly_two():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=16)
    r = _req(7, [1, 2], 2)
    engine.add_request(r)
    engine.step()
    engine.step()
    assert r.generated_tokens == (
        _placeholder_token(7, 0),
        _placeholder_token(7, 1),
    )
    assert r.is_finished()


# [ROBUST] catches FINISHED requests still emitting.
def test_finished_not_emitted_after_eviction():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    r = _req(1, [0], 1)
    engine.add_request(r)
    engine.step()
    emissions_after = engine.step()
    assert r.req_id not in emissions_after
    assert len(r.generated_tokens) == 1
    assert r.is_finished()
    assert r.req_id not in [x.req_id for x in engine.active_requests]


# [ROBUST] catches static batching.
def test_heterogeneous_lengths_no_idling():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    short = _req(1, [0], 1)
    long_ = _req(2, [0], 3)
    engine.add_request(short)
    engine.add_request(long_)
    e1 = engine.step()
    assert set(e1) == {1, 2}
    e2 = engine.step()
    assert set(e2) == {2}
    assert short.is_finished() and not long_.is_finished()
    e3 = engine.step()
    assert set(e3) == {2}
    assert long_.is_finished()


# [ROBUST] catches double execution per step.
def test_one_execution_per_request_per_step():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    r = _req(1, [0], 10)
    engine.add_request(r)
    engine.step()
    assert len(r.generated_tokens) == 1
    engine.step()
    assert len(r.generated_tokens) == 2


# [ROBUST] catches an evictor that removes only the first finished slot per
# step.
def test_multiple_requests_finish_same_step():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=8)
    r1 = _req(1, [0], 1)
    r2 = _req(2, [0], 1)
    engine.add_request(r1)
    engine.add_request(r2)
    e1 = engine.step()
    assert set(e1) == {1, 2}
    assert r1.is_finished() and r2.is_finished()
    e2 = engine.step()
    assert e2 == {}
    assert engine.active_requests == []
    assert engine.last_evicted == (1, 2)


# [ROBUST] catches a slot added to _active without being removed from
# _waiting_queue, or vice versa — the invariant that no request is in both
# collections simultaneously.
def test_waiting_and_active_disjoint():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=4)
    engine.add_request(_req(1, [0], 3))
    engine.add_request(_req(2, [0], 3))
    engine.add_request(_req(3, [0], 3))
    for step in range(6):
        engine.step()
        waiting_ids = {r.req_id for r in engine.waiting_queue}
        active_ids = {r.req_id for r in engine.active_requests}
        assert waiting_ids.isdisjoint(active_ids), (
            f"step {step}: overlap {waiting_ids & active_ids}"
        )


# --------------------------------------------------------------------------- #
# Continuous re-admission                                                     #
# --------------------------------------------------------------------------- #

# [ROBUST] catches missing re-admission after eviction.
def test_waiting_admitted_after_completion():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    engine.add_request(_req(2, [0], 1))
    engine.step()
    assert [x.req_id for x in engine.active_requests] == [1]
    assert [x.req_id for x in engine.waiting_queue] == [2]
    e2 = engine.step()
    assert 2 in e2
    assert [x.req_id for x in engine.active_requests] == [2]


# [ROBUST] catches post-cohort stall; the absence assertion makes the
# no-re-emission contract explicit.
def test_engine_continues_after_completion():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    engine.add_request(_req(2, [0], 1))
    engine.step()
    e2 = engine.step()
    assert 2 in e2
    assert 1 not in e2
    assert [x.req_id for x in engine.active_requests] == [2]
    engine.step()
    assert not engine.active_requests
    assert not engine.waiting_queue


# --------------------------------------------------------------------------- #
# Admission policy                                                            #
# --------------------------------------------------------------------------- #

# [ROBUST] catches skip-ahead admission.
def test_strict_fifo_blocks_head_of_line():
    engine = ContinuousBatchingEngine(max_batch_size=4, max_tokens_per_batch=4)
    engine.add_request(_req(1, [0, 0, 0, 0], 1))
    engine.add_request(_req(2, [0], 1))
    engine.step()
    assert [x.req_id for x in engine.active_requests] == [1]
    assert [x.req_id for x in engine.waiting_queue] == [2]


# --------------------------------------------------------------------------- #
# Request validation                                                          #
# --------------------------------------------------------------------------- #

# [ROBUST] catches duplicate acceptance.
def test_duplicate_req_id_rejected():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_request(_req(1, [0], 1))


# [ROBUST] catches id recycling after eviction.
def test_duplicate_req_id_rejected_even_after_completion():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    engine.step()
    engine.step()
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_request(_req(1, [0], 1))


# [ROBUST] catches accepting max_new_tokens=0.
def test_zero_max_new_tokens_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        _req(1, [0], 0)


# [ROBUST] catches accepting a negative max_new_tokens (which would sneak
# past a `> 0` check written as `>= 1` — distinct mutation from zero).
def test_negative_max_new_tokens_rejected():
    with pytest.raises(ValueError, match="max_new_tokens"):
        _req(1, [0], -1)


# [ROBUST] catches accepting empty prompts.
def test_empty_prompt_rejected():
    with pytest.raises(ValueError, match="prompt_tokens"):
        _req(1, [], 2)


# [ROBUST] catches accepting prompts larger than the budget.
def test_oversized_prompt_rejected():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=4)
    with pytest.raises(ValueError, match="max_tokens_per_batch"):
        engine.add_request(_req(1, [0, 0, 0, 0, 0], 1))


# [CONTRACT] catches nonsensical engine configuration.
def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ContinuousBatchingEngine(max_batch_size=0, max_tokens_per_batch=8)
    with pytest.raises(ValueError):
        ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=0)
    with pytest.raises(TypeError):
        ContinuousBatchingEngine(max_batch_size="2", max_tokens_per_batch=8)  # type: ignore[arg-type]


# [ROBUST] catches accepting non-int prompt tokens.
def test_malformed_prompt_rejected():
    with pytest.raises(TypeError):
        _req(1, [1, "two", 3], 2)


# [ROBUST] catches a failed add_request that partially mutates state.
def test_failed_add_does_not_corrupt_state():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    with pytest.raises(ValueError):
        engine.add_request(_req(1, [0], 1))
    assert len(engine.waiting_queue) == 1
    engine.add_request(_req(2, [0], 1))
    assert len(engine.waiting_queue) == 2


# [ROBUST] catches a SequenceRequest sharing the caller's prompt list.
def test_prompt_tokens_frozen_at_construction():
    src = [1, 2, 3]
    r = SequenceRequest(req_id=1, prompt_tokens=src, max_new_tokens=1)
    src.append(999)
    assert r.prompt_tokens == (1, 2, 3)


# [CONTRACT] catches adding a generated_tokens constructor parameter back in.
def test_no_generated_tokens_constructor_param():
    with pytest.raises(TypeError):
        SequenceRequest(  # type: ignore[call-arg]
            req_id=1, prompt_tokens=[0], max_new_tokens=2,
            generated_tokens=[],
        )


# [ROBUST] catches add_request skipping re-validation.
def test_add_request_revalidates_under_private_bypass():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    r = _req(1, [0], 2)
    object.__setattr__(r, "_max_new_tokens", 0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        engine.add_request(r)
    assert not engine.waiting_queue
    assert not engine.active_requests


# --------------------------------------------------------------------------- #
# Engine configuration is read-only                                           #
# --------------------------------------------------------------------------- #

# [ROBUST] catches writable capacity limits, which would allow post-
# construction mutation to violate the active-count and per-step-work bounds
# while requests are in flight. The read-only property makes such mutation
# raise AttributeError, closing the public-API route.
def test_configuration_is_read_only():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=8)
    assert engine.max_batch_size == 2
    assert engine.max_tokens_per_batch == 8
    with pytest.raises(AttributeError):
        engine.max_batch_size = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        engine.max_tokens_per_batch = 4  # type: ignore[misc]
    # Values unchanged after the attempts.
    assert engine.max_batch_size == 2
    assert engine.max_tokens_per_batch == 8


# [ROBUST] confirms that the capacity invariants hold AFTER a mutation
# attempt fails; the engine remains usable and consistent.
def test_config_mutation_does_not_perturb_engine():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=8)
    engine.add_request(_req(1, [0], 2))
    engine.add_request(_req(2, [0], 2))
    try:
        engine.max_batch_size = 1  # type: ignore[misc]
    except AttributeError:
        pass
    # Engine still admits both (max_batch_size was 2, not 1).
    engine.step()
    assert engine.last_admitted == (1, 2)


# --------------------------------------------------------------------------- #
# Read-only public request state                                              #
# --------------------------------------------------------------------------- #

# [ROBUST] catches public lifecycle fields being assignable.
@pytest.mark.parametrize("field,value", [
    ("req_id", 99),
    ("prompt_tokens", (1, 2)),
    ("max_new_tokens", 0),
    ("generated_tokens", [99]),
    ("state", RequestState.FINISHED),
    ("prefill_done", True),
])
def test_public_lifecycle_field_is_read_only(field, value):
    r = _req(1, [0], 2)
    with pytest.raises(AttributeError):
        setattr(r, field, value)


# [ROBUST] catches generated_tokens exposing a mutable list.
def test_generated_tokens_view_is_immutable():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0], 2)
    engine.add_request(r)
    engine.step()
    assert isinstance(r.generated_tokens, tuple)
    with pytest.raises(AttributeError):
        r.generated_tokens.append(99)  # type: ignore[attr-defined]


# [ROBUST] catches mutable tracing lists.
def test_tracing_fields_are_read_only_tuples():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    assert isinstance(engine.last_admitted, tuple)
    assert isinstance(engine.last_evicted, tuple)
    with pytest.raises(AttributeError):
        engine.last_admitted.append(99)  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        engine.last_evicted.append(99)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Private-bypass resistance for request mirrors                               #
# --------------------------------------------------------------------------- #

# [ROBUST] the engine must emit under the add-time snapshot req_id.
def test_engine_uses_snapshot_req_id_not_public_mirror():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0], 3)
    engine.add_request(r)
    object.__setattr__(r, "_req_id", 77)
    emissions = engine.step()
    assert emissions == {1: _placeholder_token(1, 0)}
    assert r.req_id == 1


# [ROBUST] a bypassed state mirror must be repaired at end of step.
def test_state_mirror_resynced_after_bypass():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0], 3)
    engine.add_request(r)
    engine.step()
    assert r.state == RequestState.RUNNING
    object.__setattr__(r, "_state", RequestState.FINISHED)
    assert r.state == RequestState.FINISHED
    emissions = engine.step()
    assert 1 in emissions
    assert r.state == RequestState.RUNNING
    assert len(r.generated_tokens) == 2


# [ROBUST] a bypass between enqueue and admission must not change prefill cost.
def test_bypass_after_enqueue_before_admission_does_not_affect_admission():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0, 0, 0], 2)
    engine.add_request(r)
    object.__setattr__(r, "_prompt_tokens", (0,) * 20)
    engine.step()
    assert engine.last_admitted == (1,)
    assert r.prompt_tokens == (0, 0, 0)


# [ROBUST] a max_new_tokens bypass before admission must not affect completion.
def test_bypass_max_new_tokens_before_admission_does_not_affect_completion():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0], 3)
    engine.add_request(r)
    object.__setattr__(r, "_max_new_tokens", 1)
    for _ in range(3):
        engine.step()
    assert len(r.generated_tokens) == 3
    assert r.is_finished()


# [ROBUST] _seen_ids uses add-time snapshots, so a req_id bypass on r2's
# mirror cannot remove r2's ORIGINAL id 2 from the uniqueness set. The prior
# version of this test only asserted that id 1 remained duplicate — that
# was already true from the first add and did not exercise the bypass.
def test_seen_ids_preserved_against_req_id_bypass():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=8)
    engine.add_request(_req(1, [0], 1))
    r2 = _req(2, [0], 1)
    engine.add_request(r2)
    object.__setattr__(r2, "_req_id", 1)
    # id 1 was reserved at first add_request — correctly rejected.
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_request(_req(1, [0], 1))
    # id 2 was ALSO reserved at r2's add_request; a bypass of the mirror
    # must not have dropped it. Without _Slot's add-time snapshot, an
    # implementation that derived _seen_ids from the live mirror would
    # wrongly permit this add.
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_request(_req(2, [0], 1))


# [ROBUST] corrupt every lifecycle field at once; engine behavior unaffected
# and mirror fully repaired.
def test_engine_authoritative_under_private_bypass():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    r = _req(1, [0], 2)
    engine.add_request(r)
    engine.step()

    object.__setattr__(r, "_req_id", 77)
    object.__setattr__(r, "_generated_tokens", [99, 99, 99])
    object.__setattr__(r, "_max_new_tokens", 0)
    object.__setattr__(r, "_state", RequestState.FINISHED)
    object.__setattr__(r, "_prompt_tokens", ())

    emissions = engine.step()
    assert emissions == {1: _placeholder_token(1, 1)}
    assert r.req_id == 1
    assert r.max_new_tokens == 2
    assert r.prompt_tokens == (0,)
    assert r.generated_tokens == (
        _placeholder_token(1, 0),
        _placeholder_token(1, 1),
    )
    assert r.is_finished()


# --------------------------------------------------------------------------- #
# Determinism and tracing                                                     #
# --------------------------------------------------------------------------- #

# [ROBUST] asserts EXACT FIFO order and token values across two steps. A
# buggy implementation with stable-but-wrong iteration order would pass a
# naive "two runs equal" check but fail the explicit expected-output
# assertion below.
def test_deterministic_ordering_and_tokens():
    engine = ContinuousBatchingEngine(max_batch_size=3, max_tokens_per_batch=16)
    for i in (10, 20, 30):
        engine.add_request(_req(i, [0], 2))
    e1 = engine.step()
    e2 = engine.step()
    assert list(e1.items()) == [
        (10, _placeholder_token(10, 0)),
        (20, _placeholder_token(20, 0)),
        (30, _placeholder_token(30, 0)),
    ]
    assert list(e2.items()) == [
        (10, _placeholder_token(10, 1)),
        (20, _placeholder_token(20, 1)),
        (30, _placeholder_token(30, 1)),
    ]


# [ROBUST] catches ghost mutations on empty step.
def test_step_noop_when_empty():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    assert engine.step() == {}
    assert engine.last_admitted == ()
    assert engine.last_evicted == ()
    assert not engine.waiting_queue
    assert not engine.active_requests


# [ROBUST] catches an untraceable step; pins Design-A eviction timing.
def test_step_tracing():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    engine.add_request(_req(1, [0], 1))
    engine.step()
    assert engine.last_admitted == (1,)
    assert engine.last_evicted == ()
    engine.step()
    assert engine.last_admitted == ()
    assert engine.last_evicted == (1,)


# [ROBUST] catches token drift between emitted and stored.
def test_emissions_match_request_state():
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=16)
    r = _req(42, [7, 8], 3)
    engine.add_request(r)
    for _ in range(3):
        engine.step()
    assert r.generated_tokens == (
        _placeholder_token(42, 0),
        _placeholder_token(42, 1),
        _placeholder_token(42, 2),
    )


# --------------------------------------------------------------------------- #
# API contract                                                                #
# --------------------------------------------------------------------------- #

# [CONTRACT] catches a step() that returns None or a different container.
def test_step_return_type_is_emissions_dict():
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    engine.add_request(_req(1, [0], 1))
    result = engine.step()
    assert isinstance(result, dict)
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in result.items())


# [CONTRACT] catches is_finished disagreeing with state.
def test_is_finished_matches_state():
    r = _req(1, [0], 1)
    assert not r.is_finished()
    engine = ContinuousBatchingEngine(max_batch_size=1, max_tokens_per_batch=8)
    engine.add_request(r)
    assert not r.is_finished()
    engine.step()
    assert r.is_finished()
    assert r.state == RequestState.FINISHED


# --------------------------------------------------------------------------- #
# Property test                                                               #
# --------------------------------------------------------------------------- #

# PROPERTY: for any list of unique (req_id, prompt_len, max_new) triples with
# 1 <= prompt_len <= max_tokens_per_batch and 1 <= max_new <= 20, running
# step() until drained must emit exactly max_new tokens per request, never
# exceed max_batch_size active, and never exceed max_new_tokens generated.

try:
    from hypothesis import given, settings  # type: ignore
    from hypothesis import strategies as st  # type: ignore
    _HAVE_HYPOTHESIS = True
except ImportError:
    _HAVE_HYPOTHESIS = False


@pytest.mark.skipif(not _HAVE_HYPOTHESIS, reason="hypothesis not installed")
def test_property_exact_generation_and_bounds():
    @settings(max_examples=50, deadline=None)
    @given(
        reqs=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1000),
                st.integers(min_value=1, max_value=8),
                st.integers(min_value=1, max_value=20),
            ),
            min_size=1,
            max_size=8,
            unique_by=lambda t: t[0],
        ),
        batch_size=st.integers(min_value=1, max_value=4),
        token_budget=st.integers(min_value=8, max_value=32),
    )
    def inner(reqs, batch_size, token_budget):
        engine = ContinuousBatchingEngine(
            max_batch_size=batch_size, max_tokens_per_batch=token_budget,
        )
        objs = {}
        for (rid, plen, mx) in reqs:
            if plen > token_budget:
                continue
            r = SequenceRequest(rid, [0] * plen, mx)
            objs[rid] = r
            engine.add_request(r)

        for _ in range(1000):
            if not engine.active_requests and not engine.waiting_queue:
                break
            engine.step()
            assert len(engine.active_requests) <= batch_size
            for r in objs.values():
                assert len(r.generated_tokens) <= r.max_new_tokens

        for r in objs.values():
            assert len(r.generated_tokens) == r.max_new_tokens
            assert r.is_finished()

    inner()


# --------------------------------------------------------------------------- #
# Not tested (and why):                                                       #
# --------------------------------------------------------------------------- #
# - Real transformer forward pass / model-quality tokens: scheduler sim only.
# - GPU utilization and speedup: unmeasurable in pure Python; not claimed.
# - KV-cache block allocation / release: not implemented.
# - Async serving, cancellation, timeouts, multi-worker: synchronous by design.
# - Throughput metric (spec item 18): not implemented.
# - Preemption / active-to-waiting swap: not implemented.
# - Non-integer req_id: rejected by contract.
# - Chunked prefill: rejected (prompt > budget refused at add_request).
# - _seen_ids lifetime growth: documented contract, not a bug.
# - Hypothesis property test if not installed: skipped with clear reason.
# - RuntimeError in _execute: unreachable via public API or private-field
#   bypass of SequenceRequest. Reaching it would require directly mutating
#   engine._active between _evict_finished and _execute (monkey-patching or
#   direct slot manipulation). Not covered because it is outside the
#   documented contract.
# - Engine configuration bypass via object.__setattr__(engine,
#   "_max_batch_size", ...): documented in the module docstring as outside
#   the threat model. Public-API assignment is covered by
#   test_configuration_is_read_only. A hostile caller reaching into private
#   fields is not defended against because doing so would require snapshotting
#   config at every step — an over-engineering tax for an educational sim.
# - Waiting-request mirror bypassed between add and first step: documented
#   in the module docstring as the only window where a corrupted mirror can
#   disagree with actual state. Not tested because testing it would assert
#   on a hostile-caller observation that is not a scheduling decision.
# - Transactional rollback after step() failure: not implemented (fail-fast
#   by design; step() has no reachable failure modes from user input).