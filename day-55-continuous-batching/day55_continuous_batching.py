"""
day55_continuous_batching.py — Continuous Batching & Iteration-Level Scheduling

What this solves:
    A deterministic simulation of an Orca-style iteration-level continuous
    batching scheduler. Requests are admitted from a FIFO waiting queue into
    an active batch, execute exactly one generation iteration per step(), and
    are evicted once they reach max_new_tokens — freeing capacity for waiting
    requests. Static batching (holding a batch until the longest sequence
    finishes) is the anti-pattern this avoids.

Complexity:
    Let A = active set size, k = requests admitted in one step, W = waiting
    queue length, P = a request's prompt length, and G = total generated-token
    history length across all slots (W + A).

    add_request: O(P) — validates each prompt token; one deque append is O(1).

    step():      O(A + k + G_A) where G_A = total generated-history length
                 across ACTIVE requests, copied during the mirror-sync pass.
                 The naive "O(A + k)" bound omits that copy and understates
                 work when max_new_tokens grows. Each of _evict_finished,
                 _current_iteration_cost, _execute, the mark loop, and the
                 sync loop scans A once; _admit pops at most k; the sync loop
                 copies G_A tokens. Worst case O(A + W + G) when all active
                 drain and all waiting admit in one step.

    Space:       O(W + A + |seen_ids| + G). Prompt tuples are shared by
                 reference between a request and its slot. Generated
                 histories are NOT shared: each slot holds its own list and
                 each request mirror holds a second copy produced by
                 _engine_sync.

Configuration invariants:
    max_batch_size and max_tokens_per_batch are set at construction and
    exposed as read-only properties. The active-count and per-step-work
    bounds documented above hold unconditionally under this contract.
    Public-API mutation raises AttributeError; a deliberate
    object.__setattr__ bypass of the private fields is a hostile-caller
    concern outside this simulation's threat model and is not defended
    against here. This is a deliberate asymmetry with the request-level
    mirror, which IS bypass-resistant: the request mirror was reproduced
    as a scheduling bug in review; engine config was not.

What this is NOT:
    - Not a language model. Tokens come from a pure placeholder function.
    - Not a GPU kernel. No CUDA, no kernel launches, no utilization numbers.
    - Not PagedAttention. max_batch_size and max_tokens_per_batch are
      integer scheduling-work counters, not hard token- or memory-bounds.
    - Not asynchronous. step() is synchronous and single-threaded.
    - Not fail-recovering. step() is fail-fast: if _execute encounters an
      invariant violation it raises RuntimeError; there is no rollback.
      Under the current design the guard is unreachable via the public API
      and via private-field bypass of SequenceRequest — it would require
      directly mutating engine._active between _evict_finished and
      _execute, which is outside the documented contract.
    - No throughput speedup is reported. The 2x-4x figures quoted in
      industry are workload-, hardware-, and baseline-dependent; this
      module makes no such claim.

Prefill / decode accounting:
    Prefill is the first iteration after admission. Its cost is
    len(prompt_tokens) and it emits the FIRST generated token. Every later
    iteration costs 1 and emits one token. max_new_tokens counts total
    emitted tokens. Prefill is atomic: prompts longer than
    max_tokens_per_batch are rejected at add_request rather than split.

Ownership contract (request-level):
    Each SequenceRequest is snapshotted into an engine-owned _Slot at
    add_request() time. Every scheduling decision — admission cost, batch
    limit, execution, completion, eviction — reads from the _Slot, never
    from the request object. The request's public state is a MIRROR that
    is re-synchronized from the _Slot at every exit point:
        - end of every step() in which the slot is active;
        - immediately before eviction from the active set.
    Consequence: a caller using object.__setattr__ to corrupt any private
    field of SequenceRequest (including _req_id) cannot change what the
    engine does. Their corruption is repaired at the next sync point —
    defined as the end of every step in which the slot is active, and
    immediately before eviction. A WAITING slot that is never admitted is
    never re-synced; the mirror there is trusted as-written until admission.
    This is the design's only window where a bypassed mirror can disagree
    with actual scheduling state, and it is confined to waiting requests,
    which the engine has not yet begun to execute. Lifetime uniqueness
    (_seen_ids) also uses the add-time snapshot, so a private bypass of
    _req_id cannot corrupt the uniqueness contract.

Reproducibility:
    Deterministic; no seed handling required. _placeholder_token is a pure
    function of (req_id, generated_position). No RNG anywhere.

Public state:
    engine.waiting_queue and engine.active_requests return copies.
    engine.last_admitted and engine.last_evicted are tuple properties.
    engine.max_batch_size and engine.max_tokens_per_batch are read-only
    properties.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

_VOCAB_SIZE = 32_000


def _placeholder_token(req_id: int, position: int) -> int:
    """Deterministic placeholder token id. Not model output."""
    return (req_id * 100 + position) % _VOCAB_SIZE


class RequestState(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class SequenceRequest:
    """
    One generation request. Public state is read-only.

    Lifecycle fields are mutated by the engine via _engine_sync(), which
    replaces all mutable fields at once from the engine's authoritative
    _Slot. Direct assignment to any property raises AttributeError.
    Private-field bypasses (object.__setattr__) are repaired by _engine_sync
    at the next exit point. See the module docstring for the full contract.
    """

    __slots__ = (
        "_generated_tokens",
        "_max_new_tokens",
        "_prefill_done",
        "_prompt_tokens",
        "_req_id",
        "_state",
    )

    def __init__(
        self,
        req_id: int,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
    ) -> None:
        self._req_id = req_id
        self._prompt_tokens: tuple[int, ...] = tuple(prompt_tokens)
        self._max_new_tokens = max_new_tokens
        self._generated_tokens: list[int] = []
        self._state: RequestState = RequestState.WAITING
        self._prefill_done: bool = False
        _validate_request(self)

    @property
    def req_id(self) -> int:
        return self._req_id

    @property
    def prompt_tokens(self) -> tuple[int, ...]:
        return self._prompt_tokens

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @property
    def generated_tokens(self) -> tuple[int, ...]:
        return tuple(self._generated_tokens)

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def prefill_done(self) -> bool:
        return self._prefill_done

    def is_finished(self) -> bool:
        return self._state == RequestState.FINISHED

    def _engine_sync(
        self,
        *,
        req_id: int,
        prompt_tokens: tuple[int, ...],
        max_new_tokens: int,
        state: RequestState,
        prefill_done: bool,
        generated_tokens: tuple[int, ...],
    ) -> None:
        self._req_id = req_id
        self._prompt_tokens = prompt_tokens
        self._max_new_tokens = max_new_tokens
        self._state = state
        self._prefill_done = prefill_done
        self._generated_tokens = list(generated_tokens)


def _validate_request(req: SequenceRequest) -> None:
    """Structural validation. Called from __init__ and again from add_request.

    The add_request call catches a caller who bypassed __init__ validation
    via object.__setattr__ on a private field before enqueue. After
    add_request returns, subsequent bypasses cannot reach validation, but
    they also cannot reach scheduling, because the engine snapshots at add
    time.
    """
    if not isinstance(req._req_id, int) or isinstance(req._req_id, bool):
        raise TypeError(f"req_id must be int, got {type(req._req_id).__name__}")
    if req._req_id < 0:
        raise ValueError(f"req_id must be non-negative, got {req._req_id}")

    if not isinstance(req._prompt_tokens, tuple):
        raise TypeError(
            f"prompt_tokens must be a tuple at this point, got "
            f"{type(req._prompt_tokens).__name__}"
        )
    if len(req._prompt_tokens) == 0:
        raise ValueError(
            "prompt_tokens must be non-empty; empty prompts are rejected "
            "explicitly (see TEST_STRATEGY.md)"
        )
    for i, t in enumerate(req._prompt_tokens):
        if not isinstance(t, int) or isinstance(t, bool):
            raise TypeError(f"prompt_tokens[{i}] must be int, got {type(t).__name__}")
        if t < 0 or t >= _VOCAB_SIZE:
            raise ValueError(f"prompt_tokens[{i}]={t} out of range [0, {_VOCAB_SIZE})")

    if not isinstance(req._max_new_tokens, int) or isinstance(
        req._max_new_tokens, bool
    ):
        raise TypeError(
            f"max_new_tokens must be int, got {type(req._max_new_tokens).__name__}"
        )
    if req._max_new_tokens < 1:
        raise ValueError(
            f"max_new_tokens must be >= 1, got {req._max_new_tokens}; "
            f"max_new_tokens=0 is rejected (see TEST_STRATEGY.md)"
        )


@dataclass
class _Slot:
    """Engine-authoritative per-request state, snapshotted at add_request.

    All fields are engine-owned; the request object is a mirror that gets
    re-synced at exit points.
    """

    request: SequenceRequest
    req_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    generated_tokens: list[int] = field(default_factory=list)
    prefill_done: bool = False
    finished: bool = False

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)


class ContinuousBatchingEngine:
    """
    Iteration-level continuous batching scheduler.

    step() order (do not reorder — invariant guard in _execute):
        1. Evict: remove FINISHED slots from the active list, syncing each
                  slot's mirror one final time before removal.
        2. Admit: pop from waiting_queue in strict FIFO while BOTH
                  len(active) < max_batch_size AND current_cost + prefill <=
                  max_tokens_per_batch. If the head cannot fit, STOP.
        3. Execute: one iteration per RUNNING slot, in admission order.
        4. Mark: slots with len(generated_tokens) >= max_new_tokens become
                 FINISHED. They stay in the active list until step 1 of the
                 next call (Design A).
        5. Sync: re-sync every active slot's request mirror from the slot.
                 This copies G_A token ids across active slots; the copy is
                 O(G_A), not O(A), and is included in the module's step()
                 complexity bound.
        6. Return: {req_id: emitted_token} for every request that emitted.

    Every step() ends with the invariant: for every slot the caller can see
    (via active_requests, or via a retained request reference), the request
    mirror matches the slot exactly.

    Configuration: max_batch_size and max_tokens_per_batch are read-only
    after construction. See the module docstring for why.
    """

    def __init__(self, max_batch_size: int, max_tokens_per_batch: int) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool):
            raise TypeError("max_batch_size must be int")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if not isinstance(max_tokens_per_batch, int) or isinstance(
            max_tokens_per_batch, bool
        ):
            raise TypeError("max_tokens_per_batch must be int")
        if max_tokens_per_batch < 1:
            raise ValueError("max_tokens_per_batch must be >= 1")

        self._max_batch_size: int = max_batch_size
        self._max_tokens_per_batch: int = max_tokens_per_batch

        self._waiting_queue: deque[_Slot] = deque()
        self._active: list[_Slot] = []
        self._seen_ids: set[int] = set()

        self._last_admitted: list[int] = []
        self._last_evicted: list[int] = []

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_tokens_per_batch(self) -> int:
        return self._max_tokens_per_batch

    @property
    def waiting_queue(self) -> deque[SequenceRequest]:
        return deque(slot.request for slot in self._waiting_queue)

    @property
    def active_requests(self) -> list[SequenceRequest]:
        return [slot.request for slot in self._active]

    @property
    def last_admitted(self) -> tuple[int, ...]:
        return tuple(self._last_admitted)

    @property
    def last_evicted(self) -> tuple[int, ...]:
        return tuple(self._last_evicted)

    def add_request(self, request: SequenceRequest) -> None:
        """
        Enqueue without admitting. Snapshots the request into an engine-owned
        _Slot at this point, so any post-enqueue private-field bypass cannot
        reach scheduling. Rejects before any state mutation.
        """
        if not isinstance(request, SequenceRequest):
            raise TypeError(
                f"add_request expects SequenceRequest, got {type(request).__name__}"
            )
        _validate_request(request)
        req_id = request.req_id
        prompt_tokens = request.prompt_tokens
        max_new_tokens = request.max_new_tokens
        if req_id in self._seen_ids:
            raise ValueError(
                f"duplicate req_id {req_id}; ids are unique for the "
                f"lifetime of this engine"
            )
        if len(prompt_tokens) > self._max_tokens_per_batch:
            raise ValueError(
                f"prompt for req_id={req_id} has {len(prompt_tokens)} tokens, "
                f"exceeding max_tokens_per_batch={self._max_tokens_per_batch}. "
                f"Prefill cannot be split; this request could never be "
                f"admitted under strict FIFO."
            )
        if request.state != RequestState.WAITING:
            raise ValueError(
                f"request req_id={req_id} is in state "
                f"{request.state.value}; only WAITING requests may be added"
            )
        self._seen_ids.add(req_id)
        self._waiting_queue.append(
            _Slot(
                request=request,
                req_id=req_id,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
        )

    def step(self) -> dict[int, int]:
        self._last_admitted = []
        self._last_evicted = []

        self._evict_finished()
        self._admit()
        emissions = self._execute()

        for slot in self._active:
            if len(slot.generated_tokens) >= slot.max_new_tokens:
                slot.finished = True

        for slot in self._active:
            self._sync_slot(slot)

        return emissions

    def _sync_slot(self, slot: _Slot) -> None:
        state = RequestState.FINISHED if slot.finished else RequestState.RUNNING
        slot.request._engine_sync(
            req_id=slot.req_id,
            prompt_tokens=slot.prompt_tokens,
            max_new_tokens=slot.max_new_tokens,
            state=state,
            prefill_done=slot.prefill_done,
            generated_tokens=tuple(slot.generated_tokens),
        )

    def _current_iteration_cost(self) -> int:
        cost = 0
        for slot in self._active:
            cost += slot.prompt_len if not slot.prefill_done else 1
        return cost

    def _admit(self) -> None:
        cost = self._current_iteration_cost()
        while self._waiting_queue:
            if len(self._active) >= self._max_batch_size:
                break
            head = self._waiting_queue[0]
            if cost + head.prompt_len > self._max_tokens_per_batch:
                # Strict FIFO: stop at the head; do not skip ahead.
                break
            self._waiting_queue.popleft()
            self._active.append(head)
            self._last_admitted.append(head.req_id)
            cost += head.prompt_len

    def _execute(self) -> dict[int, int]:
        emissions: dict[int, int] = {}
        for slot in self._active:
            if slot.finished:
                raise RuntimeError(
                    f"invariant violated: req_id={slot.req_id} is in active "
                    f"set with finished=True; _evict_finished should have "
                    f"removed it"
                )
            position = len(slot.generated_tokens)
            token = _placeholder_token(slot.req_id, position)
            slot.generated_tokens.append(token)
            slot.prefill_done = True
            emissions[slot.req_id] = token
        return emissions

    def _evict_finished(self) -> None:
        if not self._active:
            return
        survivors: list[_Slot] = []
        for slot in self._active:
            if slot.finished:
                self._sync_slot(slot)
                self._last_evicted.append(slot.req_id)
            else:
                survivors.append(slot)
        self._active = survivors


if __name__ == "__main__":
    engine = ContinuousBatchingEngine(max_batch_size=2, max_tokens_per_batch=8)
    req = SequenceRequest(req_id=42, prompt_tokens=[1, 2, 3], max_new_tokens=2)
    engine.add_request(req)
    print(f"after add: {req.state.value} waiting=1 active=0")
    for i in (1, 2, 3):
        emitted = engine.step()
        print(
            f"step {i}: emitted={emitted} admitted={engine.last_admitted} "
            f"evicted={engine.last_evicted} "
            f"active={[r.req_id for r in engine.active_requests]} "
            f"finished={req.is_finished()}"
        )
