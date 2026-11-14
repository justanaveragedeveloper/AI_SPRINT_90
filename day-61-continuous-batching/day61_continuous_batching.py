"""Continuous batching (iteration-level scheduling) reference simulator.

The problem
-----------
With static batching, the whole batch waits for the longest sequence before
any new request is accepted. That leaves compute idle. Continuous batching
re-fills the batch at every single decode step: finished requests leave,
waiting requests take their place at the very next scheduling opportunity,
and prefill work is chunked and interleaved alongside ongoing decodes.
This module models that policy on CPU so the scheduling rules can be
inspected and tested without a GPU.

Complexity (worked out from the code, per `schedule_iteration`)
---------------------------------------------------------------
Let A = number of active decode requests,
    W = number of waiting requests,
    T = total prompt tokens emitted this iteration.
  - Removing finished requests:            O(A + W)     two linear scans.
  - Moving empty prompts to decoding:      O(W)         one scan of the queue.
  - Scheduling decodes:                    O(A log A)   sort by admission
                                                       order, then a scan.
  - Scheduling prefills:                   O(W + T)     scan the queue and
                                                       copy T prompt tokens.
  - Total per iteration:                   O(A log A + W + T).
`add_request`: O(1) amortised.
`free_finished_requests`: O(A + W).

Why the decode sort is required: empty prompts become decodable the moment
they are admitted and can enter `_decoding_requests` before earlier-admitted
long prompts, so append order != admission order.

What this is NOT (honest limitations)
-------------------------------------
* Not an inference engine. No transformer, no CUDA, no GPU memory manager,
  no async event loop, no network serving.
* Not a benchmark. No throughput or latency numbers are claimed. This is a
  sequential CPU simulation and cannot demonstrate GPU utilisation or the
  "3x–5x" multiplier in the task brief. Those figures depend on workload,
  hardware, and model, and are not reproduced here.
* Not EOS-aware unless you pass an explicit `eos_token` per request. With
  `eos_token=None`, requests only complete via `max_new_tokens`.
* No fairness or starvation-freedom guarantee. FIFO admission order is
  deterministic; it is not a fairness proof.

Reproducibility
---------------
This module is deterministic. No randomness, no seed. Ordering comes from:
  * prefill work: FIFO admission order (the order of `add_request` calls);
  * decode work: admission order (`SequenceGroup._admission_order`),
    enforced by an explicit sort each iteration.
Both sources are explicit and stable, and both are covered by tests in
`test_continuous_batching.py`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "IterationBatch",
    "IterationScheduler",
    "PrefillChunk",
    "SequenceGroup",
    "State",
]


class State(Enum):
    """The four lifecycle states a request can be in.

    WAITING     - admitted but no prefill work has been scheduled yet.
    PREFILLING  - at least one prefill chunk scheduled; prompt tokens remain.
    DECODING    - the full prompt has been prefilled and the request can
                  take a decode slot.
    FINISHED    - generation complete; the request must never be scheduled
                  again.
    """

    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"


# --- validation helpers -----------------------------------------------------


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int (bool rejected), got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _require_non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int (bool rejected), got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _require_int_token(value: object, name: str) -> None:
    # Reject bools explicitly: isinstance(True, int) is True in Python, so a
    # True sneaking in as a token id would slip past the int check.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be int (bool rejected), got {type(value).__name__}"
        )


# --- sequence group ---------------------------------------------------------


class SequenceGroup:
    """One generation request and its lifecycle state.

    `prompt_tokens` is copied into an immutable tuple when the request is
    created. Accepts any `Sequence[int]` (list, tuple, range, ...). The
    caller may mutate their original object afterwards; scheduled prefill
    work is not affected. This is tested in
    `test_caller_mutation_of_prompt_after_construction_does_not_affect_scheduling`.

    The public lifecycle state is read-only. The scheduler advances it
    during prefill; step() advances it during generation.
    """

    __slots__ = (
        "_admission_order",
        "_already_admitted",
        "_generated_tokens",
        "_prefilled_token_count",
        "_prompt_tokens",
        "_req_id",
        "_state",
        "eos_token",
        "max_new_tokens",
    )

    def __init__(
        self,
        req_id: str,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        eos_token: int | None = None,
    ) -> None:
        if not isinstance(req_id, str) or not req_id:
            raise ValueError("req_id must be a non-empty str")
        if isinstance(prompt_tokens, (str, bytes)):
            raise TypeError("prompt_tokens must be a sequence of ints, not a string")
        try:
            prompt_as_tuple = tuple(prompt_tokens)
        except TypeError as exc:
            raise TypeError(
                "prompt_tokens must be a sequence of ints, got "
                f"{type(prompt_tokens).__name__}"
            ) from exc
        for position, token in enumerate(prompt_as_tuple):
            if isinstance(token, bool) or not isinstance(token, int):
                raise TypeError(
                    f"prompt_tokens[{position}] must be int (bool rejected),"
                    f" got {type(token).__name__}"
                )
        _require_non_negative_int(max_new_tokens, "max_new_tokens")
        if eos_token is not None:
            _require_int_token(eos_token, "eos_token")

        self._req_id: str = req_id
        self._prompt_tokens: tuple[int, ...] = prompt_as_tuple
        self.max_new_tokens: int = max_new_tokens
        self.eos_token: int | None = eos_token
        self._prefilled_token_count: int = 0
        self._generated_tokens: list[int] = []
        self._admission_order: int = -1  # set by IterationScheduler.add_request
        self._already_admitted: bool = False

        if max_new_tokens == 0:
            # A limit of zero finishes immediately without ever taking a
            # decode slot.
            self._state = State.FINISHED
        else:
            self._state = State.WAITING

    # ---- public read-only views ----

    @property
    def state(self) -> State:
        """Current lifecycle state. Read-only to outside callers."""
        return self._state

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def prompt_tokens(self) -> tuple[int, ...]:
        return self._prompt_tokens

    @property
    def prompt_len(self) -> int:
        return len(self._prompt_tokens)

    @property
    def generated_tokens(self) -> tuple[int, ...]:
        return tuple(self._generated_tokens)

    @property
    def num_generated(self) -> int:
        return len(self._generated_tokens)

    @property
    def prefilled_token_count(self) -> int:
        """How many prompt tokens have had prefill work scheduled."""
        return self._prefilled_token_count

    # ---- lifecycle ----

    def is_finished(self) -> bool:
        return self._state is State.FINISHED

    def step(self, token: int) -> None:
        """Append one generated token.

        Only valid in DECODING state. Raises:
          - TypeError if `token` is not an int (bools are rejected).
          - RuntimeError if the request is not in DECODING, or has already
            generated `max_new_tokens` tokens.
        """
        _require_int_token(token, "token")
        if self._state is not State.DECODING:
            raise RuntimeError(
                f"cannot step {self.req_id!r}: state is "
                f"{self._state.value}, step() requires DECODING"
            )
        if len(self._generated_tokens) >= self.max_new_tokens:
            raise RuntimeError(
                f"request {self.req_id!r} already generated "
                f"{self.max_new_tokens} tokens"
            )
        self._generated_tokens.append(token)
        if (self.eos_token is not None and token == self.eos_token) or (
            len(self._generated_tokens) >= self.max_new_tokens
        ):
            self._state = State.FINISHED

    # ---- helper used by the scheduler ----

    def _remaining_prompt_tokens(self) -> int:
        return self.prompt_len - self._prefilled_token_count

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SequenceGroup(req_id={self.req_id!r}, "
            f"state={self._state.value}, "
            f"prompt_len={self.prompt_len}, "
            f"prefilled={self._prefilled_token_count}, "
            f"generated={self.num_generated}/{self.max_new_tokens})"
        )


# --- iteration batch schema -------------------------------------------------


@dataclass(frozen=True)
class PrefillChunk:
    """One contiguous slice of a prompt scheduled for prefill this iteration.

    `start` is inclusive, `end` is exclusive, and
    `tokens == prompt[start:end]` (a fresh tuple copy). Concatenating every
    chunk ever emitted for a request across all iterations gives back the
    original prompt exactly once, in the original order.
    """

    request: SequenceGroup
    start: int
    end: int
    tokens: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def req_id(self) -> str:
        return self.request.req_id


@dataclass(frozen=True)
class IterationBatch:
    """The work that was scheduled for one iteration.

    Attributes
    ----------
    iteration
        1-based count of `schedule_iteration` calls on this scheduler.
    decode_requests
        Requests that got a decode slot this iteration, in admission order.
        Each one contributes exactly one token to the token budget.
    prefill_chunks
        Prefill chunks emitted this iteration, in FIFO admission order.
        Each chunk contributes `length` prompt tokens to the token budget.
    """

    iteration: int
    decode_requests: tuple[SequenceGroup, ...]
    prefill_chunks: tuple[PrefillChunk, ...]

    @property
    def active_requests(self) -> tuple[SequenceGroup, ...]:
        """Unique requests scheduled this iteration (decodes + prefills)."""
        seen: set[SequenceGroup] = set()
        result: list[SequenceGroup] = []
        for request in self.decode_requests:
            if request not in seen:
                seen.add(request)
                result.append(request)
        for chunk in self.prefill_chunks:
            if chunk.request not in seen:
                seen.add(chunk.request)
                result.append(chunk.request)
        return tuple(result)

    @property
    def num_items(self) -> int:
        """How many items this batch consumed against `max_batch_size`."""
        return len(self.decode_requests) + len(self.prefill_chunks)

    @property
    def token_count(self) -> int:
        """How much token work this batch consumed against `max_tokens_per_batch`."""
        return len(self.decode_requests) + sum(
            chunk.length for chunk in self.prefill_chunks
        )


# --- scheduler --------------------------------------------------------------


class IterationScheduler:
    """FIFO-admission continuous-batching scheduler (deterministic).

    Policy
    ------
    1. Remove FINISHED requests from the active and waiting pools.
    2. Move WAITING requests with empty prompts straight to DECODING
       (they have no prefill work; they can be decoded this iteration).
    3. Schedule DECODING requests in admission order, one token each, until
       either the item budget or the token budget runs out.
    4. Spend any remaining capacity on WAITING / PREFILLING requests in FIFO
       admission order. Each one emits at most `prefill_chunk_size` prompt
       tokens, further capped by the remaining token budget. Chunks that do
       not fit the remaining token budget are shortened rather than deferred,
       so a waiting request makes progress whenever at least one token of
       budget remains.
    5. When a request's prefill reaches the end of its prompt, it moves to
       DECODING. It becomes eligible for a decode slot on the *next*
       iteration (decodes are scheduled before prefills in each iteration).

    Design note — why we sort the active list every iteration
    ---------------------------------------------------------
    Items land in `_decoding_requests` in *activation* order, not admission
    order: empty-prompt requests are activated in step 2, and long-prompt
    requests are appended only when their last chunk is emitted. So an
    empty-prompt request admitted second can appear before a long-prompt
    request admitted first. Without sorting by `_admission_order`, decode
    precedence would follow activation instead of admission — which is not
    the documented policy. We chose a per-iteration sort over keeping the
    list sorted on insert because insert-time sorting needs a binary search
    per activation and still needs repair on removal. On workloads where
    everything is already in admission order, Python's Timsort detects the
    sorted run and this becomes an O(A) pass. Matters when a mixed batch of
    empty and long prompts arrives together.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_tokens_per_batch: int,
        prefill_chunk_size: int = 512,
    ) -> None:
        _require_positive_int(max_batch_size, "max_batch_size")
        _require_positive_int(max_tokens_per_batch, "max_tokens_per_batch")
        _require_positive_int(prefill_chunk_size, "prefill_chunk_size")

        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.prefill_chunk_size = prefill_chunk_size

        self._waiting_queue: deque[SequenceGroup] = deque()
        self._decoding_requests: list[SequenceGroup] = []
        self._active_request_ids: set[str] = set()

        self._next_admission_order: int = 0
        self._iteration_count: int = 0

    # ---- public API ----

    def add_request(self, request: SequenceGroup) -> None:
        """Admit a request into the scheduler.

        Rejects:
          - input that is not a SequenceGroup (TypeError);
          - a request object already admitted to any scheduler (ValueError);
          - a request whose state is not WAITING or FINISHED (ValueError),
            which protects against someone bypassing the read-only property;
          - a duplicate req_id among live or queued requests (ValueError).
        """
        if not isinstance(request, SequenceGroup):
            raise TypeError(
                f"add_request expects SequenceGroup, got {type(request).__name__}"
            )
        if request._already_admitted:
            raise ValueError(
                f"request {request.req_id!r} was already admitted; a "
                f"SequenceGroup is single-use per lifecycle"
            )
        if request.state not in (State.WAITING, State.FINISHED):
            raise ValueError(
                f"cannot admit {request.req_id!r}: state is "
                f"{request.state.value}, expected WAITING or FINISHED"
            )
        if request.req_id in self._active_request_ids:
            raise ValueError(
                f"duplicate req_id {request.req_id!r} among live or queued requests"
            )
        request._already_admitted = True
        request._admission_order = self._next_admission_order
        self._next_admission_order += 1

        if request.is_finished():
            # e.g. max_new_tokens == 0 — admitted but never scheduled.
            return

        self._active_request_ids.add(request.req_id)
        self._waiting_queue.append(request)

    def free_finished_requests(self) -> None:
        """Remove finished requests. Idempotent; safe to call any time.

        Provided for starter-code compatibility. `schedule_iteration` removes
        finished requests itself, so callers do not need to invoke this for
        backfill to work.
        """
        self._remove_finished_requests()

    def schedule_iteration(self) -> IterationBatch:
        self._remove_finished_requests()
        self._move_empty_prompts_to_decoding()

        remaining_item_budget = self.max_batch_size
        remaining_token_budget = self.max_tokens_per_batch

        # (1) Schedule decodes in admission order.
        decodable_requests = [
            request
            for request in self._decoding_requests
            if request.state is State.DECODING and not request.is_finished()
        ]
        decodable_requests.sort(key=lambda request: request._admission_order)
        scheduled_decodes: list[SequenceGroup] = []
        for request in decodable_requests:
            if remaining_item_budget <= 0 or remaining_token_budget <= 0:
                break
            scheduled_decodes.append(request)
            remaining_item_budget -= 1
            remaining_token_budget -= 1

        # (2) Schedule prefills in FIFO admission order.
        scheduled_prefills: list[PrefillChunk] = []
        still_waiting: deque[SequenceGroup] = deque()
        while self._waiting_queue:
            # _remove_finished_requests and _move_empty_prompts_to_decoding
            # guarantee every request here still has remaining prompt tokens.
            request = self._waiting_queue.popleft()
            if remaining_item_budget <= 0 or remaining_token_budget <= 0:
                still_waiting.append(request)
                continue

            chunk_length = min(
                self.prefill_chunk_size,
                request._remaining_prompt_tokens(),
                remaining_token_budget,
            )
            chunk_start = request._prefilled_token_count
            chunk_end = chunk_start + chunk_length
            chunk_tokens = request._prompt_tokens[chunk_start:chunk_end]
            request._state = State.PREFILLING
            request._prefilled_token_count = chunk_end
            scheduled_prefills.append(
                PrefillChunk(request, chunk_start, chunk_end, chunk_tokens)
            )
            remaining_item_budget -= 1
            remaining_token_budget -= chunk_length
            if request._prefilled_token_count >= request.prompt_len:
                request._state = State.DECODING
                self._decoding_requests.append(request)
            else:
                still_waiting.append(request)
        self._waiting_queue = still_waiting

        self._iteration_count += 1
        return IterationBatch(
            iteration=self._iteration_count,
            decode_requests=tuple(scheduled_decodes),
            prefill_chunks=tuple(scheduled_prefills),
        )

    # ---- internals ----

    def _remove_finished_requests(self) -> None:
        """Remove FINISHED requests from the active and waiting pools.

        Retired requests are simply dropped, not stored. Callers can check
        `request.is_finished()` on the references they already hold, so
        keeping finished requests here would only leak memory across a
        long-running simulation.
        """
        still_active: list[SequenceGroup] = []
        for request in self._decoding_requests:
            if request.is_finished():
                self._active_request_ids.discard(request.req_id)
            else:
                still_active.append(request)
        self._decoding_requests = still_active

        still_waiting: deque[SequenceGroup] = deque()
        for request in self._waiting_queue:
            if request.is_finished():
                self._active_request_ids.discard(request.req_id)
            else:
                still_waiting.append(request)
        self._waiting_queue = still_waiting

    def _move_empty_prompts_to_decoding(self) -> None:
        if not self._waiting_queue:
            return
        still_waiting: deque[SequenceGroup] = deque()
        for request in self._waiting_queue:
            if request.prompt_len == 0:
                request._state = State.DECODING
                self._decoding_requests.append(request)
            else:
                still_waiting.append(request)
        self._waiting_queue = still_waiting
