"""Day 45 — a deterministic state machine that routes a query to one of three paths.

What it does
------------
Takes a query string, picks one of three intents (analytical_rag,
graph_traversal, direct_execution), walks that intent through an explicit
finite state machine, and returns the result along with the full transition
history. It's the orchestration layer — the piece that decides *which*
retriever runs — not the retriever itself.

What it doesn't do
------------------
No embeddings, no BM25, no RRF, no Knowledge Graph, no tool execution, no LLM
synthesis. Route selection is a keyword-count heuristic. The retrieval outputs
are placeholder strings. The formulas in the Day 45 spec —

    Score_RAG   = a*cos(q_v, d_v) + (1-a)*BM25(q_s, D)
    Score_GRAPH = 1[|E_q| > 0] * sum(deg(e))

— describe where this state machine would plug into a real system. They are
not computed here, and nothing in this module pretends otherwise.

Determinism
-----------
Same input, same output, every run, every machine. No random, no time, no
env vars, no network. Every user-visible collection is built by sorted() or
by per-instance append order.

Complexity
----------
With Q = query length and U = distinct tokens:
    tokenize      O(Q)
    score         O(U)         (keyword sets are size 6)
    transitions   O(1) each, at most 4 per query
    payload       O(Q)
Total: O(Q) time, O(Q) memory per call. Not O(K log N) — nothing in this
module touches an index or a graph.

Notes for readers
-----------------
- Engine is single-use by design. Second call raises EngineReuseError. This
  is the fix for the stale-intent bug the spec warns about.
- Custom keyword overrides are normalized (strip, lowercase, drop blanks).
  IntentClassifier(graph_keywords=["Connected"]) works like you'd expect.
- Transition history records (from, to, reason, seq), not just the departing
  state, so the recorded path can be replayed against the real one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "AgentState",
    "AgentStateEngine",
    "Classification",
    "EngineReuseError",
    "IllegalTransitionError",
    "IntentClassifier",
    "InvalidQueryError",
    "ProcessingStatus",
    "RouteResult",
    "RoutingIntent",
    "Transition",
]


# ---------------------------------------------------------------------------
# States and intents
# ---------------------------------------------------------------------------


class AgentState(Enum):
    INIT = "init"
    ROUTING = "routing"
    VECTOR_RETRIEVAL = "vector_retrieval"
    GRAPH_TRAVERSAL = "graph_traversal"
    SYNTHESIS = "synthesis"
    TERMINATED = "terminated"
    ERROR = "error"


class RoutingIntent(Enum):
    ANALYTICAL_RAG = "analytical_rag"
    GRAPH_TRAVERSAL = "graph_traversal"
    DIRECT_EXECUTION = "direct_execution"


class ProcessingStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"


# Enum over raw strings: a typo in "routinng" raises at construction instead
# of silently creating an unreachable second state.


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidQueryError(ValueError):
    """Classifier got a non-string or blank query."""


class EngineReuseError(RuntimeError):
    """AgentStateEngine is single-use. Build a new one per query."""


class IllegalTransitionError(RuntimeError):
    """Someone tried to move to a state the adjacency map forbids."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Classification:
    """The evidence behind a routing decision.

    Carries the scores and the matched keywords, not just the winning intent,
    so Transition.reason can explain *why* without re-running the classifier.
    """

    intent: RoutingIntent
    graph_score: int
    rag_score: int
    unique_token_count: int
    matched_graph_keywords: tuple[str, ...]
    matched_rag_keywords: tuple[str, ...]

    def evidence(self) -> str:
        """One-line, order-stable summary. Sorted keywords, fixed field order."""
        return (
            f"graph_score={self.graph_score} rag_score={self.rag_score} "
            f"unique_tokens={self.unique_token_count} "
            f"matched_graph={self.matched_graph_keywords} "
            f"matched_rag={self.matched_rag_keywords} "
            f"intent={self.intent.value}"
        )


# Frozen so evidence can't drift between the transition reason and the
# returned result. Matched keywords are tuples, not sets, because they end up
# in a user-visible string and need a fixed order.


class IntentClassifier:
    """Deterministic keyword router.

    Rules (from the Day 45 blueprint, unchanged):

        graph_score = |tokens & GRAPH_KEYWORDS|
        rag_score   = |tokens & RAG_KEYWORDS|

        if graph_score > rag_score:            -> GRAPH_TRAVERSAL
        elif rag_score > 0 or |tokens| > 5:    -> ANALYTICAL_RAG
        else:                                  -> DIRECT_EXECUTION

    Tokenization is `query.lower().split()`. That means no punctuation
    stripping, no stemming, no stop words. "search?" is not "search". This is
    a documented contract, not an oversight — see the test that pins it.

    Custom vocabularies are normalized: each entry is stripped and lowercased,
    blanks are dropped, non-strings raise TypeError.
    """

    GRAPH_KEYWORDS: frozenset[str] = frozenset(
        {"relationship", "connected", "depends", "uses", "hierarchy", "builds"}
    )
    RAG_KEYWORDS: frozenset[str] = frozenset(
        {"search", "find", "document", "explain", "detail", "concept"}
    )

    # Applies to *distinct* tokens, matching the blueprint's
    # len(set(query.lower().split())).
    LONG_QUERY_UNIQUE_TOKEN_THRESHOLD: int = 5

    def __init__(
        self,
        graph_keywords: Iterable[str] | None = None,
        rag_keywords: Iterable[str] | None = None,
    ) -> None:
        self.graph_keywords: frozenset[str] = self._normalize_keywords(
            self.GRAPH_KEYWORDS if graph_keywords is None else graph_keywords
        )
        self.rag_keywords: frozenset[str] = self._normalize_keywords(
            self.RAG_KEYWORDS if rag_keywords is None else rag_keywords
        )

    # -- public API ---------------------------------------------------------

    def classify(self, query: str) -> RoutingIntent:
        """Just the winning intent."""
        return self.explain(query).intent

    def explain(self, query: str) -> Classification:
        """Full decision evidence.

        Raises:
            InvalidQueryError: query is not a non-blank str.
        """
        token_set = self._token_set(query)

        # sorted() because set intersection has no defined order, and these
        # tuples end up in a user-visible reason string.
        matched_graph = tuple(sorted(token_set & self.graph_keywords))
        matched_rag = tuple(sorted(token_set & self.rag_keywords))

        graph_score = len(matched_graph)
        rag_score = len(matched_rag)
        unique_token_count = len(token_set)

        if graph_score > rag_score:
            intent = RoutingIntent.GRAPH_TRAVERSAL
        elif (
            rag_score > 0 or unique_token_count > self.LONG_QUERY_UNIQUE_TOKEN_THRESHOLD
        ):
            intent = RoutingIntent.ANALYTICAL_RAG
        else:
            intent = RoutingIntent.DIRECT_EXECUTION

        return Classification(
            intent=intent,
            graph_score=graph_score,
            rag_score=rag_score,
            unique_token_count=unique_token_count,
            matched_graph_keywords=matched_graph,
            matched_rag_keywords=matched_rag,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _normalize_keywords(keywords: Iterable[str]) -> frozenset[str]:
        """Strip, lowercase, drop blanks, reject non-strings.

        We normalize over documenting "must be lowercase" because the latter
        fails silently — you get a classifier that just doesn't match, and
        you find out three layers up.
        """
        normalized: set[str] = set()
        for keyword in keywords:
            if not isinstance(keyword, str):
                raise TypeError(f"keywords must be str, got {type(keyword).__name__}")
            cleaned = keyword.strip().lower()
            if cleaned:
                normalized.add(cleaned)
        return frozenset(normalized)

    @staticmethod
    def _token_set(query: str) -> frozenset[str]:
        """Lowercase, split on whitespace, deduplicate.

        str.split() with no argument splits on any whitespace run and never
        yields empty tokens. No regex compiled per call.
        """
        if not isinstance(query, str):
            raise InvalidQueryError(f"query must be a str, got {type(query).__name__}")
        if not query.strip():
            raise InvalidQueryError(
                "query must contain at least one non-whitespace character"
            )
        return frozenset(query.lower().split())


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One edge of the state machine."""

    sequence: int
    from_state: AgentState
    to_state: AgentState
    reason: str


# (from, to, reason, seq) rather than (state, reason): the latter can't
# reconstruct the terminal state, so the recorded path and the real one can
# diverge without anyone noticing.


@dataclass(frozen=True)
class RouteResult:
    """What process_query returns."""

    status: ProcessingStatus
    intent: RoutingIntent | None
    payload: str
    state_path: tuple[AgentState, ...]
    matched_keywords: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, object]:
        """Plain-primitive view. Safe to json.dumps()."""
        return {
            "status": self.status.value,
            "intent": self.intent.value if self.intent is not None else None,
            "payload": self.payload,
            "state_path": [state.value for state in self.state_path],
            "matched_keywords": list(self.matched_keywords),
            "detail": self.detail,
        }


_ALLOWED_TRANSITIONS: Mapping[AgentState, frozenset[AgentState]] = {
    AgentState.INIT: frozenset({AgentState.ROUTING, AgentState.ERROR}),
    AgentState.ROUTING: frozenset(
        {
            AgentState.VECTOR_RETRIEVAL,
            AgentState.GRAPH_TRAVERSAL,
            AgentState.SYNTHESIS,
        }
    ),
    AgentState.VECTOR_RETRIEVAL: frozenset({AgentState.SYNTHESIS}),
    AgentState.GRAPH_TRAVERSAL: frozenset({AgentState.SYNTHESIS}),
    AgentState.SYNTHESIS: frozenset({AgentState.TERMINATED}),
    AgentState.TERMINATED: frozenset(),
    AgentState.ERROR: frozenset(),
}

# Explicit map rather than a free-form setter because a setter can't tell an
# intended route from a bug. This turns "someone edited process_query wrong"
# into an exception at the offending line instead of a wrong answer.


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class AgentStateEngine:
    """One query per instance.

    Second call raises EngineReuseError. Silently resetting would re-introduce
    exactly the stale-intent bug the spec warns about, and it's easy to forget
    to reset in a long-lived handler.
    """

    def __init__(self, classifier: IntentClassifier | None = None) -> None:
        self._state: AgentState = AgentState.INIT
        # Injection is for tests: they can drive routes without building
        # keyword-laden queries, and the machine's behavior stays independent
        # of the classifier's vocabulary.
        self._classifier: IntentClassifier = (
            classifier if classifier is not None else IntentClassifier()
        )
        self._history: list[Transition] = []
        self._context: dict[str, str] = {}
        self._consumed: bool = False

    # -- read-only views ----------------------------------------------------

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def history(self) -> tuple[Transition, ...]:
        return tuple(self._history)

    @property
    def state_path(self) -> tuple[AgentState, ...]:
        """Full path including the implicit INIT."""
        return (AgentState.INIT,) + tuple(t.to_state for t in self._history)

    @property
    def context_buffer(self) -> dict[str, str]:
        """A copy. Callers can't corrupt the engine's own record."""
        return dict(self._context)

    # -- core ---------------------------------------------------------------

    def _transition(self, next_state: AgentState, reason: str) -> None:
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if next_state not in allowed:
            raise IllegalTransitionError(
                f"{self._state.value} -> {next_state.value} is not allowed "
                f"(allowed: {sorted(s.value for s in allowed)})"
            )
        self._history.append(
            Transition(
                sequence=len(self._history) + 1,
                from_state=self._state,
                to_state=next_state,
                reason=reason,
            )
        )
        self._state = next_state

    # -- public entry point -------------------------------------------------

    def process_query(self, query: str) -> RouteResult:
        """Run one query to TERMINATED or ERROR.

        Raises:
            EngineReuseError: engine already used.
        """
        if self._consumed:
            raise EngineReuseError(
                "AgentStateEngine is single-use; build a new one per query. "
                f"This one ended in {self._state.value!r}."
            )
        self._consumed = True

        # Validation runs before ROUTING so the error path is INIT -> ERROR
        # and nothing else. No "log and continue" branch anywhere.
        if not isinstance(query, str):
            return self._reject(f"query type {type(query).__name__} is not str")
        if not query.strip():
            return self._reject("query is empty or whitespace-only")

        self._transition(AgentState.ROUTING, "query accepted; classifying intent")

        classification = self._classifier.explain(query)
        self._context["query"] = query
        self._context["intent"] = classification.intent.value

        route_reason = (
            f"route={classification.intent.value} | {classification.evidence()}"
        )

        if classification.intent is RoutingIntent.GRAPH_TRAVERSAL:
            self._transition(AgentState.GRAPH_TRAVERSAL, route_reason)
            self._context["retrieval_output"] = self._graph_placeholder(
                query, classification
            )
            self._transition(
                AgentState.SYNTHESIS, "consolidating graph placeholder output"
            )
        elif classification.intent is RoutingIntent.ANALYTICAL_RAG:
            self._transition(AgentState.VECTOR_RETRIEVAL, route_reason)
            self._context["retrieval_output"] = self._rag_placeholder(
                query, classification
            )
            self._transition(
                AgentState.SYNTHESIS, "consolidating vector placeholder output"
            )
        else:
            # Direct execution: ROUTING -> SYNTHESIS, no retrieval hop.
            self._transition(AgentState.SYNTHESIS, route_reason)
            self._context["retrieval_output"] = self._direct_placeholder(
                query, classification
            )

        self._transition(AgentState.TERMINATED, "pipeline completed")

        return RouteResult(
            status=ProcessingStatus.SUCCESS,
            intent=classification.intent,
            payload=self._context["retrieval_output"],
            state_path=self.state_path,
            matched_keywords=tuple(
                sorted(
                    classification.matched_graph_keywords
                    + classification.matched_rag_keywords
                )
            ),
            detail=classification.evidence(),
        )

    # -- helpers ------------------------------------------------------------

    def _reject(self, reason: str) -> RouteResult:
        self._transition(AgentState.ERROR, f"rejected: {reason}")
        return RouteResult(
            status=ProcessingStatus.ERROR,
            intent=None,
            payload="",
            state_path=self.state_path,
            matched_keywords=(),
            detail=f"Invalid query input: {reason}.",
        )

    @staticmethod
    def _graph_placeholder(query: str, classification: Classification) -> str:
        return (
            "[placeholder:graph_traversal] no Knowledge Graph was queried; "
            f"query={query!r} "
            f"matched_graph_keywords={classification.matched_graph_keywords}"
        )

    @staticmethod
    def _rag_placeholder(query: str, classification: Classification) -> str:
        return (
            "[placeholder:analytical_rag] no dense, BM25, or RRF retrieval was "
            f"executed; query={query!r} "
            f"matched_rag_keywords={classification.matched_rag_keywords}"
        )

    @staticmethod
    def _direct_placeholder(query: str, classification: Classification) -> str:
        return (
            "[placeholder:direct_execution] no external execution engine was "
            f"invoked; query={query!r} "
            f"matched_keywords="
            f"{tuple(sorted(classification.matched_graph_keywords + classification.matched_rag_keywords))}"
        )
