"""Tests for day45_agentic_router.

Every test names the mutation it catches. If you can't finish the sentence
"this fails if X", the test isn't doing anything.

Tags:
    [ROBUST]   catches a specific wrong implementation
    [CONTRACT] pins a public boundary (would only fail if the interface changed)
    [GUARD]    catches a defensive-check removal

Run: pytest -v test_agentic_router.py
"""

import json

import pytest
from day45_agentic_router import (
    AgentState,
    AgentStateEngine,
    Classification,
    EngineReuseError,
    IllegalTransitionError,
    IntentClassifier,
    InvalidQueryError,
    ProcessingStatus,
    RoutingIntent,
)

# Each intent has one fixture that reaches it via exactly one rule, so the
# "fails if" sentences stay precise.
GRAPH_QUERY = "What depends on attention mechanisms?"  # 1 graph, 0 rag, 5 tokens
RAG_QUERY = "Search document details for vector indices"  # 2 rag, 0 graph
DIRECT_QUERY = "hello world"  # no keywords, 2 tokens


# ===========================================================================
# Intent classification
# ===========================================================================


def test_graph_keyword_match_outranks_a_single_rag_keyword():
    """[ROBUST] Fails if the branches are reordered: this query scores 1 on
    RAG and 6 tokens, so checking `rag_score > 0` before the strict graph
    comparison would return ANALYTICAL_RAG instead of GRAPH_TRAVERSAL."""
    classifier = IntentClassifier()
    classification = classifier.explain("explain what depends on connected layers")
    assert (classification.graph_score, classification.rag_score) == (2, 1)
    assert classification.intent is RoutingIntent.GRAPH_TRAVERSAL


def test_graph_rag_tie_resolves_to_rag_not_graph():
    """[ROBUST] Fails if `>` becomes `>=`: a 1-1 tie would then win for graph."""
    classification = IntentClassifier().explain("search the connected graph")
    assert (classification.graph_score, classification.rag_score) == (1, 1)
    assert classification.intent is RoutingIntent.ANALYTICAL_RAG


def test_single_rag_keyword_routes_to_rag_on_a_short_query():
    """[ROBUST] Fails if the RAG branch's `or` becomes `and`
    (`rag_score > 0 and unique_tokens > 5`): this 3-token query has one RAG
    keyword and would fall through to DIRECT_EXECUTION."""
    classification = IntentClassifier().explain("find the concept")
    assert classification.unique_token_count == 3
    assert classification.intent is RoutingIntent.ANALYTICAL_RAG


def test_long_keywordless_query_routes_to_rag_by_length_fallback():
    """[ROBUST] Fails if the `> 5` fallback is dropped or raised: six distinct
    non-keyword tokens would become DIRECT_EXECUTION."""
    assert (
        IntentClassifier().classify("alpha beta gamma delta epsilon zeta")
        is RoutingIntent.ANALYTICAL_RAG
    )


def test_five_token_keywordless_query_stays_direct():
    """[ROBUST] Boundary partner to the previous test. Fails if the threshold
    moves from `> 5` to `>= 5`."""
    assert (
        IntentClassifier().classify("alpha beta gamma delta epsilon")
        is RoutingIntent.DIRECT_EXECUTION
    )


def test_repeated_tokens_do_not_trigger_the_long_query_fallback():
    """[ROBUST] Pins that the fallback counts *distinct* tokens. Fails if
    `len(set(...))` becomes `len(query.split())`, which would send this
    6-token query to ANALYTICAL_RAG."""
    classification = IntentClassifier().explain("alpha alpha alpha alpha alpha alpha")
    assert classification.unique_token_count == 1
    assert classification.intent is RoutingIntent.DIRECT_EXECUTION


def test_keyword_matching_is_case_insensitive():
    """[ROBUST] Fails if `.lower()` is dropped from tokenization: uppercase
    keywords stop matching and this falls to DIRECT_EXECUTION."""
    assert (
        IntentClassifier().classify("SEARCH the DOCUMENT")
        is RoutingIntent.ANALYTICAL_RAG
    )


def test_punctuation_attached_tokens_do_not_match_keywords():
    """[CONTRACT] Pins tokenization: `str.lower().split()` doesn't strip
    punctuation, so "search?" is not "search". Fails if punctuation stripping
    is added — that's a contract change, not a bug fix."""
    classifier = IntentClassifier()
    assert classifier.classify("search?") is RoutingIntent.DIRECT_EXECUTION
    assert classifier.classify("search") is RoutingIntent.ANALYTICAL_RAG


def test_matched_keyword_evidence_is_canonically_sorted():
    """[ROBUST] Fails if `sorted()` is dropped: set iteration order is not
    alphabetical in CPython, so the tuple would not match the literal."""
    classification = IntentClassifier().explain("uses builds connected")
    assert classification.matched_graph_keywords == ("builds", "connected", "uses")
    assert classification.graph_score == 3


def test_duplicate_keyword_entries_cannot_inflate_a_score():
    """[ROBUST] Fails if keywords become a list scored by occurrence count: a
    duplicated entry would report graph_score=2 for one token."""
    classifier = IntentClassifier(graph_keywords=["connected", "connected"])
    classification = classifier.explain("connected")
    assert classification.graph_score == 1
    assert classification.intent is RoutingIntent.GRAPH_TRAVERSAL


@pytest.mark.parametrize(
    "bad", [None, 123, 4.5, b"search", ["search"], {"q": "search"}]
)
def test_classifier_rejects_non_string_queries(bad):
    """[GUARD] Fails if the isinstance check is removed: `.lower()` would
    AttributeError (or bytes would silently produce zero matches) instead of
    InvalidQueryError."""
    with pytest.raises(InvalidQueryError):
        IntentClassifier().classify(bad)


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n\n", "   \t \n "])
def test_classifier_rejects_blank_queries(blank):
    """[GUARD] Fails if `query.strip()` is removed: a blank query would score
    zero of everything and be silently classified as DIRECT_EXECUTION."""
    with pytest.raises(InvalidQueryError):
        IntentClassifier().classify(blank)


# ===========================================================================
# Custom vocabulary
# ===========================================================================


def test_custom_keywords_are_normalized_to_lowercase():
    """[ROBUST] Fails if overrides are stored verbatim:
    `graph_keywords=["Connected"]` would never match the lowercase token, and
    this would route to DIRECT_EXECUTION."""
    classifier = IntentClassifier(
        graph_keywords=["Connected"],
        rag_keywords=["Search"],
    )
    assert classifier.graph_keywords == frozenset({"connected"})
    assert classifier.rag_keywords == frozenset({"search"})
    assert classifier.classify("connected") is RoutingIntent.GRAPH_TRAVERSAL
    assert classifier.classify("SEARCH") is RoutingIntent.ANALYTICAL_RAG


def test_blank_and_whitespace_custom_keywords_are_dropped():
    """[ROBUST] Fails if blanks survive normalization: the resulting frozenset
    would contain "" and "   " and the equality check below would fail."""
    classifier = IntentClassifier(
        graph_keywords=[" ", "CONNECTED", "", " connected ", "\tconnected\n"],
    )
    assert classifier.graph_keywords == frozenset({"connected"})


def test_custom_keywords_must_be_strings():
    """[GUARD] Fails if the isinstance check inside `_normalize_keywords` is
    removed: a non-string entry would linger in the set and raise
    AttributeError at some distant call site during `.strip()`."""
    with pytest.raises(TypeError):
        IntentClassifier(graph_keywords=["connected", 123])


# ===========================================================================
# State paths — exact sequences, not "contains"
# ===========================================================================


def test_graph_route_records_the_exact_state_path():
    """[ROBUST] Fails if graph skips SYNTHESIS, reorders states, or reports
    the wrong terminal state. A membership check would miss a skip; the exact
    tuple does not."""
    engine = AgentStateEngine()
    result = engine.process_query(GRAPH_QUERY)

    assert result.state_path == (
        AgentState.INIT,
        AgentState.ROUTING,
        AgentState.GRAPH_TRAVERSAL,
        AgentState.SYNTHESIS,
        AgentState.TERMINATED,
    )
    assert result.status is ProcessingStatus.SUCCESS
    assert result.intent is RoutingIntent.GRAPH_TRAVERSAL
    assert engine.state is AgentState.TERMINATED


def test_analytical_rag_route_records_the_exact_state_path():
    """[ROBUST] Fails if RAG reuses the graph state or skips
    VECTOR_RETRIEVAL."""
    engine = AgentStateEngine()
    result = engine.process_query(RAG_QUERY)

    assert result.state_path == (
        AgentState.INIT,
        AgentState.ROUTING,
        AgentState.VECTOR_RETRIEVAL,
        AgentState.SYNTHESIS,
        AgentState.TERMINATED,
    )
    assert result.status is ProcessingStatus.SUCCESS
    assert result.intent is RoutingIntent.ANALYTICAL_RAG


def test_direct_route_records_the_exact_state_path():
    """[ROBUST] Fails if direct execution falls into the RAG or graph branch,
    which would insert a retrieval hop the spec forbids."""
    engine = AgentStateEngine()
    result = engine.process_query(DIRECT_QUERY)

    assert result.state_path == (
        AgentState.INIT,
        AgentState.ROUTING,
        AgentState.SYNTHESIS,
        AgentState.TERMINATED,
    )
    assert result.intent is RoutingIntent.DIRECT_EXECUTION


def test_direct_route_never_enters_a_retrieval_state():
    """[ROBUST] Catches the sneaky version: engine transitions into
    VECTOR_RETRIEVAL and immediately out, so the final result still looks
    right but the direct path is polluted."""
    result = AgentStateEngine().process_query(DIRECT_QUERY)
    assert AgentState.VECTOR_RETRIEVAL not in result.state_path
    assert AgentState.GRAPH_TRAVERSAL not in result.state_path


@pytest.mark.parametrize("query", [GRAPH_QUERY, RAG_QUERY, DIRECT_QUERY])
def test_every_successful_route_passes_through_synthesis_before_termination(query):
    """[ROBUST] Fails if any route terminates without SYNTHESIS, or ends
    anywhere other than TERMINATED. Parametrized so a fix applied to one route
    can't silently miss another."""
    result = AgentStateEngine().process_query(query)

    assert result.status is ProcessingStatus.SUCCESS
    assert result.state_path[0] is AgentState.INIT
    assert result.state_path[1] is AgentState.ROUTING
    assert AgentState.SYNTHESIS in result.state_path
    assert result.state_path[-1] is AgentState.TERMINATED
    assert result.state_path.index(AgentState.SYNTHESIS) < len(result.state_path) - 1


# ===========================================================================
# Invalid input
# ===========================================================================


def test_empty_query_terminates_in_error_directly_from_init():
    """[ROBUST] Fails if the blank check is dropped or moved after ROUTING:
    the path would then begin (INIT, ROUTING, ...) instead of (INIT, ERROR)."""
    engine = AgentStateEngine()
    result = engine.process_query("")

    assert result.state_path == (AgentState.INIT, AgentState.ERROR)
    assert result.status is ProcessingStatus.ERROR
    assert result.intent is None
    assert engine.state is AgentState.ERROR


@pytest.mark.parametrize("blank", [" ", "\t\n", "     "])
def test_whitespace_only_query_terminates_in_error(blank):
    """[ROBUST] Fails if the guard is `not query` instead of
    `not query.strip()`: whitespace-only input would route as a 1-token query
    into DIRECT_EXECUTION."""
    engine = AgentStateEngine()
    result = engine.process_query(blank)
    assert result.state_path == (AgentState.INIT, AgentState.ERROR)
    assert engine.state is AgentState.ERROR


@pytest.mark.parametrize("bad", [None, 123, 4.5, b"search", ["search"]])
def test_non_string_query_terminates_in_error_without_raising(bad):
    """[ROBUST] Fails if the engine's isinstance guard is removed: the
    classifier's InvalidQueryError would escape process_query instead of the
    engine returning its documented error result."""
    engine = AgentStateEngine()
    result = engine.process_query(bad)

    assert result.status is ProcessingStatus.ERROR
    assert result.state_path == (AgentState.INIT, AgentState.ERROR)
    assert engine.state is AgentState.ERROR


def test_invalid_query_never_reaches_routing_or_success():
    """[ROBUST] Fails if the engine records the error but keeps going — the
    "log and continue" mutation."""
    engine = AgentStateEngine()
    result = engine.process_query("   ")

    assert result.status is not ProcessingStatus.SUCCESS
    assert AgentState.ROUTING not in result.state_path
    assert AgentState.TERMINATED not in result.state_path
    assert engine.state is AgentState.ERROR


# ===========================================================================
# Transition history
# ===========================================================================


def test_history_records_from_and_to_state_for_every_transition_in_order():
    """[ROBUST] Fails if history stores only the departing state (the
    blueprint's shape), which makes the terminal state unrecoverable, or if
    transitions append out of order."""
    engine = AgentStateEngine()
    result = engine.process_query(RAG_QUERY)

    recorded = [(t.from_state, t.to_state) for t in engine.history]
    assert recorded == [
        (AgentState.INIT, AgentState.ROUTING),
        (AgentState.ROUTING, AgentState.VECTOR_RETRIEVAL),
        (AgentState.VECTOR_RETRIEVAL, AgentState.SYNTHESIS),
        (AgentState.SYNTHESIS, AgentState.TERMINATED),
    ]
    assert result.state_path == (AgentState.INIT,) + tuple(
        t.to_state for t in engine.history
    )


def test_history_sequence_numbers_are_one_based_and_monotonic():
    """[ROBUST] Fails if sequence numbers are zero-based, reused, or computed
    from something other than the current history length — any of which makes
    two transitions indistinguishable in a trace."""
    engine = AgentStateEngine()
    engine.process_query(GRAPH_QUERY)
    sequences = [t.sequence for t in engine.history]
    assert sequences == list(range(1, len(sequences) + 1))


def test_history_sequence_matches_position_in_history_list():
    """[ROBUST] Fails if `sequence` comes from a shared counter (e.g. a class
    attribute) instead of the per-instance history length: sequence and index
    would diverge."""
    engine = AgentStateEngine()
    engine.process_query(RAG_QUERY)
    for index, transition in enumerate(engine.history, start=1):
        assert transition.sequence == index


def test_every_transition_reason_is_non_empty_and_explains_the_decision():
    """[ROBUST] Fails if any reason is empty or boilerplate, or if the route
    reason drops the classifier evidence that makes the path auditable."""
    engine = AgentStateEngine()
    engine.process_query(GRAPH_QUERY)

    assert all(t.reason.strip() for t in engine.history)
    route_reason = engine.history[1].reason  # ROUTING -> GRAPH_TRAVERSAL
    assert "route=graph_traversal" in route_reason
    assert "graph_score=1" in route_reason
    assert "rag_score=0" in route_reason


def test_route_transition_reason_matches_result_detail():
    """[ROBUST] Fails if `RouteResult.detail` and the ROUTING transition's
    reason come from different evidence — the returned detail would no longer
    be traceable to the history entry that explains the route."""
    engine = AgentStateEngine()
    result = engine.process_query("explain what depends on connected layers")

    route_transition = engine.history[1]
    assert result.detail in route_transition.reason
    assert "intent=graph_traversal" in route_transition.reason


# ===========================================================================
# Payload / intent consistency
# ===========================================================================


@pytest.mark.parametrize(
    "query, expected_intent, payload_marker",
    [
        (GRAPH_QUERY, RoutingIntent.GRAPH_TRAVERSAL, "[placeholder:graph_traversal]"),
        (RAG_QUERY, RoutingIntent.ANALYTICAL_RAG, "[placeholder:analytical_rag]"),
        (
            DIRECT_QUERY,
            RoutingIntent.DIRECT_EXECUTION,
            "[placeholder:direct_execution]",
        ),
    ],
)
def test_payload_marker_matches_the_selected_intent(
    query, expected_intent, payload_marker
):
    """[ROBUST] Fails if the three placeholder payloads get swapped by a
    copy-paste edit — you'd get graph text under an analytical_rag intent."""
    result = AgentStateEngine().process_query(query)
    assert result.intent is expected_intent
    assert result.payload.startswith(payload_marker)
    assert query in result.payload


def test_result_intent_equals_the_classifier_decision_for_the_same_query():
    """[ROBUST] Fails if the engine routes on one classification but records
    a different intent (e.g. stores a hard-coded default)."""
    for query in (
        GRAPH_QUERY,
        RAG_QUERY,
        DIRECT_QUERY,
        "alpha beta gamma delta epsilon zeta",
    ):
        result = AgentStateEngine().process_query(query)
        assert result.intent is IntentClassifier().classify(query)


def test_result_matched_keywords_equal_the_classifier_evidence():
    """[ROBUST] Fails if `RouteResult.matched_keywords` is hard-coded to ()
    or built from the whole vocabularies rather than the matched tokens."""
    query = "explain what depends on connected layers"
    classification = IntentClassifier().explain(query)
    expected = tuple(
        sorted(
            classification.matched_graph_keywords + classification.matched_rag_keywords
        )
    )
    result = AgentStateEngine().process_query(query)
    assert result.matched_keywords == expected == ("connected", "depends", "explain")


# ===========================================================================
# Engine lifetime
# ===========================================================================


def test_second_query_on_the_same_engine_is_rejected():
    """[CONTRACT] Fails if the engine silently accepts a second query (the
    blueprint's behavior), which would leak the first query's intent and
    payload into the second result."""
    engine = AgentStateEngine()
    first = engine.process_query(GRAPH_QUERY)

    with pytest.raises(EngineReuseError):
        engine.process_query(RAG_QUERY)

    assert first.intent is RoutingIntent.GRAPH_TRAVERSAL
    assert engine.state is AgentState.TERMINATED


def test_engine_that_ended_in_error_is_also_spent():
    """[CONTRACT] Fails if the single-use flag is only set on the success
    path, letting a caller "retry" on an ERROR engine whose history already
    contains a rejection."""
    engine = AgentStateEngine()
    engine.process_query("")

    with pytest.raises(EngineReuseError):
        engine.process_query(RAG_QUERY)

    assert engine.state is AgentState.ERROR


def test_context_buffer_exposes_a_copy_not_the_live_internal_dict():
    """[ROBUST] Fails if the property returns the internal dict: a caller
    mutating the returned value would rewrite the engine's record and
    result.intent could disagree with context_buffer["intent"]."""
    engine = AgentStateEngine()
    engine.process_query(RAG_QUERY)

    leaked = engine.context_buffer
    leaked["intent"] = "tampered"

    assert engine.context_buffer["intent"] == RoutingIntent.ANALYTICAL_RAG.value


def test_two_engines_do_not_share_context_or_history():
    """[ROBUST] Fails if context/history live on the class (e.g. a mutable
    default arg): engine B would inherit engine A's history and intent."""
    engine_a = AgentStateEngine()
    engine_b = AgentStateEngine()

    engine_a.process_query(GRAPH_QUERY)
    engine_b.process_query(DIRECT_QUERY)

    assert engine_b.context_buffer["intent"] == RoutingIntent.DIRECT_EXECUTION.value
    assert len(engine_b.history) == 3
    assert all(t.to_state is not AgentState.GRAPH_TRAVERSAL for t in engine_b.history)


class _AlwaysDirectClassifier(IntentClassifier):
    """Test double: forces DIRECT_EXECUTION regardless of keywords.

    Hand-written, not a mock, so it goes through the real Classification type
    and the real validation contract.
    """

    def explain(self, query: str) -> Classification:
        self._token_set(query)  # same validation contract as the base class
        return Classification(
            intent=RoutingIntent.DIRECT_EXECUTION,
            graph_score=0,
            rag_score=0,
            unique_token_count=0,
            matched_graph_keywords=(),
            matched_rag_keywords=(),
        )


def test_injected_classifier_controls_route_selection():
    """[ROBUST] Fails if AgentStateEngine ignores the constructor argument
    and builds its own IntentClassifier: this graph-heavy query would route
    to GRAPH_TRAVERSAL instead of DIRECT_EXECUTION."""
    engine = AgentStateEngine(classifier=_AlwaysDirectClassifier())
    result = engine.process_query(GRAPH_QUERY)

    assert result.intent is RoutingIntent.DIRECT_EXECUTION
    assert result.state_path == (
        AgentState.INIT,
        AgentState.ROUTING,
        AgentState.SYNTHESIS,
        AgentState.TERMINATED,
    )


# ===========================================================================
# Guards and public surface
# ===========================================================================


def test_illegal_transition_is_rejected_and_state_is_unchanged():
    """[GUARD] Fails if the transition table is bypassed (e.g. `_transition`
    becomes a bare assignment), letting INIT jump straight to TERMINATED and
    produce a "successful" result with no routing at all.

    Uses a private member on purpose: the guard is the invariant, and widening
    the public API purely for testability would be worse.
    """
    engine = AgentStateEngine()
    with pytest.raises(IllegalTransitionError):
        engine._transition(AgentState.TERMINATED, "skip everything")

    assert engine.state is AgentState.INIT
    assert engine.history == ()


@pytest.mark.parametrize(
    "current_state, next_state",
    [
        (AgentState.INIT, AgentState.TERMINATED),
        (AgentState.ROUTING, AgentState.TERMINATED),
        (AgentState.SYNTHESIS, AgentState.ROUTING),
        (AgentState.ERROR, AgentState.ROUTING),
        (AgentState.TERMINATED, AgentState.ROUTING),
    ],
)
def test_representative_illegal_transitions_are_rejected(current_state, next_state):
    """[GUARD] Fails if any of these five forbidden edges gets added to
    `_ALLOWED_TRANSITIONS`. Representative sample, not all 49 invalid pairs —
    every disallowed edge goes through the same membership check, so a broken
    check fails all five and a broken table fails exactly one."""
    engine = AgentStateEngine()
    engine._state = current_state

    with pytest.raises(IllegalTransitionError):
        engine._transition(next_state, "invalid")


def test_as_dict_is_json_serialisable_and_uses_primitive_values():
    """[ROBUST] Fails if enum members leak into the serialized form
    (TypeError on json.dumps) or if the state path serializes out of order."""
    result = AgentStateEngine().process_query(GRAPH_QUERY)
    payload = result.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["status"] == "success"
    assert payload["intent"] == "graph_traversal"
    assert payload["state_path"] == [
        "init",
        "routing",
        "graph_traversal",
        "synthesis",
        "terminated",
    ]


# ===========================================================================
# Property specs (Hypothesis isn't a stdlib dep, so these are executable
# specifications rather than runnable tests)
# ===========================================================================
#
# PROPERTY (classifier totality + determinism):
#   For any s in st.text(min_size=1).filter(lambda s: s.strip()),
#   classify(s) is deterministic and is exactly one member of RoutingIntent.
#
# PROPERTY (decision boundary soundness):
#   For any query built from st.lists(st.sampled_from(sorted(GRAPH_KEYWORDS |
#   RAG_KEYWORDS)), min_size=1, max_size=12) joined by spaces:
#       graph_score > rag_score             implies GRAPH_TRAVERSAL
#       rag_score > 0 or |tokens| > 5       implies not GRAPH_TRAVERSAL
#
# PROPERTY (custom vocabulary closure):
#   For any iterable of str entries (blanks and mixed case allowed), the
#   resulting frozenset contains only non-empty lowercase strings, and
#   classify(k) for any k in the frozenset matches the registered intent.
#
# PROPERTY (state machine totality):
#   For any valid q: status is SUCCESS, path starts INIT, ends TERMINATED,
#   contains SYNTHESIS. Direct route additionally excludes both retrieval
#   states.
#
# PROPERTY (error totality):
#   For any q where `not isinstance(q, str) or not q.strip()`: status is
#   ERROR and path is exactly (INIT, ERROR).

# Not tested (and why):
# - Retrieval quality: no embedding model, BM25 index, RRF fusion, or graph
#   exists here. The payloads are placeholders; asserting on their content
#   beyond the marker and the query echo would be asserting on a string literal.
# - Classifier accuracy on real language: no labelled corpus. The heuristic's
#   precision/recall against true intents is unknown and unmeasurable here.
# - Thread safety: the engine is single-use and unsynchronised. Two threads
#   calling process_query on the same instance race on `_consumed`. Out of
#   scope, and stated as such rather than papered over.
# - The full 7x7 illegal-transition matrix: only 5 of the 49 disallowed pairs
#   are checked. They all pass through the same membership check.
# - Cross-process hash-seed determinism: in-process repetition cannot observe
#   per-process hash salting. A subprocess test with two PYTHONHASHSEED values
#   would close this. Not included — it adds subprocess and env manipulation
#   to a stdlib-only educational module, and the docstring records the gap.
# - Unicode normalization (NFC/NFD) and CJK segmentation: str.split() treats
#   "café" and "cafe\u0301" as one token and doesn't split CJK text. Fixing
#   this changes the routing contract, so it must be a documented decision,
#   not a quiet test edit.
# - Performance benchmarks: the complexity claim is derived analytically. A
#   microbenchmark on an O(Q) function measures pytest overhead, not the code.
# - Transition/history serialization round-trip: `as_dict()` covers the public
#   result. There's no persistence layer, so history round-tripping is untested.
# - Hypothesis properties above: dependency unavailable in a stdlib-only
#   submission, so they're specifications, not executed checks.
# - Documentation integrity: no test can detect a future developer editing a
#   comment from "no BM25 retrieval" to "BM25 retrieval" without a behavior
#   change. Review-time concern, not test-time.
