"""Tests for Day 44 — Graph RAG.

Grouped by what's being tested:
* TestKnowledgeGraph — the graph data structure
* TestExtractor      — turning text into triplets
* TestBFS            — traversing the graph from query seeds
* TestContext        — formatting the result
* TestIngestion      — the orchestrator's document ingestion path

The suite is intentionally a bit on the larger side. Most of the interesting
behavior here lives in edge cases (cycles, back-edges, hop boundaries, empty
results), and it's easy to break those with a "cleanup" commit.
"""

from __future__ import annotations

import pytest
from day44_graph_rag import (
    EntityRelationExtractor,
    GraphRAGOrchestrator,
    KnowledgeGraph,
)


# ===========================================================================
# KnowledgeGraph
# ===========================================================================
class TestKnowledgeGraph:
    def test_endpoints_are_normalized(self) -> None:
        kg = KnowledgeGraph()
        assert kg.add_relation("Python", "uses", "Autograd") is True
        assert set(kg.nodes) == {"python", "autograd"}
        assert ("python", "USES", "autograd") in kg.edges

    def test_edges_are_directed(self) -> None:
        kg = KnowledgeGraph()
        kg.add_relation("a", "USES", "b")
        assert kg.get_neighbors("a") == [("b", "USES")]
        assert kg.get_neighbors("b") == []  # no reverse edge

    def test_duplicate_edges_are_ignored(self) -> None:
        kg = KnowledgeGraph()
        assert kg.add_relation("a", "uses", "b") is True
        assert kg.add_relation("A", "USES", "B") is False
        assert kg.edges == (("a", "USES", "b"),)

    def test_self_loops_and_cycles_are_allowed(self) -> None:
        kg = KnowledgeGraph()
        kg.add_relation("a", "USES", "a")
        kg.add_relation("a", "USES", "b")
        kg.add_relation("b", "USES", "a")
        assert ("a", "USES", "a") in kg.edges
        assert ("b", "USES", "a") in kg.edges

    def test_empty_inputs_are_noops(self) -> None:
        kg = KnowledgeGraph()
        assert kg.add_entity("") is False
        assert kg.add_relation("", "USES", "b") is False
        assert kg.add_relation("a", "", "b") is False
        assert kg.edges == ()

    def test_wrong_types_raise(self) -> None:
        kg = KnowledgeGraph()
        with pytest.raises(TypeError):
            kg.add_entity(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            kg.add_relation("a", 5, "b")  # type: ignore[arg-type]

    def test_nodes_come_back_sorted(self) -> None:
        kg = KnowledgeGraph()
        for n in ["z", "a", "m", "b"]:
            kg.add_entity(n)
        assert kg.nodes == ("a", "b", "m", "z")

    def test_first_seen_casing_wins_for_display(self) -> None:
        kg = KnowledgeGraph()
        kg.add_entity("GPU")
        kg.add_entity("gpu")  # same node, second insertion ignored for display
        assert kg.display_name("GPU") == "GPU"
        assert kg.display_name("gpu") == "GPU"
        assert kg.display_name("GPU ") == "GPU"  # strip on lookup

    def test_unknown_entity_display_falls_back_to_lowercase(self) -> None:
        kg = KnowledgeGraph()
        assert kg.display_name("Unknown") == "unknown"


# ===========================================================================
# Extractor
# ===========================================================================
class TestExtractor:
    def setup_method(self) -> None:
        self.ex = EntityRelationExtractor()

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Python uses autograd.", ("python", "USES", "autograd")),
            ("Node implements stream.", ("node", "IMPLEMENTS", "stream")),
            ("Transformer builds attention.", ("transformer", "BUILDS", "attention")),
            ("Model requires gpu.", ("model", "REQUIRES", "gpu")),
            ("Express extends router.", ("express", "EXTENDS", "router")),
            ("Python is a language.", ("python", "IS_A", "language")),
            ("Python is an engine.", ("python", "IS_A", "engine")),
            ("Autograd depends on backpropagation.",
             ("autograd", "DEPENDS_ON", "backpropagation")),
        ],
    )
    def test_all_seven_relation_types(self, text: str, expected: tuple) -> None:
        assert expected in self.ex.extract_triplets(text)

    def test_default_returns_lowercased_entities(self) -> None:
        assert self.ex.extract_triplets("PYTHON uses AUToGRAD.") == [
            ("python", "USES", "autograd")
        ]

    def test_preserve_case_option(self) -> None:
        assert self.ex.extract_triplets(
            "GPU uses VRAM.", preserve_case=True
        ) == [("GPU", "USES", "VRAM")]

    def test_single_letter_entities_are_ok(self) -> None:
        # Regression: "a" used to be in the stopword list, which silently
        # dropped every triplet in the cycle/self-loop fixtures.
        assert self.ex.extract_triplets("A uses B.") == [("a", "USES", "b")]

    def test_stopwords_dont_become_entities(self) -> None:
        text = "Python uses autograd and depends on backpropagation."
        triplets = self.ex.extract_triplets(text)
        assert ("and", "DEPENDS_ON", "backpropagation") not in triplets
        assert ("python", "USES", "autograd") in triplets

    def test_pronouns_dont_become_entities(self) -> None:
        assert self.ex.extract_triplets("This is a test.") == []
        assert self.ex.extract_triplets("It is a thing.") == []

    def test_empty_and_no_match_inputs(self) -> None:
        assert self.ex.extract_triplets("") == []
        assert self.ex.extract_triplets("   \n\t  ") == []
        assert self.ex.extract_triplets("The fox jumps.") == []

    def test_non_string_raises(self) -> None:
        with pytest.raises(TypeError):
            self.ex.extract_triplets(None)  # type: ignore[arg-type]

    def test_repeated_triplets_collapse(self) -> None:
        text = "Python uses autograd. Python uses autograd."
        triplets = self.ex.extract_triplets(text)
        assert triplets.count(("python", "USES", "autograd")) == 1

    def test_custom_patterns_are_supported(self) -> None:
        custom = EntityRelationExtractor(
            patterns=[(r"\b(\w+)\s+loves\s+(\w+)\b", "LOVES")]
        )
        assert custom.extract_triplets("Alice loves Bob.") == [
            ("alice", "LOVES", "bob")
        ]

    def test_bad_pattern_specs_raise(self) -> None:
        with pytest.raises(TypeError):
            EntityRelationExtractor(patterns=[("only",)])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            EntityRelationExtractor(patterns=[("(", "X")])  # bad regex
        with pytest.raises(ValueError):
            EntityRelationExtractor(patterns=[])  # empty list


# ===========================================================================
# BFS / Orchestrator
# ===========================================================================
def _chain() -> GraphRAGOrchestrator:
    """The canonical three-hop chain used by the traversal tests:

        Transformer --BUILDS--> attention --USES--> query --DEPENDS_ON--> vector
    """
    o = GraphRAGOrchestrator()
    o.ingest_document(
        "Transformer builds attention. "
        "Attention uses query. "
        "Query depends on vector."
    )
    return o


class TestBFS:
    def test_zero_hops_returns_nothing(self) -> None:
        assert _chain().traverse_subgraph(["transformer"], max_hops=0) == []

    def test_one_hop_returns_direct_edges(self) -> None:
        edges = _chain().traverse_subgraph(["transformer"], max_hops=1)
        assert edges == [("transformer", "BUILDS", "attention")]

    def test_two_hops_excludes_the_third(self) -> None:
        edges = _chain().traverse_subgraph(["transformer"], max_hops=2)
        assert ("transformer", "BUILDS", "attention") in edges
        assert ("attention", "USES", "query") in edges
        assert ("query", "DEPENDS_ON", "vector") not in edges

    def test_three_hops_reaches_the_end(self) -> None:
        edges = _chain().traverse_subgraph(["transformer"], max_hops=3)
        assert ("query", "DEPENDS_ON", "vector") in edges

    def test_traversal_follows_direction(self) -> None:
        # Starting at "query" should not reach "transformer" (that edge
        # points the other way).
        edges = _chain().traverse_subgraph(["query"], max_hops=5)
        sources = {s for s, _, _ in edges}
        assert "transformer" not in sources

    def test_multiple_seeds_are_unioned(self) -> None:
        edges = _chain().traverse_subgraph(["transformer", "query"], max_hops=1)
        assert ("transformer", "BUILDS", "attention") in edges
        assert ("query", "DEPENDS_ON", "vector") in edges

    def test_duplicate_seeds_collapse(self) -> None:
        a = _chain().traverse_subgraph(
            ["transformer", "Transformer", "TRANSFORMER"], max_hops=1
        )
        b = _chain().traverse_subgraph(["transformer"], max_hops=1)
        assert a == b

    def test_unknown_seeds_are_skipped(self) -> None:
        assert _chain().traverse_subgraph(["nope"], max_hops=2) == []
        # Mixed case: unknown dropped, valid still works.
        edges = _chain().traverse_subgraph(["nope", "transformer"], max_hops=1)
        assert edges == [("transformer", "BUILDS", "attention")]

    def test_empty_seed_list(self) -> None:
        assert _chain().traverse_subgraph([], max_hops=2) == []

    def test_cycle_terminates(self) -> None:
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses B. B uses C. C uses A.")
        edges = o.traverse_subgraph(["a"], max_hops=10)
        assert set(edges) == {
            ("a", "USES", "b"),
            ("b", "USES", "c"),
            ("c", "USES", "a"),
        }

    def test_back_edges_into_visited_nodes_are_still_collected(self) -> None:
        # A -> B, B -> C, A -> C. When we process B, C is already visited
        # (via A's shortcut), but the edge B -> C is real and belongs in
        # the sub-graph. This is the classic node-vs-edge tracking case.
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses B. B uses C. A uses C.")
        edges = o.traverse_subgraph(["a"], max_hops=2)
        assert ("a", "USES", "b") in edges
        assert ("a", "USES", "c") in edges
        assert ("b", "USES", "c") in edges

    def test_self_loop(self) -> None:
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses A.")
        assert o.traverse_subgraph(["a"], max_hops=5) == [("a", "USES", "a")]

    def test_hop_distance_is_shortest_path(self) -> None:
        # Two length-2 paths to D. At H=1 we can't reach D yet; at H=2 we
        # reach it via both parents, and both edges should be collected.
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses B. A uses C. B uses D. C uses D.")
        h1 = set(o.traverse_subgraph(["a"], max_hops=1))
        assert h1 == {("a", "USES", "b"), ("a", "USES", "c")}
        h2 = set(o.traverse_subgraph(["a"], max_hops=2))
        assert ("b", "USES", "d") in h2
        assert ("c", "USES", "d") in h2

    def test_shortcut_stops_deeper_revisits(self) -> None:
        # A -> C directly, and A -> B -> C. C is reached at depth 1 via the
        # shortcut, so B's edge to C doesn't extend the frontier — but it
        # still gets collected as an edge.
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses B. A uses C. B uses C.")
        edges = o.traverse_subgraph(["a"], max_hops=2)
        assert ("a", "USES", "c") in edges
        assert ("b", "USES", "c") in edges

    # ---- max_edges -------------------------------------------------------
    def test_max_edges_zero_returns_nothing(self) -> None:
        # Regression: this used to leak the first edge before bailing.
        o = _chain()
        assert o.traverse_subgraph(
            ["transformer"], max_hops=5, max_edges=0
        ) == []

    def test_max_edges_caps_the_output(self) -> None:
        o = GraphRAGOrchestrator()
        for i in range(50):
            o.graph.add_relation("hub", "USES", f"leaf{i}")
        edges = o.traverse_subgraph(["hub"], max_hops=1, max_edges=10)
        assert len(edges) == 10

    # ---- input validation ------------------------------------------------
    def test_negative_hops_raise(self) -> None:
        with pytest.raises(ValueError):
            _chain().traverse_subgraph(["transformer"], max_hops=-1)

    def test_bool_hops_rejected(self) -> None:
        # isinstance(True, int) is True, so we have to check explicitly.
        with pytest.raises(TypeError):
            _chain().traverse_subgraph(["transformer"], max_hops=True)  # type: ignore[arg-type]

    def test_float_hops_rejected(self) -> None:
        with pytest.raises(TypeError):
            _chain().traverse_subgraph(["transformer"], max_hops=2.5)  # type: ignore[arg-type]

    def test_bare_string_seeds_rejected(self) -> None:
        # "transformer" would otherwise iterate as characters.
        with pytest.raises(TypeError):
            _chain().traverse_subgraph("transformer", max_hops=1)  # type: ignore[arg-type]

    def test_non_string_seed_rejected(self) -> None:
        with pytest.raises(TypeError):
            _chain().traverse_subgraph([123], max_hops=1)  # type: ignore[list-item]

    def test_negative_max_edges_rejected(self) -> None:
        with pytest.raises(ValueError):
            _chain().traverse_subgraph(["transformer"], max_hops=1, max_edges=-1)


# ===========================================================================
# Context formatting
# ===========================================================================
class TestContext:
    def test_format_uses_first_seen_casing(self) -> None:
        # Input: "Transformer builds attention. Attention uses query."
        # First-seen forms: "Transformer", "attention" (lowercase!), "query".
        # Second-sentence "Attention" normalizes to the same node as
        # "attention" — and first-seen wins, so it stays lowercase.
        ctx = _chain().assemble_graph_context(["transformer"], max_hops=2)
        assert ctx == (
            "[Knowledge Graph Context]\n"
            "(Transformer) --[BUILDS]--> (attention)\n"
            "(attention) --[USES]--> (query)"
        )

    def test_empty_result_has_explicit_marker(self) -> None:
        ctx = _chain().assemble_graph_context(["missing"], max_hops=2)
        assert ctx.startswith("[Knowledge Graph Context]")
        assert "empty" in ctx.lower()

    def test_zero_hops_yields_empty_result(self) -> None:
        ctx = _chain().assemble_graph_context(["transformer"], max_hops=0)
        assert "empty" in ctx.lower()

    def test_output_is_deterministic(self) -> None:
        a = _chain().assemble_graph_context(["transformer"], max_hops=3)
        b = _chain().assemble_graph_context(["transformer"], max_hops=3)
        assert a == b

    def test_acronym_casing_survives(self) -> None:
        o = GraphRAGOrchestrator()
        o.ingest_document("GPU uses VRAM.")
        ctx = o.assemble_graph_context(["gpu"], max_hops=1)
        assert "(GPU) --[USES]--> (VRAM)" in ctx

    def test_newline_injection_cant_forge_a_fake_header(self) -> None:
        # Threat model: entity text contains "\n[Fake] ..." hoping to inject
        # a fake framing line. Defense: newlines get folded to spaces, so
        # the output stays at exactly 1 header + N edge lines.
        kg = KnowledgeGraph()
        kg.add_relation("evil\n[Fake]", "USES", "target")
        o = GraphRAGOrchestrator()
        o.graph = kg

        ctx = o.assemble_graph_context(["evil\n[fake]"], max_hops=1)
        lines = ctx.split("\n")
        assert len(lines) == 2
        assert lines[0] == "[Knowledge Graph Context]"
        # No non-header line starts with "[" — that's our framing sentinel.
        for line in lines[1:]:
            assert not line.startswith("[")
        # The brackets in the entity are data, not framing.
        assert "(evil [Fake])" in ctx

    def test_tab_injection_is_stripped(self) -> None:
        kg = KnowledgeGraph()
        kg.add_relation("evil\tX", "USES", "target")
        o = GraphRAGOrchestrator()
        o.graph = kg
        ctx = o.assemble_graph_context(["evil\tx"], max_hops=1)
        assert "\t" not in ctx


# ===========================================================================
# Ingestion
# ===========================================================================
class TestIngestion:
    def test_repeat_ingestion_is_idempotent(self) -> None:
        o = GraphRAGOrchestrator()
        assert o.ingest_document("A uses B.") == 1
        assert o.ingest_document("A uses B.") == 0

    def test_casing_variants_are_the_same_edge(self) -> None:
        o = GraphRAGOrchestrator()
        o.ingest_document("A uses B.")
        o.ingest_document("a USES b.")
        assert o.graph.edges == (("a", "USES", "b"),)

    def test_malformed_document_is_a_noop(self) -> None:
        o = GraphRAGOrchestrator()
        assert o.ingest_document("!!! ??? --- ") == 0

    def test_non_string_document_raises(self) -> None:
        o = GraphRAGOrchestrator()
        with pytest.raises(TypeError):
            o.ingest_document(None)  # type: ignore[arg-type]

    def test_display_casing_survives_ingestion(self) -> None:
        o = GraphRAGOrchestrator()
        o.ingest_document("GPU uses VRAM.")
        assert o.graph.display_name("gpu") == "GPU"
        assert o.graph.display_name("vram") == "VRAM"