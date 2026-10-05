"""Graph RAG, from scratch.

The idea: turn plain text into a small directed knowledge graph, then walk
that graph from a query entity to pull back structured context.

    text -> (entity, RELATION, entity) triplets -> graph -> BFS -> context string

Nothing here needs a framework. It's stdlib only on purpose — the point is
to see the mechanism, not to hide it.

Notes on honesty:
* The extractor is regex-based. It's a teaching tool, not a real NER model.
  It will miss multi-word entities, aliases, negation, and everything else
  that makes language hard. See EntityRelationExtractor for the full list.
* No randomness anywhere, so no seeds needed. Output order is deterministic.

Rough complexity:
* Ingest:   O(T) over the text length (for the built-in patterns).
* BFS:      O(V_sub + E_sub) over the sub-graph we actually touch.
* Memory:   O(V + E) for the graph itself.
* Worst-case reachable nodes: O(b**H) with branching factor b and depth H.
  That's an upper bound — real graphs share nodes and stay much smaller.

Python 3.10+ (uses `X | Y` type hints and PEP 585 built-ins).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TypeAlias

logger = logging.getLogger(__name__)

__all__ = [
    "EntityRelationExtractor",
    "GraphRAGOrchestrator",
    "KnowledgeGraph",
    "Triplet",
]

# (source, RELATION, target) — entities lowercased, relations uppercased.
Triplet: TypeAlias = tuple[str, str, str]

# Each pattern is (regex, relation_spec). relation_spec is either a literal
# label like "IS_A" or an int pointing at the capture group that holds the
# relation verb (e.g. group 1 in "X uses Y").
PatternSpec: TypeAlias = tuple[str, "str | int"]

# Words that slip through the permissive \w+ patterns but aren't entities.
# Note: "a" and "an" are deliberately NOT here. The test fixtures use
# single letters ("A uses B"), and those are legitimate entities under the
# documented contract. Filtering "a" broke the cycle tests.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "or", "but", "so", "if", "then", "than",
    "it", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "he", "she", "they", "we", "you",
})


def _is_usable_entity(lowercased: str) -> bool:
    """True if this token (already lowered/stripped) can be a graph node."""
    return bool(lowercased) and lowercased not in _STOPWORDS


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------
class KnowledgeGraph:
    """A directed graph G = (V, E, R) of entities and their relations.

    The representation is a plain adjacency dict:

        _adj["python"] == [("autograd", "USES"), ("pytest", "USES"), ...]

    Conventions:
    * Entities are lowercased and stripped on the way in. That means
      "GPU", "gpu", and " Gpu " are the same node.
    * Relations are uppercased and stripped.
    * Edges are directed. Adding a -> b does not add b -> a.
    * Adding the same edge twice is a no-op; the first one wins.
    * Self-loops and cycles are fine. BFS is what keeps us from looping.

    Display casing
    --------------
    When an entity is first added, we remember the exact casing the caller
    used. That way "GPU uses VRAM" prints as "(GPU) --[USES]--> (VRAM)"
    instead of the uglier all-lowercase form, without changing node identity.
    Later insertions with different casing don't overwrite the first one.
    """

    def __init__(self) -> None:
        self._adj: dict[str, list[tuple[str, str]]] = {}
        self._nodes: set[str] = set()
        self._edges: list[Triplet] = []
        self._edge_set: set[Triplet] = set()          # for O(1) duplicate check
        self._display: dict[str, str] = {}            # normalized -> first-seen

    # -- normalization helpers ---------------------------------------------
    @staticmethod
    def _norm_entity(entity: str) -> str:
        if not isinstance(entity, str):
            raise TypeError(f"entity must be str, got {type(entity).__name__}")
        return entity.strip().lower()

    @staticmethod
    def _norm_relation(relation: str) -> str:
        if not isinstance(relation, str):
            raise TypeError(
                f"relation must be str, got {type(relation).__name__}"
            )
        return relation.strip().upper()

    # -- writing to the graph ---------------------------------------------
    def add_entity(self, entity: str) -> bool:
        """Add a node. Returns True if it was new, False if it already existed."""
        original = entity.strip() if isinstance(entity, str) else ""
        norm = self._norm_entity(entity)
        if not norm or norm in self._nodes:
            return False
        self._nodes.add(norm)
        self._adj.setdefault(norm, [])
        self._display[norm] = original
        return True

    def add_relation(self, source: str, relation: str, target: str) -> bool:
        """Add a directed edge. Returns True if it was new.

        Endpoints are added on the fly, so callers don't have to remember
        to register nodes first.
        """
        src = self._norm_entity(source)
        rel = self._norm_relation(relation)
        tgt = self._norm_entity(target)
        if not (src and rel and tgt):
            return False

        self.add_entity(source)
        self.add_entity(target)

        edge: Triplet = (src, rel, tgt)
        if edge in self._edge_set:
            return False

        self._edge_set.add(edge)
        self._edges.append(edge)
        self._adj[src].append((tgt, rel))
        return True

    # -- reading from the graph -------------------------------------------
    def get_neighbors(self, entity: str) -> list[tuple[str, str]]:
        """Outgoing edges as [(neighbor, RELATION), ...], in insertion order.

        Returns a copy so the caller can't accidentally mutate the graph.
        """
        return list(self._adj.get(self._norm_entity(entity), []))

    def display_name(self, entity: str) -> str:
        """The original casing for this entity, or its lowercase form if unknown."""
        norm = self._norm_entity(entity)
        return self._display.get(norm, norm)

    @property
    def nodes(self) -> tuple[str, ...]:
        """All nodes, sorted. Sorting costs O(V log V) but gives us a stable
        iteration order for tests and for user-visible output."""
        return tuple(sorted(self._nodes))

    @property
    def edges(self) -> tuple[Triplet, ...]:
        """All edges, in the order they were added."""
        return tuple(self._edges)

    def __contains__(self, entity: object) -> bool:
        return isinstance(entity, str) and entity.strip().lower() in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"KnowledgeGraph(|V|={len(self._nodes)}, |E|={len(self._edges)})"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class EntityRelationExtractor:
    """Pull (source, RELATION, target) triplets out of text with regex.

    What it actually handles:

        X uses Y            X implements Y       X builds Y
        X requires Y        X extends Y          X is a Y
        X depends on Y

    It's deterministic: same input, same output, same order. No guessing.

    What it does NOT do — and this matters if you're tempted to trust it:

    * Multi-word entities. "New York uses Python" becomes ("york", "USES",
      "python"). Yes, really.
    * Aliases. "JS" and "JavaScript" are two different nodes.
    * Negation. "X does not use Y" still produces X --USES--> Y.
    * Coreference. "It depends on Y" gives you an entity literally named "it"
      (well, it would, except "it" is on the stopword list).
    * Ambiguity. "Apple" is one node whether you meant the company or the fruit.

    The purpose is to demonstrate the *shape* of Graph RAG: extract, build,
    traverse. Swapping this class for a real NER model later is the point.
    """

    DEFAULT_PATTERNS: tuple[PatternSpec, ...] = (
        # Verb-driven: source, verb (group 1 = relation), target.
        (r"\b(\w+)\s+(uses|implements|builds|requires|extends)\s+(\w+)\b", 1),
        # Fixed-label patterns.
        (r"\b(\w+)\s+is\s+an?\s+(\w+)\b", "IS_A"),
        (r"\b(\w+)\s+depends\s+on\s+(\w+)\b", "DEPENDS_ON"),
    )

    def __init__(self, patterns: Sequence[PatternSpec] | None = None) -> None:
        specs = tuple(patterns) if patterns is not None else self.DEFAULT_PATTERNS
        if not specs:
            raise ValueError("patterns must be a non-empty sequence")

        compiled: list[tuple[re.Pattern[str], str | int]] = []
        for spec in specs:
            if (
                not isinstance(spec, tuple)
                or len(spec) != 2
                or not isinstance(spec[0], str)
                or not isinstance(spec[1], (str, int))
            ):
                raise TypeError(
                    f"each pattern must be (str, str|int), got {spec!r}"
                )
            try:
                # IGNORECASE here means we don't have to pre-lowercase the
                # text, which lets us preserve original casing when asked.
                compiled.append((re.compile(spec[0], re.IGNORECASE), spec[1]))
            except re.error as exc:
                raise ValueError(f"invalid regex {spec[0]!r}: {exc}") from exc

        self._patterns = tuple(compiled)

    # -- internals ---------------------------------------------------------
    def _extract_raw(self, text: str) -> list[Triplet]:
        """Same as extract_triplets but keeps the input's casing.

        Dedup is keyed on the normalized form, so "Python uses X" and
        "python uses x" in the same document collapse into one entry.
        Whichever variant we saw first wins for display.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return []

        out: list[Triplet] = []
        seen: set[Triplet] = set()

        for pattern, rel_spec in self._patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) < 2:
                    continue

                src_raw = groups[0].strip()
                tgt_raw = groups[-1].strip()

                if isinstance(rel_spec, int):
                    if not 0 <= rel_spec < len(groups):
                        continue
                    rel_raw = groups[rel_spec].strip()
                else:
                    rel_raw = rel_spec.strip()

                src_lc = src_raw.lower()
                tgt_lc = tgt_raw.lower()
                rel = rel_raw.upper()

                # Reject tokens that matched the pattern but aren't entities.
                if not (_is_usable_entity(src_lc) and _is_usable_entity(tgt_lc)):
                    continue
                if not rel:
                    continue

                key: Triplet = (src_lc, rel, tgt_lc)
                if key in seen:
                    continue
                seen.add(key)
                out.append((src_raw, rel, tgt_raw))

        return out

    # -- public API --------------------------------------------------------
    def extract_triplets(
        self, text: str, *, preserve_case: bool = False
    ) -> list[Triplet]:
        """Extract triplets from ``text``.

        By default entities are lowercased — that's the normalized form the
        graph expects. Pass ``preserve_case=True`` to keep the original
        casing (useful when you want the graph to remember "GPU" as "GPU"
        for display purposes). Relations are always uppercase either way.
        """
        raw = self._extract_raw(text)
        if preserve_case:
            return raw
        return [(s.lower(), rel, t.lower()) for s, rel, t in raw]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class GraphRAGOrchestrator:
    """The pipeline: ingest documents, walk the graph, format context.

    BFS semantics
    -------------
    Starting from a set of query entities, we walk outgoing edges up to
    ``max_hops``. The reachable set is:

        { u : dist(seed, u) <= max_hops for some seed }

    where ``dist`` is the shortest directed path length (in edges). Because
    BFS visits nodes level-by-level, the first time we see a node we've
    found its shortest distance from the seed set.

    A subtlety worth knowing: we track visited *nodes* and visited *edges*
    separately. If we only tracked nodes, we'd miss edges like b -> c in
    this graph:

        a -> b -> c
        a ------> c

    When we process b, c is already visited (via the shortcut), but the
    edge b -> c is still real and belongs in the sub-graph. So we always
    collect edges when we see them, and only skip *enqueueing* a node the
    second time we reach it.

    About ``max_edges``
    -------------------
    Optional safety valve for hub nodes. A single entity with a million
    outgoing edges would otherwise blow up the context. When you set
    ``max_edges``, what you get back is *bounded* graph retrieval — not the
    complete SubGraph(Q, H) — because we stop as soon as we hit the cap.
    """

    def __init__(
        self, extractor: EntityRelationExtractor | None = None
    ) -> None:
        self.graph = KnowledgeGraph()
        self.extractor = (
            extractor if extractor is not None else EntityRelationExtractor()
        )

    # -- ingest ------------------------------------------------------------
    def ingest_document(self, text: str) -> int:
        """Extract triplets from ``text`` and merge them into the graph.

        Returns the number of *new* edges (duplicates don't count).

        We pass preserve_case=True so that the graph's display layer can
        show "GPU" instead of "gpu" for entities that were spelled that way.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        added = 0
        for src, rel, tgt in self.extractor.extract_triplets(
            text, preserve_case=True
        ):
            if self.graph.add_relation(src, rel, tgt):
                added += 1

        logger.info(
            "ingest_document: +%d edges |V|=%d |E|=%d",
            added, len(self.graph.nodes), len(self.graph.edges),
        )
        return added

    # -- traversal ---------------------------------------------------------
    def traverse_subgraph(
        self,
        query_entities: Iterable[str],
        max_hops: int = 2,
        max_edges: int | None = None,
    ) -> list[Triplet]:
        """BFS from the seed entities. Returns edges reachable within max_hops.

        Seeds that don't exist in the graph are skipped silently — that's
        usually what you want when the user's query mentions entities we
        never extracted. Duplicate seeds are collapsed.

        Raises TypeError for non-int max_hops (including bool — Python's
        isinstance(True, int) is True, which is a footgun), and ValueError
        for negative hops.
        """
        # Guardrails first. Anything obviously wrong, fail loud and early.
        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise TypeError(
                f"max_hops must be int, got {type(max_hops).__name__}"
            )
        if max_hops < 0:
            raise ValueError(f"max_hops must be >= 0, got {max_hops}")
        if max_edges is not None and (
            isinstance(max_edges, bool) or not isinstance(max_edges, int)
        ):
            raise TypeError("max_edges must be int or None")
        if max_edges is not None and max_edges < 0:
            raise ValueError(f"max_edges must be >= 0, got {max_edges}")
        if isinstance(query_entities, (str, bytes)):
            # A bare string would be iterated character-by-character. Reject it.
            raise TypeError("query_entities must be an iterable of strings")

        if max_edges == 0:
            return []  # nothing to do

        try:
            seeds = list(query_entities)
        except TypeError as exc:
            raise TypeError("query_entities must be iterable") from exc

        visited_nodes: set[str] = set()
        visited_edges: set[Triplet] = set()
        collected: list[Triplet] = []
        queue: deque[tuple[str, int]] = deque()

        # Seed nodes sit at depth 0.
        for raw in seeds:
            if not isinstance(raw, str):
                raise TypeError(f"seed must be str, got {type(raw).__name__}")
            norm = raw.strip().lower()
            if not norm or norm not in self.graph or norm in visited_nodes:
                continue
            visited_nodes.add(norm)
            queue.append((norm, 0))

        # Main BFS loop.
        while queue:
            curr, depth = queue.popleft()

            # Nodes at the hop boundary get visited but don't expand.
            if depth >= max_hops:
                continue

            for neighbor, rel in self.graph.get_neighbors(curr):
                edge: Triplet = (curr, rel, neighbor)

                if edge not in visited_edges:
                    visited_edges.add(edge)
                    collected.append(edge)
                    if max_edges is not None and len(collected) >= max_edges:
                        logger.warning(
                            "traverse_subgraph: hit max_edges=%d cap "
                            "(high-degree node?)",
                            max_edges,
                        )
                        return collected

                # Enqueue only if we haven't seen this node before.
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        logger.debug(
            "traverse_subgraph: seeds=%d max_hops=%d visited=%d edges=%d",
            len(seeds), max_hops, len(visited_nodes), len(collected),
        )
        return collected

    # -- formatting --------------------------------------------------------
    def assemble_graph_context(
        self,
        query_entities: Iterable[str],
        max_hops: int = 2,
        max_edges: int | None = None,
    ) -> str:
        """Format the reachable sub-graph as a block of text.

        Output looks like:

            [Knowledge Graph Context]
            (Python) --[USES]--> (Autograd)

        Security note: this output is *retrieved data*, not instructions.
        A malicious document could try to smuggle a newline into an entity
        name and forge a fake header line — we strip newlines, carriage
        returns, and tabs from every field to prevent that. That said, this
        is one layer of defense. The prompt that consumes this text should
        still wrap it in clear delimiters (e.g. <retrieved_graph_context>)
        and tell the model to treat the contents as data, not commands.
        """
        triplets = self.traverse_subgraph(
            query_entities, max_hops=max_hops, max_edges=max_edges
        )

        if not triplets:
            return (
                "[Knowledge Graph Context]\n"
                "(empty: no matching entities or relations)"
            )

        def _clean(s: str) -> str:
            return s.replace("\n", " ").replace("\r", " ").replace("\t", " ")

        lines = ["[Knowledge Graph Context]"]
        for src, rel, tgt in triplets:
            s = _clean(self.graph.display_name(src))
            t = _clean(self.graph.display_name(tgt))
            r = _clean(rel)
            lines.append(f"({s}) --[{r}]--> ({t})")

        return "\n".join(lines)