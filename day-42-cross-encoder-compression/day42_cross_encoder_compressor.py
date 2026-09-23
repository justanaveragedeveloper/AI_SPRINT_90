"""
Day 42 — Cross-Encoder Re-Ranking & Contextual Compression.

This module implements the post-retrieval half of a two-stage RAG pipeline:

    hybrid retrieval (Day 41)
        → top-K candidate documents
        → re-rank them with a cross-encoder-style scorer      (this file)
        → drop sentences that don't match the query           (this file)
        → hand the pruned text to the LLM

Quickstart
----------
    from day42_cross_encoder_compressor import (
        CrossEncoderReranker, ContextualCompressionFilter,
    )

    reranker   = CrossEncoderReranker()
    compressor = ContextualCompressionFilter(reranker, threshold=0.55)

    candidates = [
        (0, 0.90, "MERN stack web app with React and Express."),
        (1, 0.70, "Deep learning transformer models with self-attention."),
    ]

    reranked = reranker.rerank("transformer self attention", candidates)
    payloads = compressor.compress_payloads("transformer self attention", reranked)

What this is — and what it isn't
--------------------------------
A real cross-encoder feeds the concatenated sequence

    [CLS] query [SEP] document

through a trained Transformer and squashes the output through a sigmoid:

    Score(Q, D) = sigmoid( W · Transformer([CLS] Q [SEP] D) )

This module is a *lexical simulator* of that idea. It has no Transformer,
no self-attention, no learned weights, and no embeddings. It scores a
query-document pair using:

    • exact-token term-frequency saturation
    • substring ("partial") token matching
    • document-length normalization
    • a numerically stable sigmoid

Substring matching catches pairs like ("cat", "concatenate") that share no
meaning — it is *lexical containment*, not semantic understanding.

How the score is computed
-------------------------
For each query token:

    exact match  →  exact_match_weight · tf / (tf + 1)
    partial      →  semantic_decay     · m  / (m  + 2)

The raw total is then divided by log2(1 + len(doc_tokens)) and squashed
through a sigmoid. With default weights (exact=2.0, partial=0.5):

    ("cat", "cat")               →  0.731   exact match
    ("cat", "concatenate")       →  ~0.54   partial only ("cat" ⊂ "concatenate")
    ("cat", "dog")               →  0.500   no overlap → sigmoid(0)
    ("",    "cat")               →  0.000   empty query → explicitly zero

Reproducibility
---------------
The module consumes no randomness, so no seed is required. Every score is
a deterministic function of the two input strings.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np

# Module logger; NullHandler keeps pytest quiet when the library is imported
# without configuring a logging system.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "Candidate",
    "RerankedCandidate",
    "CompressedPayload",
    "CrossEncoderReranker",
    "ContextualCompressionFilter",
]


# =========================================================================== #
# Types
# =========================================================================== #

#: What the re-ranker accepts: (doc_id, retrieval_score, document_text).
Candidate = tuple[int, float, str]

#: What the re-ranker emits: (doc_id, retrieval_score, ce_score, document_text).
RerankedCandidate = tuple[int, float, float, str]


class CompressedPayload(TypedDict):
    """One compressed document, ready to be placed in the LLM context.

    `original_score` and `cross_encoder_score` are carried through unchanged
    so downstream stages can log, compare, or filter on either ranking signal.
    """

    doc_id: int
    original_score: float
    cross_encoder_score: float
    compressed_text: str
    original_char_count: int
    compressed_char_count: int


# =========================================================================== #
# Tokenization & scoring primitives
# =========================================================================== #

_TOKEN_RE = re.compile(r"\w+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?]) +")


def _tokenize(text: str) -> list[str]:
    """Lowercase `text` and return `\\w+` tokens in order of appearance."""
    return _TOKEN_RE.findall(text.lower())


def _stable_sigmoid(x: float) -> float:
    """Sigmoid computed in a way that never overflows `math.exp`.

    The naive form `1 / (1 + exp(-x))` overflows when `x` is very negative.
    By using `exp(-x)` for `x >= 0` and `exp(x)` for `x < 0`, the exponent
    is always non-positive and the result is bounded in [0, 1].
    """
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _count_partial_matches(q_tok: str, d_tokens: Sequence[str]) -> int:
    """Count document tokens that share a substring with `q_tok`.

    A "partial match" means one is contained in the other:

        "cat"   in "concatenate"   → True
        "run"   in "running"       → True
        "transformer" / "transformers" → True

    This is lexical containment, not semantic similarity. It rewards
    morphological variants (run/runner) at the cost of false positives
    (cat/concatenate). The trade-off is documented in the module docstring.

    Deliberately implemented as a plain Python loop: substring containment
    isn't a NumPy numeric ufunc, so wrapping this in `np.fromiter` would
    add ceremony without any performance win.
    """
    return sum(1 for d in d_tokens if q_tok in d or d in q_tok)


def _score_query_token(
    q_tok: str,
    d_counts: Counter[str],
    d_tokens: Sequence[str],
    exact_weight: float,
    partial_weight: float,
) -> float:
    """Contribution of a single query token to the raw (pre-normalized) score.

    Exact match  →  exact_weight · tf / (tf + 1)
    Partial only →  partial_weight · m / (m + 2)
    No match     →  0.0

    The `tf / (tf + 1)` shape gives *diminishing returns*: going from 0→1
    occurrence is worth a lot; going from 10→11 is worth almost nothing.
    """
    tf = d_counts.get(q_tok, 0)
    if tf > 0:
        return exact_weight * (tf / (tf + 1.0))

    partial = _count_partial_matches(q_tok, d_tokens)
    if partial > 0:
        return partial_weight * (partial / (partial + 2.0))

    return 0.0


# =========================================================================== #
# Validation helpers
#
# These live in one place so the rest of the module stays readable, and so
# callers get clear errors that distinguish "wrong type" (TypeError) from
# "right type, bad value" (ValueError).
#
# Note: `bool` is a subclass of `int` in Python, so `isinstance(True, int)`
# is True. Every check below rejects booleans explicitly — otherwise
# `top_k=True` would silently pass integer validation.
# =========================================================================== #

def _check_str(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")


def _check_int(name: str, value: Any) -> int:
    """Return `value` as an int, rejecting bools (which subclass int)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _check_positive_int(name: str, value: Any) -> int:
    value = _check_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_number(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return `value` as a float, verifying type, finiteness, and range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _check_sequence(name: str, value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence, got {type(value).__name__}")
    return value


def _check_candidate(index: int, item: Any) -> Candidate:
    """Validate a single (doc_id, retrieval_score, document_text) tuple."""
    if not isinstance(item, tuple) or len(item) != 3:
        raise ValueError(
            f"candidate[{index}] must be a 3-tuple "
            f"(doc_id, original_score, document_text); got {item!r}"
        )
    doc_id, score, text = item
    doc_id = _check_int(f"candidate[{index}].doc_id", doc_id)
    score = _check_number(f"candidate[{index}].original_score", score)
    _check_str(f"candidate[{index}].document_text", text)
    return doc_id, score, text


def _check_reranked_candidate(index: int, item: Any) -> RerankedCandidate:
    """Validate a single 4-tuple emitted by `CrossEncoderReranker.rerank`."""
    if not isinstance(item, tuple) or len(item) != 4:
        raise ValueError(
            f"reranked_candidate[{index}] must be a 4-tuple "
            f"(doc_id, original_score, cross_encoder_score, document_text); "
            f"got {item!r}"
        )
    doc_id, orig, ce, text = item
    doc_id = _check_int(f"reranked_candidate[{index}].doc_id", doc_id)
    orig = _check_number(f"reranked_candidate[{index}].original_score", orig)
    ce = _check_number(f"reranked_candidate[{index}].cross_encoder_score", ce)
    _check_str(f"reranked_candidate[{index}].document_text", text)
    return doc_id, orig, ce, text


# =========================================================================== #
# Cross-encoder re-ranker
# =========================================================================== #

class CrossEncoderReranker:
    """Lexical simulator of a cross-encoder scoring head.

    Scores each (query, document) pair with the formula described in the
    module docstring, then re-orders a candidate list by that new score.

    Args:
        exact_match_weight: Multiplier for exact-token contributions.
        semantic_decay:     Multiplier for substring / partial contributions.

    Both weights must be finite and non-negative.
    """

    def __init__(
        self,
        exact_match_weight: float = 2.0,
        semantic_decay: float = 0.5,
    ) -> None:
        self.exact_match_weight = _check_number(
            "exact_match_weight", exact_match_weight, minimum=0.0
        )
        self.semantic_decay = _check_number(
            "semantic_decay", semantic_decay, minimum=0.0
        )

    # -- scoring --------------------------------------------------------- #

    def score_pair(self, query: str, document: str) -> float:
        """Return the simulated cross-encoder relevance score in [0, 1].

        The score is computed in five steps:

            1. Tokenize `query` and `document`.
            2. If either side has no tokens, return 0.0 — this is *not* the
               same as the neutral sigmoid(0) = 0.5, so callers can tell
               "no interaction" apart from "scored at the midpoint".
            3. Sum the per-query-token contributions (see `_score_query_token`).
            4. Divide by log2(1 + len(document_tokens)) — a document-length
               penalty so longer docs don't win by sheer volume.
            5. Squash through the numerically stable sigmoid.

        Returns:
            A float in [0.0, 1.0]. Empty or tokenless inputs return exactly
            0.0.
        """
        _check_str("query", query)
        _check_str("document", document)

        q_tokens = _tokenize(query)
        d_tokens = _tokenize(document)
        if not q_tokens or not d_tokens:
            return 0.0

        d_counts = Counter(d_tokens)
        raw = sum(
            _score_query_token(
                tok,
                d_counts,
                d_tokens,
                self.exact_match_weight,
                self.semantic_decay,
            )
            for tok in q_tokens
        )

        # length_penalty >= 1 for any non-empty document → never divides by 0.
        length_penalty = math.log2(1.0 + len(d_tokens))
        return _stable_sigmoid(raw / length_penalty)

    # -- reranking ------------------------------------------------------- #

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int = 3,
    ) -> list[RerankedCandidate]:
        """Re-score candidates and keep the best `top_k`.

        Ordering is deterministic. Ties are broken first by the original
        retrieval score (higher first), then by `doc_id` (lower first), so
        running the same query twice always yields the same list.

        Args:
            query: Raw query string.
            candidates: Sequence of `(doc_id, original_score, document_text)`.
            top_k: Maximum number of results to return (must be > 0).

        Returns:
            A new list of `(doc_id, original_score, cross_encoder_score,
            document_text)`. Empty input returns `[]`.
        """
        _check_str("query", query)
        _check_positive_int("top_k", top_k)
        _check_sequence("candidates", candidates)

        validated = [_check_candidate(i, c) for i, c in enumerate(candidates)]
        if not validated:
            return []

        scored: list[RerankedCandidate] = [
            (doc_id, orig, self.score_pair(query, doc), doc)
            for doc_id, orig, doc in validated
        ]
        scored.sort(key=lambda row: (-row[2], -row[1], row[0]))

        result = scored[:top_k]
        logger.debug(
            "rerank: %d candidates in, %d kept (top_k=%d)",
            len(validated), len(result), top_k,
        )
        return result


# =========================================================================== #
# Contextual compression
# =========================================================================== #

class ContextualCompressionFilter:
    """Sentence-level salience filter.

    Splits each document into sentences, scores each sentence with the same
    re-ranker used for the candidate pool, and keeps the sentences whose
    score is >= `threshold`. If no sentence clears the bar, the highest-
    scoring one is kept anyway so downstream context is never empty.

    Args:
        reranker:  An instantiated `CrossEncoderReranker`.
        threshold: Cut-off in [0, 1]. Inclusive lower bound (`>=`, not `>`).
    """

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        threshold: float = 0.5,
    ) -> None:
        if not isinstance(reranker, CrossEncoderReranker):
            raise TypeError(
                f"reranker must be a CrossEncoderReranker, "
                f"got {type(reranker).__name__}"
            )
        self.reranker = reranker
        self.threshold = _check_number(
            "threshold", threshold, minimum=0.0, maximum=1.0
        )

    # -- sentence utilities --------------------------------------------- #

    def split_sentences(self, text: str) -> list[str]:
        """Split `text` into ordered, non-empty sentences.

        Uses the Day 42 spec regex `(?<=[.!?]) +` — only a literal run of
        spaces triggers a split, so `"Hello.\\nWorld."` stays as one
        sentence. Whitespace-only input returns `[]`.
        """
        _check_str("text", text)
        stripped = text.strip()
        if not stripped:
            return []
        return [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]

    # -- document compression ------------------------------------------- #

    def compress_document(self, query: str, document: str) -> str:
        """Keep the sentences whose salience clears `self.threshold`.

        Sentence order is preserved — we never sort the kept sentences by
        score, because reordering would break the document's narrative flow.

        Returns:
            The kept sentences joined by single spaces, or `""` for an
            empty / whitespace-only document.
        """
        _check_str("query", query)
        _check_str("document", document)

        sentences = self.split_sentences(document)
        if not sentences:
            return ""

        # Vectorized scoring via NumPy: one call into the reranker per
        # sentence, then a single boolean mask for the threshold.
        scores = np.fromiter(
            (self.reranker.score_pair(query, s) for s in sentences),
            dtype=np.float64,
            count=len(sentences),
        )

        if (scores >= self.threshold).any():
            kept = [s for s, ok in zip(sentences, scores >= self.threshold) if ok]
        else:
            # Fallback: no sentence passed, keep the single best one.
            kept = [sentences[int(np.argmax(scores))]]

        return " ".join(kept)

    # -- payload compression -------------------------------------------- #

    def compress_payloads(
        self,
        query: str,
        reranked_candidates: Sequence[RerankedCandidate],
    ) -> list[CompressedPayload]:
        """Apply `compress_document` to every reranked candidate.

        Input order is preserved, and `doc_id`, `original_score`, and
        `cross_encoder_score` are copied through unchanged so the payload
        keeps its full provenance.
        """
        _check_str("query", query)
        _check_sequence("reranked_candidates", reranked_candidates)

        payloads: list[CompressedPayload] = []
        for i, item in enumerate(reranked_candidates):
            doc_id, orig, ce, text = _check_reranked_candidate(i, item)
            compressed = self.compress_document(query, text)
            payloads.append(
                CompressedPayload(
                    doc_id=doc_id,
                    original_score=orig,
                    cross_encoder_score=ce,
                    compressed_text=compressed,
                    original_char_count=len(text),
                    compressed_char_count=len(compressed),
                )
            )

        logger.debug(
            "compress_payloads: %d docs, %d → %d chars total",
            len(payloads),
            sum(p["original_char_count"] for p in payloads),
            sum(p["compressed_char_count"] for p in payloads),
        )
        return payloads


# =========================================================================== #
# Runnable demo
#
# `python day42_cross_encoder_compressor.py` shows the full pipeline on a
# tiny toy query, so you can see the shape of the output without wiring up
# Day 41 retrieval first.
# =========================================================================== #

if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    compressor = ContextualCompressionFilter(reranker, threshold=0.55)

    query = "gradient clipping exploding gradients"
    candidates: list[Candidate] = [
        (0, 0.62, "Photosynthesis converts light into chemical energy in plants."),
        (1, 0.55, (
            "Gradient clipping rescales gradients to prevent exploding "
            "gradients. The technique is standard in recurrent network training."
        )),
        (2, 0.71, (
            "Batch normalization stabilizes training. "
            "Unrelated: the MERN stack uses MongoDB and Express."
        )),
    ]

    reranked = reranker.rerank(query, candidates, top_k=3)
    payloads = compressor.compress_payloads(query, reranked)

    print(f"Query: {query}\n")
    for p in payloads:
        print(
            f"[doc {p['doc_id']:>2}]  "
            f"ce={p['cross_encoder_score']:.3f}  "
            f"{p['original_char_count']:>3} → {p['compressed_char_count']:>3} chars"
        )
        print(f"        {p['compressed_text']}\n")