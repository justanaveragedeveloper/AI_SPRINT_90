"""Pytest suite for Day 42 — cross-encoder re-ranking & contextual compression."""
from __future__ import annotations

import math

import pytest
from day42_cross_encoder_compressor import (
    ContextualCompressionFilter,
    CrossEncoderReranker,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@pytest.fixture()
def compressor(reranker: CrossEncoderReranker) -> ContextualCompressionFilter:
    return ContextualCompressionFilter(reranker, threshold=0.55)


# --------------------------------------------------------------------------- #
# 1. Cross-encoder scoring
# --------------------------------------------------------------------------- #

class TestScorePairBehaviour:
    def test_relevant_beats_irrelevant(self, reranker: CrossEncoderReranker) -> None:
        q = "autograd backpropagation engine"
        relevant = (
            "This document explains the autograd engine and "
            "backpropagation mechanics in Python."
        )
        irrelevant = "MongoDB Express React and Node form the MERN web stack."

        assert reranker.score_pair(q, relevant) > reranker.score_pair(q, irrelevant)

    @pytest.mark.parametrize(
        "q, d",
        [
            ("cat", "cat"),
            ("cat", "dog"),
            ("a b c", "c b a"),
            ("transformer", "transformers transformer transforming"),
            ("gradient clipping", "Gradient clipping prevents exploding gradients."),
        ],
    )
    def test_scores_are_finite_and_in_unit_interval(
        self, reranker: CrossEncoderReranker, q: str, d: str
    ) -> None:
        s = reranker.score_pair(q, d)
        assert 0.0 <= s <= 1.0
        assert math.isfinite(s)

    def test_exact_match_formula_single_token(self) -> None:
        """Directly verify the closed-form score for the simplest case.

        With ``exact_match_weight=2``, ``semantic_decay=0`` and a
        single-token query / document:

            tf = 1
            raw = 2 * 1 / (1 + 1) = 1.0
            length_penalty = log2(1 + 1) = 1.0
            normalized = 1.0
            score = sigmoid(1.0)
        """
        r = CrossEncoderReranker(exact_match_weight=2.0, semantic_decay=0.0)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert r.score_pair("cat", "cat") == pytest.approx(expected)

    def test_exact_match_beats_partial_match(self) -> None:
        r = CrossEncoderReranker(exact_match_weight=2.0, semantic_decay=0.5)
        exact = r.score_pair("running", "running fast today")
        partial = r.score_pair("running", "run fast today")
        assert exact > partial

    def test_partial_match_is_lexical_not_semantic(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """Substring overlap is *containment*, not meaning.

        ``"cat"`` is a substring of ``"concatenate"``, so it produces a
        strictly higher score than a token that shares nothing — even
        though semantically ``cat`` and ``concatenate`` are unrelated.
        """
        partial = reranker.score_pair("cat", "concatenate")
        unrelated = reranker.score_pair("cat", "banana")
        assert partial > unrelated

    def test_tf_saturation_at_constant_length(self, reranker: CrossEncoderReranker) -> None:
        """TF contributes with diminishing marginal returns when length is fixed."""
        s1 = reranker.score_pair("cat", "cat filler filler")
        s2 = reranker.score_pair("cat", "cat cat filler")
        s3 = reranker.score_pair("cat", "cat cat cat")

        assert s1 < s2 < s3
        # Marginal gain must shrink — that is exactly what tf/(tf+1) encodes.
        assert (s2 - s1) > (s3 - s2)

    def test_partial_substring_matching_helps(self) -> None:
        r = CrossEncoderReranker(exact_match_weight=2.0, semantic_decay=0.5)
        with_partial = r.score_pair("transformer", "transformers are useful")
        without = r.score_pair("transformer", "completely unrelated words")
        assert with_partial > without

    @pytest.mark.parametrize(
        "q, d",
        [
            ("", "some document"),
            ("some query", ""),
            ("", ""),
            ("   ", "some document"),
            ("some query", "   \t\n"),
            ("!!!", "???"),          # no \w tokens at all
        ],
    )
    def test_empty_or_tokenless_returns_zero(
        self, reranker: CrossEncoderReranker, q: str, d: str
    ) -> None:
        assert reranker.score_pair(q, d) == 0.0

    def test_deterministic_repeated_calls(self, reranker: CrossEncoderReranker) -> None:
        q = "gradient clipping"
        d = "Gradient clipping prevents exploding gradients during backprop."
        assert reranker.score_pair(q, d) == reranker.score_pair(q, d)

    @pytest.mark.parametrize("bad", [None, 1, b"bytes", ["list"]])
    def test_invalid_query_type(self, reranker: CrossEncoderReranker, bad: object) -> None:
        with pytest.raises(TypeError):
            reranker.score_pair(bad, "doc")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, 1, b"bytes", ["list"]])
    def test_invalid_document_type(self, reranker: CrossEncoderReranker, bad: object) -> None:
        with pytest.raises(TypeError):
            reranker.score_pair("query", bad)  # type: ignore[arg-type]


class TestInitAndNumericalStability:
    @pytest.mark.parametrize("bad", [None, "2.0", [2.0], True, False])
    def test_exact_match_weight_type(self, bad: object) -> None:
        with pytest.raises(TypeError):
            CrossEncoderReranker(exact_match_weight=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, "0.5", [0.5], True])
    def test_semantic_decay_type(self, bad: object) -> None:
        with pytest.raises(TypeError):
            CrossEncoderReranker(semantic_decay=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad", [-1.0, -1e-9, float("inf"), float("-inf"), float("nan")]
    )
    def test_weight_values(self, bad: float) -> None:
        with pytest.raises(ValueError):
            CrossEncoderReranker(exact_match_weight=bad)
        with pytest.raises(ValueError):
            CrossEncoderReranker(semantic_decay=bad)

    def test_extreme_large_weights_sigmoid_stability(self) -> None:
        huge = CrossEncoderReranker(exact_match_weight=1e12, semantic_decay=1e12)
        s = huge.score_pair("cat", "cat")
        assert math.isfinite(s)
        assert s == pytest.approx(1.0)

    def test_zero_weights_collapse_to_neutral_sigmoid(self) -> None:
        tiny = CrossEncoderReranker(exact_match_weight=0.0, semantic_decay=0.0)
        # No contribution → sigmoid(0) = 0.5 for a non-empty pair.
        assert tiny.score_pair("cat", "cat") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 2. Re-ranking
# --------------------------------------------------------------------------- #

class TestRerank:
    def test_reorders_by_cross_encoder_score(self, reranker: CrossEncoderReranker) -> None:
        q = "transformer self attention"
        candidates = [
            (0, 0.90, "MERN stack web application with React and Express."),
            (1, 0.70, "Deep learning transformer models with multi-head self attention."),
            (2, 0.85, "Python autograd differentiation engine."),
        ]
        out = reranker.rerank(q, candidates, top_k=3)
        assert out[0][0] == 1
        assert out[0][2] >= out[1][2] >= out[2][2]

    def test_preserves_original_score_and_text(self, reranker: CrossEncoderReranker) -> None:
        out = reranker.rerank("cat", [(7, 0.42, "the cat sat")], top_k=1)
        assert out[0][0] == 7
        assert out[0][1] == pytest.approx(0.42)
        assert out[0][3] == "the cat sat"

    def test_top_k_truncates(self, reranker: CrossEncoderReranker) -> None:
        cands = [(i, 0.5, f"doc {i} mentions cat") for i in range(10)]
        assert len(reranker.rerank("cat", cands, top_k=3)) == 3

    def test_empty_candidates_returns_empty(self, reranker: CrossEncoderReranker) -> None:
        assert reranker.rerank("q", [], top_k=3) == []

    def test_deterministic_tie_breaking(self) -> None:
        r = CrossEncoderReranker()
        # Identical text → identical CE score; tie-break by orig desc, id asc.
        cands = [
            (2, 0.5, "the cat sat"),
            (1, 0.5, "the cat sat"),
            (0, 0.9, "the cat sat"),
        ]
        out = r.rerank("cat", cands, top_k=3)
        assert [x[0] for x in out] == [0, 1, 2]

    @pytest.mark.parametrize("bad", ["", "not-a-list", 42, None])
    def test_malformed_candidates_container(
        self, reranker: CrossEncoderReranker, bad: object
    ) -> None:
        with pytest.raises(TypeError):
            reranker.rerank("q", bad, top_k=1)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad",
        [
            [(0, 0.5)],                    # tuple too short
            [(0, 0.5, "doc", "extra")],    # tuple too long
            [("x", 0.5, "doc")],           # doc_id wrong type
            [(0, "0.5", "doc")],           # original_score wrong type
            [(0, float("nan"), "doc")],    # non-finite original score
            [(0, float("inf"), "doc")],    # non-finite original score
            [(0, 0.5, 123)],               # document wrong type
        ],
    )
    def test_malformed_candidate_items(
        self, reranker: CrossEncoderReranker, bad: list
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            reranker.rerank("q", bad, top_k=1)

    @pytest.mark.parametrize("bad_top_k", [0, -1, -100])
    def test_invalid_top_k_value(
        self, reranker: CrossEncoderReranker, bad_top_k: int
    ) -> None:
        with pytest.raises(ValueError):
            reranker.rerank("q", [(0, 0.5, "doc")], top_k=bad_top_k)

    @pytest.mark.parametrize("bad_top_k", [None, 1.5, "3", True, False])
    def test_invalid_top_k_type(
        self, reranker: CrossEncoderReranker, bad_top_k: object
    ) -> None:
        with pytest.raises(TypeError):
            reranker.rerank("q", [(0, 0.5, "doc")], top_k=bad_top_k)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 3. Contextual compression
# --------------------------------------------------------------------------- #

class TestSentenceSplitting:
    def test_splits_on_terminal_punctuation(
        self, reranker: CrossEncoderReranker
    ) -> None:
        c = ContextualCompressionFilter(reranker)
        text = "First sentence. Second sentence! Third sentence? Fourth fragment"
        assert c.split_sentences(text) == [
            "First sentence.",
            "Second sentence!",
            "Third sentence?",
            "Fourth fragment",
        ]

    @pytest.mark.parametrize("text", ["", "   ", "\t\n  \t"])
    def test_empty_or_whitespace_returns_empty_list(
        self, reranker: CrossEncoderReranker, text: str
    ) -> None:
        assert ContextualCompressionFilter(reranker).split_sentences(text) == []

    def test_single_sentence(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker)
        assert c.split_sentences("Only one sentence.") == ["Only one sentence."]


class TestCompressDocument:
    def test_relevant_retained_irrelevant_removed(
        self, reranker: CrossEncoderReranker
    ) -> None:
        c = ContextualCompressionFilter(reranker, threshold=0.55)
        q = "gradient clipping"
        doc = (
            "Gradient clipping prevents exploding gradients during backpropagation. "
            "The web framework uses Node and Express. "
            "L1 and L2 regularization help prevent model overfitting."
        )
        out = c.compress_document(q, doc)
        assert "Gradient clipping" in out
        assert "Node and Express" not in out
        assert len(out) < len(doc)

    def test_order_preserved(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker, threshold=0.5)
        out = c.compress_document(
            "alpha beta", "Alpha is first. Beta is second. Gamma is third."
        )
        assert out.index("Alpha") < out.index("Beta")

    def test_threshold_boundary_is_inclusive(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """A sentence scoring exactly ``threshold`` must be retained.

        Uses the scorer itself to compute the exact cutoff, so we're proving
        ``>=`` semantics rather than a stale hard-coded constant.
        """
        exact = reranker.score_pair("cat", "Cat.")
        c = ContextualCompressionFilter(reranker, threshold=exact)
        out = c.compress_document("cat", "Cat. Dog.")
        assert "Cat." in out

    def test_fallback_to_top_sentence(self, reranker: CrossEncoderReranker) -> None:
        # threshold == 1.0 → nothing clears the bar → fallback required.
        c = ContextualCompressionFilter(reranker, threshold=1.0)
        out = c.compress_document("cat", "Cat. Dog. Bird.")
        assert out == "Cat."

    def test_fallback_picks_highest_scoring_middle_sentence(
        self, reranker: CrossEncoderReranker
    ) -> None:
        c = ContextualCompressionFilter(reranker, threshold=1.0)
        doc = "Unrelated text. The target keyword lives here. Another filler sentence."
        assert c.compress_document("target keyword", doc) == (
            "The target keyword lives here."
        )

    def test_threshold_zero_keeps_all_sentences(
        self, reranker: CrossEncoderReranker
    ) -> None:
        c = ContextualCompressionFilter(reranker, threshold=0.0)
        out = c.compress_document("alpha", "Alpha. Beta. Gamma.")
        for s in ("Alpha.", "Beta.", "Gamma."):
            assert s in out

    @pytest.mark.parametrize("text", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_document(
        self, reranker: CrossEncoderReranker, text: str
    ) -> None:
        c = ContextualCompressionFilter(reranker)
        assert c.compress_document("query", text) == ""

    def test_invalid_reranker(self) -> None:
        with pytest.raises(TypeError):
            ContextualCompressionFilter(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, "0.5", True, [0.5]])
    def test_invalid_threshold_type(
        self, reranker: CrossEncoderReranker, bad: object
    ) -> None:
        with pytest.raises(TypeError):
            ContextualCompressionFilter(reranker, threshold=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_threshold_value(
        self, reranker: CrossEncoderReranker, bad: float
    ) -> None:
        with pytest.raises(ValueError):
            ContextualCompressionFilter(reranker, threshold=bad)

    def test_invalid_query_type(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker)
        with pytest.raises(TypeError):
            c.compress_document(1, "doc")  # type: ignore[arg-type]

    def test_invalid_document_type(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker)
        with pytest.raises(TypeError):
            c.compress_document("q", 1)  # type: ignore[arg-type]


class TestCompressPayloads:
    def test_payload_structure(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker, threshold=0.5)
        reranked = [
            (0, 0.8, 0.75, "React is a UI library. Express is a Node framework."),
        ]
        payloads = c.compress_payloads("react express", reranked)
        assert len(payloads) == 1
        p = payloads[0]
        assert set(p.keys()) == {
            "doc_id",
            "original_score",
            "cross_encoder_score",
            "compressed_text",
            "original_char_count",
            "compressed_char_count",
        }
        assert p["doc_id"] == 0
        assert p["original_score"] == pytest.approx(0.8)
        assert p["cross_encoder_score"] == pytest.approx(0.75)

    def test_accurate_char_counts(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker, threshold=0.55)
        doc = "React is a UI library. Zebra quokka narwhal."
        reranked = [(0, 0.9, 0.9, doc)]
        p = c.compress_payloads("react", reranked)[0]
        assert p["original_char_count"] == len(doc)
        assert p["compressed_char_count"] == len(p["compressed_text"])
        assert p["compressed_char_count"] < p["original_char_count"]
        assert p["compressed_text"] == "React is a UI library."

    def test_provenance_preserved_across_payloads(
        self, reranker: CrossEncoderReranker
    ) -> None:
        """Every input candidate's identifiers must survive compression."""
        c = ContextualCompressionFilter(reranker, threshold=0.5)
        reranked = [
            (10, 0.9, 0.8, "React is a UI library. Express is a Node framework."),
            (20, 0.7, 0.6, "Gradient clipping prevents exploding gradients."),
        ]
        payloads = c.compress_payloads("react", reranked)
        assert [p["doc_id"] for p in payloads] == [10, 20]
        assert [p["original_score"] for p in payloads] == pytest.approx([0.9, 0.7])
        assert [p["cross_encoder_score"] for p in payloads] == pytest.approx([0.8, 0.6])

    def test_empty_payloads(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker)
        assert c.compress_payloads("q", []) == []

    @pytest.mark.parametrize(
        "bad",
        [
            [(0, 0.5, "only-three")],                  # tuple too short
            [(0, 0.5, 0.5, "doc", "extra")],           # tuple too long
            [("x", 0.5, 0.5, "doc")],                  # doc_id wrong type
            [(0, "0.5", 0.5, "doc")],                  # orig wrong type
            [(0, 0.5, float("nan"), "doc")],           # non-finite CE score
            [(0, 0.5, 0.5, 123)],                      # doc wrong type
        ],
    )
    def test_malformed_reranked_items(
        self, reranker: CrossEncoderReranker, bad: list
    ) -> None:
        c = ContextualCompressionFilter(reranker)
        with pytest.raises((TypeError, ValueError)):
            c.compress_payloads("q", bad)

    def test_malformed_container(self, reranker: CrossEncoderReranker) -> None:
        c = ContextualCompressionFilter(reranker)
        with pytest.raises(TypeError):
            c.compress_payloads("q", "not-a-sequence")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 4. End-to-end integration
# --------------------------------------------------------------------------- #

class TestEndToEndPipeline:
    def test_candidate_pool_to_pruned_payload(self) -> None:
        reranker = CrossEncoderReranker()
        compressor = ContextualCompressionFilter(reranker, threshold=0.55)

        query = "gradient clipping exploding gradients"
        candidates = [
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

        # Expected ordering under the *lexical* simulator:
        #   doc 1 — strong exact-token overlap (gradient / clipping /
        #           exploding / gradients×2) → highest CE score.
        #   doc 0 — zero exact overlap but partial overlap because "in"
        #           is a substring of both "clipping" and "exploding",
        #           giving a small positive contribution above the neutral
        #           sigmoid(0) = 0.5.
        #   doc 2 — zero lexical overlap at all → lands exactly on 0.5.
        assert [r[0] for r in reranked] == [1, 0, 2]
        assert reranked[0][2] > reranked[1][2] > reranked[2][2]
        assert reranked[2][2] == pytest.approx(0.5)

        # Original retrieval score survives re-ranking.
        assert reranked[0][1] == pytest.approx(0.55)

        payloads = compressor.compress_payloads(query, reranked)
        assert len(payloads) == 3

        top = payloads[0]
        assert top["doc_id"] == 1
        assert "Gradient clipping" in top["compressed_text"]
        # Second sentence is off-topic and must be dropped.
        assert "recurrent network training" not in top["compressed_text"]
        assert top["compressed_char_count"] < top["original_char_count"]

        for p in payloads:
            assert isinstance(p["compressed_text"], str)
            assert 0 <= p["compressed_char_count"] <= p["original_char_count"]