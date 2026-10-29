"""test_speculative_decoding.py — categorized tests for Day 56.

Every test is tagged with the mutation it catches:
  [ROBUST]   catches a specific mutated implementation
  [CONTRACT] catches a changed public interface
  [GUARD]    validation test; a subset of ROBUST under a three-category
             review framework (it catches "remove validation" as a mutation)

Deterministic branch control without scripting the RNG: an acceptance
probability of exactly 1.0 accepts for every u in [0, 1), and exactly 0.0
rejects for every u in [0, 1), because the comparison is strict
(`u < alpha`). One-hot residual distributions make replacement sampling
deterministic for any u. Only the tests that must observe a coin flip are
statistical.

Mutation-trace claims in test comments are STATIC traces; no mutation
variants were executed against this file. Treat "catches X" as "designed
to catch X" unless a run is reported.
"""

import dataclasses
import itertools

import numpy as np
import pytest
from day56_speculative_decoding import (
    DraftTargetEnginePair,
    acceptance_probability,
    residual_distribution,
    softmax,
    speculative_decode_step,
)


def _one_hot_logits(vocab: int, index: int) -> np.ndarray:
    """Finite logits whose softmax is exactly one-hot (exp(-1e9) underflows to 0)."""
    row = np.full(vocab, -1e9, dtype=np.float64)
    row[index] = 0.0
    return row


def _base_kwargs():
    q = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    logits = np.log(np.vstack([q, np.full(3, 1.0 / 3.0)]))
    return {
        "draft_tokens": [0, 1],
        "target_logits_sequence": logits,
        "draft_probs_sequence": q,
        "rng": np.random.default_rng(0),
    }


# --------------------------------------------------------------------------- #
# [ROBUST] correctness of the accept / reject branches
# --------------------------------------------------------------------------- #


def test_identical_rows_accept_every_draft_token():
    # [ROBUST] Catches an inverted acceptance comparison (`u > alpha`
    # accepts, which under p == q rejects every token) and catches comparing
    # target logits directly against draft probabilities without softmax
    # (the ratio then differs wildly from 1 and positions reject, or
    # acceptance_probability rejects the negative raw logits as
    # non-probabilities).
    # Float caveat: softmax(log q) differs from q by ~1e-16, so a correct
    # implementation can see alpha = 1 - 1e-16; this test would fail with
    # probability ~3e-16. That is acceptable; the mutations fail with
    # probability ~1.
    q = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.40, 0.30, 0.20],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    target_logits = np.vstack([np.log(q), _one_hot_logits(4, 2)])
    accepted, final_token = speculative_decode_step(
        [0, 1, 2], target_logits, q, np.random.default_rng(1234)
    )
    assert accepted == 3
    assert final_token == 2


def test_acceptance_probability_clamps_ratio_at_one():
    # [ROBUST] First assertion catches the min(1, .) clamp being removed:
    # for p[0]/q[0] = 0.8/0.2 = 4.0 the correct value is min(1, 4.0) = 1.0,
    # and the clamp-removed mutant returns 4.0.
    # Second assertion catches the ratio being reversed to q[x]/p[x]: the
    # correct value is p[0]/q[0] = 0.2/0.8 = 0.25, while the swapped mutant
    # computes q[0]/p[0] = 0.8/0.2 = 4.0 and the surviving clamp reduces it
    # to 1.0 — still a fail (the test expects 0.25).
    assert acceptance_probability(np.array([0.8, 0.2]), np.array([0.2, 0.8]), 0) == 1.0
    assert acceptance_probability(
        np.array([0.2, 0.8]), np.array([0.8, 0.2]), 0
    ) == pytest.approx(0.25)


def test_zero_target_mass_at_drafted_token_rejects():
    # [ROBUST] Catches the swapped ratio q[x]/p[x]: p[0] == 0 makes that
    # ratio infinite and clamps to alpha = 1.0, which would accept. With
    # the correct ratio alpha == 0 and the step must reject
    # deterministically.
    target_logits = np.stack([_one_hot_logits(2, 1), np.zeros(2)])
    q = np.array([[0.5, 0.5]])
    accepted, final_token = speculative_decode_step(
        [0], target_logits, q, np.random.default_rng(7)
    )
    assert accepted == 0
    # residual = max([0, 1] - [0.5, 0.5], 0) = [0, 0.5] -> [0, 1] -> token 1
    assert final_token == 1


def test_acceptance_frequency_tracks_alpha():
    # [ROBUST] Statistical. Catches alpha inverted to 1 - alpha (mean
    # 1500), the swapped ratio (alpha = 4 -> clamped to 1, mean 2000),
    # always-accept (2000) and always-reject (0).
    # alpha = p[0]/q[0] = 0.1/0.4 = 0.25. Accepts ~ Binomial(2000, 0.25):
    # mean 500, sigma 19.4. [380, 620] is > 6 sigma, so the correct
    # implementation fails this bound with probability < 1e-9. Limitation:
    # this pins alpha to roughly +/- 10%, not to 1e-3.
    q = np.array([[0.4, 0.6]])
    logits = np.log(np.array([[0.1, 0.9], [0.5, 0.5]]))
    rng = np.random.default_rng(2024)
    accepts = sum(
        speculative_decode_step([0], logits, q, rng)[0] == 1 for _ in range(2000)
    )
    assert 380 <= accepts <= 620


def test_rejection_samples_from_normalized_residual_not_from_p_or_q():
    # [ROBUST] Catches sampling the replacement from p (token 1 with prob
    # 0.06), from q (prob 0.05), or uniformly (prob 0.25) instead of from
    # the residual. The residual here is exactly one-hot at token 1, so
    # the correct implementation is deterministic for any u.
    #   p = [~0, 0.06, 0.50, 0.44], q = [0.01, 0.05, 0.50, 0.44]
    #   alpha = p[0]/q[0] ~ 1e-41  -> deterministic rejection
    #   residual = [0, 0.01, 0, 0] -> normalized [0, 1, 0, 0]
    # False-pass probability if the implementation sampled from p: 0.06^100.
    # Also catches in-place mutation of q inside residual_distribution: the
    # shared `q` array is reused for 100 iterations; a mutation that
    # corrupts q breaks downstream validation or the residual itself.
    q = np.array([[0.01, 0.05, 0.50, 0.44]])
    p = np.array([1e-43, 0.06, 0.50, 0.44])
    logits = np.vstack([np.log(p), _one_hot_logits(4, 0)])
    rng = np.random.default_rng(2024)
    for _ in range(100):
        accepted, final_token = speculative_decode_step([0], logits, q, rng)
        assert accepted == 0
        assert final_token == 1


def test_full_acceptance_samples_bonus_from_row_k():
    # [ROBUST] Catches the bonus token being drawn from row 0 (token 0) or
    # row K-1 (token 2) instead of row K (token 1).
    q = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    target_logits = np.stack(
        [
            _one_hot_logits(3, 0),  # p[0] = 1 >= q[0] = 0.5 -> accept
            _one_hot_logits(3, 2),  # p[2] = 1 >= q[2] = 0.5 -> accept
            _one_hot_logits(3, 1),  # bonus row
        ]
    )
    accepted, final_token = speculative_decode_step(
        [0, 2], target_logits, q, np.random.default_rng(11)
    )
    assert accepted == 2
    assert final_token == 1


def test_first_rejection_ends_the_scan():
    # [ROBUST] Catches counting non-consecutive acceptances (would report
    # 2) and catches scanning past the first rejection and using a later
    # position's residual (would report final_token 0 instead of 2).
    q = np.array(
        [
            [0.5, 0.3, 0.2],  # position 0: p[0] = 1 -> accept
            [0.1, 0.2, 0.7],  # position 1: p[1] = 0 -> reject
            [0.2, 0.3, 0.5],  # position 2: never reached
        ]
    )
    target_logits = np.stack(
        [
            _one_hot_logits(3, 0),
            _one_hot_logits(3, 2),
            _one_hot_logits(3, 0),
            _one_hot_logits(3, 1),
        ]
    )
    accepted, final_token = speculative_decode_step(
        [0, 1, 2], target_logits, q, np.random.default_rng(5)
    )
    assert accepted == 1
    # residual at position 1: max([0,0,1] - [.1,.2,.7], 0) = [0,0,.3]
    # -> token 2
    assert final_token == 2


def test_empty_draft_returns_only_the_bonus_token():
    # [ROBUST] Catches an off-by-one bonus row index (row K+1 is out of
    # bounds for K=0) and catches crashing on an empty draft list.
    target_logits = _one_hot_logits(3, 1).reshape(1, 3)
    draft_probs = np.zeros((0, 3), dtype=np.float64)
    accepted, final_token = speculative_decode_step(
        [], target_logits, draft_probs, np.random.default_rng(3)
    )
    assert accepted == 0
    assert final_token == 1


def test_single_token_vocabulary_accepts():
    # [ROBUST] Catches vocabulary indexing and bonus-row indexing errors on
    # the degenerate V=1 shape.
    target_logits = np.zeros((2, 1), dtype=np.float64)
    q = np.ones((1, 1), dtype=np.float64)
    accepted, final_token = speculative_decode_step(
        [0], target_logits, q, np.random.default_rng(0)
    )
    assert accepted == 1
    assert final_token == 0


def test_softmax_is_stable_on_extreme_finite_logits():
    # [ROBUST] Catches softmax without max-subtraction: exp(1000) overflows
    # to inf, inf/inf is NaN, and the finiteness assertion fails.
    probs = softmax(np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0]]))
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] == pytest.approx(1.0)
    assert probs[0, 2] == pytest.approx(0.0)
    assert np.allclose(probs[1], 1.0 / 3.0)


def test_seeded_generator_reproduces_and_changes_outcome():
    # [ROBUST] Catches an implementation that ignores `rng` (uses global RNG
    # or a hard-coded uniform draw): same-seed calls diverge, and the
    # across-seed sweep collapses to a single outcome.
    # Setup: K=1, V=2. alpha = p[0]/q[0] = 0.25/0.5 = 0.5 at the only
    # draft position. Accept -> (1, 1) because the bonus is one-hot at 1.
    # Reject -> (0, 1) because the residual is also one-hot at 1. The
    # tuple's first field therefore distinguishes the two branches.
    # Scope: verifies reproducibility within the current runtime and code
    # path; cross-NumPy-version stream stability is not established.
    q = np.array([[0.5, 0.5]])
    logits = np.vstack([np.log(np.array([0.25, 0.75])), _one_hot_logits(2, 1)])

    def run(seed: int):
        return speculative_decode_step([0], logits, q, np.random.default_rng(seed))

    for seed in (0, 7, 19):
        assert run(seed) == run(seed)

    outcomes = {run(seed) for seed in range(40)}
    # Both branches appear with probability 1 - 2 * 0.5**40 ~ 1e-12.
    assert outcomes == {(1, 1), (0, 1)}


def test_inputs_are_not_mutated():
    # [ROBUST] Catches in-place normalization inside
    # _ensure_step_inputs_are_valid and in-place max-subtraction inside
    # softmax on the caller's logits. Does NOT catch in-place mutation
    # inside residual_distribution because _base_kwargs fully accepts; that
    # mutation is caught by
    # test_rejection_samples_from_normalized_residual_not_from_p_or_q.
    kwargs = _base_kwargs()
    logits_before = kwargs["target_logits_sequence"].copy()
    probs_before = kwargs["draft_probs_sequence"].copy()
    tokens_before = list(kwargs["draft_tokens"])
    speculative_decode_step(**kwargs)
    assert np.array_equal(kwargs["target_logits_sequence"], logits_before)
    assert np.array_equal(kwargs["draft_probs_sequence"], probs_before)
    assert kwargs["draft_tokens"] == tokens_before


def test_residual_distribution_normalizes_and_has_epsilon_floor():
    # [ROBUST] Catches dividing by a ~zero mass (NaN/garbage) and catches
    # an implementation with no epsilon floor, which would return [1, 0]
    # here instead of the documented p fallback.
    p = np.array([0.5 + 5e-13, 0.5 - 5e-13])
    q = np.array([0.5, 0.5])
    assert np.allclose(residual_distribution(p, q), p)

    p_big = np.array([0.6, 0.4])
    q_big = np.array([0.4, 0.6])
    assert np.allclose(residual_distribution(p_big, q_big), np.array([1.0, 0.0]))


def test_rows_normalized_within_tolerance_are_accepted():
    # [ROBUST] Catches an exact `sum == 1.0` check: 1 + 1e-9 is a valid row
    # under the documented 1e-6 tolerance and must not raise.
    # Note: this test verifies ACCEPTANCE of within-tolerance rows, not
    # that they are renormalized. The renormalization cannot be observed
    # through the public API at the ~1e-9 drift level without exposing
    # internals; its exactness contribution is a float64-rounding-level
    # effect.
    q = np.array([[0.5, 0.5 + 1e-9]])
    logits = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    result = speculative_decode_step([0], logits, q, np.random.default_rng(1))
    assert isinstance(result, tuple) and len(result) == 2


def test_legacy_randomstate_is_rejected_as_rng():
    # [ROBUST] Catches relaxing the rng type check to accept legacy
    # RandomState or the global module. RandomState and Generator expose
    # different stream guarantees; silently accepting one for the other
    # breaks reproducibility contracts.
    with pytest.raises(TypeError):
        speculative_decode_step(
            [0],
            np.zeros((2, 2)),
            np.array([[0.5, 0.5]]),
            np.random.RandomState(0),
        )


def test_residual_distribution_rejects_unnormalized_rows():
    # [ROBUST] Catches the residual helper accepting invalid inputs.
    # Counterexample: p = [0.2, 0.2], q = [0.5, 0.5] has zero residual
    # mass, so an unvalidated helper returns p.copy() whose sum is 0.4 — a
    # value that is not a probability distribution.
    with pytest.raises(ValueError, match="sum to 1"):
        residual_distribution(np.array([0.2, 0.2]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="sum to 1"):
        residual_distribution(np.array([0.5, 0.5]), np.array([0.2, 0.2]))


@pytest.mark.parametrize(
    "bad_eps,exc",
    [
        (-1.0, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        ("1e-8", TypeError),
        (np.bool_(True), TypeError),
        (np.bool_(False), TypeError),
    ],
)
def test_residual_distribution_rejects_invalid_eps(bad_eps, exc):
    # [ROBUST] Catches an unvalidated or under-validated epsilon.
    # Counterexamples:
    #   eps = -1.0 -> `residual_mass > eps` becomes 0.0 > -1.0 (True), the
    #                 zero residual is divided by zero, NaNs returned with
    #                 a NumPy warning.
    #   eps = "1e-8" -> float("1e-8") silently coerces to a float.
    #   eps = np.bool_(True) -> isinstance(np.bool_(True), bool) is False,
    #                 so a plain bool check misses it;
    #                 float(np.bool_(True)) == 1.0 is silently accepted.
    with pytest.raises(exc):
        residual_distribution(np.array([0.5, 0.5]), np.array([0.5, 0.5]), eps=bad_eps)


def test_residual_distribution_accepts_numeric_eps():
    # [ROBUST] Guards the new epsilon validator against over-restriction:
    # valid numeric types (int, float, np.float64) must continue to work. A
    # future change that rejects `int(0)` or `np.float64(0.0)` would break
    # callers who pass integer eps=0 explicitly.
    p = np.array([0.6, 0.4])
    q = np.array([0.4, 0.6])
    for eps in (0, 0.0, np.float64(0.0), 1e-8):
        assert np.allclose(residual_distribution(p, q, eps=eps), np.array([1.0, 0.0]))


# --------------------------------------------------------------------------- #
# [CONTRACT]
# --------------------------------------------------------------------------- #


def test_returns_plain_python_ints():
    # [CONTRACT] Catches leaking numpy scalars (np.int64), which break JSON
    # serialization and isinstance(x, int) checks in downstream code.
    # Accept path (multiple seeds exercise the bonus-token sampling):
    for seed in range(5):
        kwargs = _base_kwargs()
        kwargs["rng"] = np.random.default_rng(seed)
        accepted, final_token = speculative_decode_step(**kwargs)
        assert type(accepted) is int
        assert type(final_token) is int
    # Reject path (deterministic rejection, replacement-token sampling):
    q_rej = np.array([[0.5, 0.5]])
    logits_rej = np.vstack([_one_hot_logits(2, 1), np.zeros(2)])
    accepted, final_token = speculative_decode_step(
        [0], logits_rej, q_rej, np.random.default_rng(0)
    )
    assert accepted == 0
    assert final_token == 1
    assert type(accepted) is int
    assert type(final_token) is int


def test_engine_pair_is_frozen():
    # [CONTRACT] Catches a mutable dataclass: callers could swap one engine
    # mid-run and silently break the p/q pairing that exactness depends on.
    pair = DraftTargetEnginePair(
        lambda ctx: np.ones(2) / 2.0, lambda ctx: np.zeros(2), 2, 1
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.draft_len = 5


def test_list_inputs_are_accepted_without_mutation():
    # [ROBUST] Catches an implementation that requires ndarray inputs and
    # crashes on lists (e.g. `target = target_logits_sequence` without
    # np.asarray). Also checks that coercion does not mutate the list.
    q_list = [[0.5, 0.5]]
    logits_list = [[0.0, 0.0], [0.0, 0.0]]
    accepted, final_token = speculative_decode_step(
        [0], logits_list, q_list, np.random.default_rng(0)
    )
    assert type(accepted) is int and type(final_token) is int
    assert q_list == [[0.5, 0.5]]
    assert logits_list == [[0.0, 0.0], [0.0, 0.0]]


# --------------------------------------------------------------------------- #
# [ROBUST] DraftTargetEnginePair behaviour
# --------------------------------------------------------------------------- #


def test_sample_draft_tokens_returns_valid_rows_and_tokens():
    # [ROBUST] Catches a sample_draft_tokens that returns unnormalized rows,
    # returns tokens outside [0, vocab_size), or advances context
    # incorrectly.
    contexts_seen = []

    def draft_fn(ctx):
        contexts_seen.append(tuple(ctx))
        return np.array([0.3, 0.7])

    pair = DraftTargetEnginePair(draft_fn, lambda c: np.zeros(2), 2, 2)
    tokens, probs = pair.sample_draft_tokens([1], np.random.default_rng(42))
    assert len(tokens) == 2
    assert all(0 <= t < 2 for t in tokens)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert contexts_seen[0] == (1,)
    assert len(contexts_seen[1]) == 2 and contexts_seen[1][0] == 1
    assert contexts_seen[1][1] == tokens[0]


def test_sample_draft_tokens_validates_rng():
    # [ROBUST] Catches a sample_draft_tokens that accepts a wrong RNG type
    # and lets the error surface as AttributeError deep inside sampling.
    # Counterexample: sample_draft_tokens([0], 123) with a one-token draft
    # length would reach rng.random() and raise AttributeError instead of
    # the documented TypeError.
    pair = DraftTargetEnginePair(
        lambda ctx: np.array([0.5, 0.5]), lambda ctx: np.zeros(2), 2, 1
    )
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        pair.sample_draft_tokens([0], 123)


def test_engine_pair_conditions_target_on_growing_prefix():
    # [ROBUST] Catches a `step` that hands the target callable the full
    # draft prefix at every position and catches a step that omits the
    # bonus evaluation (3 target calls instead of 4).
    seen = []

    def draft_fn(ctx):
        seen.append(("draft", tuple(ctx)))
        return np.array([0.5, 0.5])

    def target_fn(ctx):
        seen.append(("target", tuple(ctx)))
        return np.zeros(2)

    pair = DraftTargetEnginePair(draft_fn, target_fn, vocab_size=2, draft_len=3)
    result = pair.step([0, 1], np.random.default_rng(5))
    assert isinstance(result, tuple) and len(result) == 2
    assert 0 <= result[0] <= 3 and 0 <= result[1] < 2

    drafts = [entry for entry in seen if entry[0] == "draft"]
    targets = [entry for entry in seen if entry[0] == "target"]
    assert len(drafts) == 3
    assert len(targets) == 4
    assert [len(entry[1]) for entry in targets] == [2, 3, 4, 5]
    for earlier, later in itertools.pairwise(targets):
        assert later[1][: len(earlier[1])] == earlier[1]


def test_engine_pair_step_matches_manual_composition():
    # [ROBUST] Catches a `step` that forks or re-seeds the RNG between the
    # draft loop and the verify step, and catches passing logits/probs in
    # the wrong order to speculative_decode_step.
    # Scope: both branches call the same underlying helper, so this
    # verifies wiring and RNG continuity, not the algorithm itself.
    def draft_fn(ctx):
        return np.array([0.2, 0.3, 0.5])

    def target_fn(ctx):
        return np.array([0.0, 1.0, 0.5])

    pair = DraftTargetEnginePair(draft_fn, target_fn, vocab_size=3, draft_len=2)
    context = [1]
    seeded = pair.step(context, np.random.default_rng(99))

    rng = np.random.default_rng(99)
    tokens, probs = pair.sample_draft_tokens(context, rng)
    logits = np.stack([target_fn(context + tokens[:i]) for i in range(3)])
    manual = speculative_decode_step(tokens, logits, probs, rng)
    assert seeded == manual


# --------------------------------------------------------------------------- #
# [GUARD] input validation (a subset of ROBUST under 3-category frameworks)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides,exc",
    [
        ({"target_logits_sequence": np.zeros((2, 3))}, ValueError),
        ({"draft_probs_sequence": np.full((3, 3), 1.0 / 3.0)}, ValueError),
        ({"draft_tokens": [0, 3]}, ValueError),
        ({"draft_tokens": [-1, 0]}, ValueError),
        ({"draft_tokens": [0]}, ValueError),
        ({"draft_tokens": [0, 1.0]}, TypeError),
        ({"draft_tokens": [0, True]}, TypeError),
        (
            {"draft_probs_sequence": np.array([[0.55, 0.33, 0.22], [0.2, 0.5, 0.3]])},
            ValueError,
        ),
        (
            {"draft_probs_sequence": np.array([[0.5, 0.3, 0.2], [-0.1, 0.6, 0.5]])},
            ValueError,
        ),
        ({"draft_probs_sequence": np.zeros((2, 3))}, ValueError),
        ({"target_logits_sequence": np.zeros(3)}, ValueError),
        (
            {
                "target_logits_sequence": np.array(
                    [[0.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]
                )
            },
            ValueError,
        ),
        (
            {
                "target_logits_sequence": np.array(
                    [[0.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 0.0]]
                )
            },
            ValueError,
        ),
        (
            {
                "target_logits_sequence": np.zeros((3, 0)),
                "draft_probs_sequence": np.zeros((2, 0)),
            },
            ValueError,
        ),
        ({"rng": 42}, TypeError),
        ({"rng": None}, TypeError),
    ],
)
def test_malformed_inputs_raise(overrides, exc):
    # [GUARD] Catches silently normalizing, silently reshaping, or accepting
    # non-finite / out-of-range inputs instead of raising.
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    with pytest.raises(exc):
        speculative_decode_step(**kwargs)


def test_draft_token_with_zero_draft_probability_raises():
    # [GUARD] Catches the "treat q[x] == 0 as alpha = 1" shortcut, which
    # hides an inconsistency between draft_tokens and draft_probs_sequence.
    q = np.array([[0.5, 0.5, 0.0]])
    logits = np.log(np.array([[0.5, 0.5, 1e-30], [0.3, 0.3, 0.4]]))
    with pytest.raises(ValueError, match="zero draft probability"):
        speculative_decode_step([2], logits, q, np.random.default_rng(0))


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (
            {
                "draft_probs_fn": None,
                "target_logits_fn": lambda c: np.zeros(2),
                "vocab_size": 2,
                "draft_len": 1,
            },
            TypeError,
        ),
        (
            {
                "draft_probs_fn": lambda c: np.ones(2) / 2,
                "target_logits_fn": "not-callable",
                "vocab_size": 2,
                "draft_len": 1,
            },
            TypeError,
        ),
        (
            {
                "draft_probs_fn": lambda c: np.ones(2) / 2,
                "target_logits_fn": lambda c: np.zeros(2),
                "vocab_size": 0,
                "draft_len": 1,
            },
            ValueError,
        ),
        (
            {
                "draft_probs_fn": lambda c: np.ones(2) / 2,
                "target_logits_fn": lambda c: np.zeros(2),
                "vocab_size": 2,
                "draft_len": -1,
            },
            ValueError,
        ),
    ],
)
def test_engine_pair_construction_validation(kwargs, exc):
    # [GUARD] Catches constructors that accept unusable engines.
    with pytest.raises(exc):
        DraftTargetEnginePair(**kwargs)


# PROPERTY (Hypothesis, not executed — Hypothesis is not a project
# dependency):
#
# 1. For any finite (K+1, V) logit matrix, any (K, V) row-stochastic q,
#    any valid token list and any seed:
#      0 <= accepted_count <= K and 0 <= final_token < V.
# 2. If p_i == q_i for every position, accepted_count == K for every seed.
# 3. residual_distribution(p, q) has nonnegative entries, sums to 1 within
#    1e-6, and is 0 wherever p <= q.
# 4. Over N independent seeds, the accept frequency at a position with
#    alpha = 0.25 lies in [0.20, 0.30] for N = 2000.


# Not tested (and why):
# - Real model loading / GPU execution: this module is a simulator.
# - Measured wall-clock speedup: the spec's expression is a cost model; the
#   (K - m)/k term is inherited from the spec, not derived from this code.
# - End-to-end distributional exactness: the tests exercise accept,
#   reject-correction, bonus, and edge cases separately but do not compare
#   the empirical output distribution of speculative_decode_step against
#   direct target sampling over many seeds. That comparison would need
#   ~10^5 trials per cell for tight confidence intervals; out of scope here.
# - Renormalization is not observable through the public API at the ~1e-9
#   drift level it corrects.
#   test_rows_normalized_within_tolerance_are_accepted confirms the row is
#   accepted; it does not confirm it was renormalized. The renormalization
#   strengthens the exactness claim for within-tolerance inputs but cannot
#   be statistically distinguished from raw use at the accepted tolerance.
# - The zero-residual fallback inside speculative_decode_step is
#   unreachable through the public API; exercised directly on
#   residual_distribution.
# - Alpha precision: the frequency test bounds Binomial(2000, 0.25) at ~6σ;
#   it does not distinguish 0.25 from 0.27.
# - NumPy Generator stream stability across versions: no test hard-codes a
#   drawn value.
# - Concurrency and thread safety: the module is stateless by construction.
# - In-place mutation inside residual_distribution is not covered by
#   test_inputs_are_not_mutated (accept path only); it is covered by
#   test_rejection_samples_from_normalized_residual_not_from_p_or_q
#   (reject path, 100 iterations on a shared q).
