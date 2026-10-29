"""day56_speculative_decoding.py — one-step speculative decoding (CPU/NumPy).

What this file does
-------------------
This module implements a single draft–verify cycle of speculative decoding —
the exactness-preserving core of the algorithm — as a CPU/NumPy educational
reference.

In speculative decoding, a small and fast "draft" model proposes a short
sequence of K candidate tokens. A larger, slower "target" model then checks
all of them in one go. If the draft got them right, the target accepts
several tokens for the price of one target-model step. If the draft got one
wrong, the target rejects at the first disagreement and supplies a
replacement token that is distributed exactly as if the target itself had
produced it.

The algorithm, step by step:

  1. Draft tokens x_0, ..., x_{K-1} arrive together with their draft-model
     probability rows q_0, ..., q_{K-1}. The target model's per-position
     logits also arrive; they are converted into probability rows
     p_0, ..., p_{K-1} by a numerically stable softmax. A final target row
     p_K is held back for the bonus token.

  2. At each position i, the draft token x_i is accepted with probability

         alpha_i = min(1, p_i[x_i] / q_i[x_i])

     A token the target likes at least as much as the draft is accepted
     outright. A token the target likes less is accepted with probability
     proportional to that ratio.

  3. On the *first* rejection, one replacement token is drawn from the
     "residual" distribution — the part of the target's probability mass
     that the draft did not cover — namely max(p_i - q_i, 0), normalized.

  4. If all K drafted tokens are accepted, one bonus token is drawn from
     target row p_K.

Return value
------------
The function returns a tuple (accepted_count, final_token):

  accepted_count   number of *consecutive* draft tokens accepted before the
                   first rejection, or K if every drafted token was accepted.

  final_token      replacement token drawn after the first rejection, or
                   bonus token drawn after full acceptance.

The accepted prefix is NOT returned by this function; it is reconstructible
as draft_tokens[:accepted_count]. The returned pair is one step's worth of
information — not a full generated sequence.

Complexity
----------
Let K be the number of draft tokens and V the vocabulary size.

  Softmax over the (K+1, V) target-logit matrix
      O((K+1) * V) time and O((K+1) * V) temporary memory. The K+1 rows
      are the K draft-verification rows plus one bonus-token row. In
      conventional shorthand this is written O(K * V) when K >= 1; K = 0
      is an explicitly supported boundary case (a single bonus row, V
      exponentials). The softmax pass does one row maximum, one
      exponentiation, one row sum, and one division.

  Acceptance scan
      O(K) time — one ratio computation and one uniform draw per position,
      plus O(V) per-position row checks inside acceptance_probability
      (which the O(K * V) softmax already dominates).

  Residual construction
      O(V) time, executed only at the first rejection.

  Categorical sampling
      O(V) cumulative sum plus O(log V) binary search.

  Total
      O((K+1) * V), dominated by softmax.

Peak *additional* memory is O((K+1) * V), because softmax materializes the
full (K+1, V) target probability matrix before the acceptance scan begins.
The residual plus cumulative-sum workspace on the rejection path adds O(V)
on top. The caller's input arrays are not duplicated when they are already
float64 ndarrays; np.asarray(..., dtype=np.float64) may allocate a copy
when the caller passes Python lists, non-float64 arrays, or any input whose
dtype is not already float64.

One important distinction: reducing the number of target *forward passes*
is not the same as reducing target *arithmetic*. This simulator performs
no model compute — it exponentiates a (K+1, V) matrix and draws at most two
categorical samples. Its runtime should not be read as a prediction of real
model latency.

What this module is NOT
-----------------------
  * Not a GPU/TPU runtime. It has no HBM access, no KV cache, no batched
    kernels, and no measured bandwidth. The speedup expression that
    appears in the original spec,

        speedup ~= (m + 1) / (1 + (K - m) / k)

    where K is the drafted-token count per step, m is the expected number
    of accepted tokens per step (m <= K), and k is the target-to-draft
    cost ratio, is a *theoretical cost model* taken from that spec. The
    (K - m) / k term models expected wasted draft-model work per step
    under a linear cost model; it is inherited from the spec, not derived
    from this module's cost structure. The expression is not a theorem,
    not a measurement, and not a claim of 1.8x–2.5x real-world speedup.

  * Not a model loader. DraftTargetEnginePair holds two plain callables;
    it does not load, quantize, or execute a real model.

  * Not a silent input repair. Malformed shapes, non-finite logits,
    unnormalized rows, and out-of-range tokens raise loudly instead of
    being fixed silently. Within-tolerance rows (summing within
    PROBABILITY_ROW_SUM_TOLERANCE of 1) are accepted and then
    renormalized, so downstream code sees an exact distribution up to
    float64 round-off.

Reproducibility
---------------
All randomness flows through the caller-supplied numpy.random.Generator.
The module never reads or writes global RNG state. Identical inputs plus
an equally seeded generator produce identical output. There is no other
source of nondeterminism: iteration is positional, reductions are float64
and order-fixed, no sets or dicts are iterated, and no time or process
state is consulted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# The threshold below which a residual mass is treated as "numerically zero".
# When max(p - q, 0) sums to no more than this, the fallback policy in
# residual_distribution fires; see that function's docstring for the
# reasoning behind the specific value.
NEAR_ZERO_RESIDUAL_MASS: float = 1e-8

# How close a probability row must sum to 1.0 to be accepted by validation.
# Rows outside this tolerance raise; rows inside are renormalized to sum
# exactly to 1 (up to float64 round-off), so downstream code can treat them
# as exact distributions.
PROBABILITY_ROW_SUM_TOLERANCE: float = 1e-6


def _ensure_epsilon_is_valid(epsilon) -> float:
    """Return `epsilon` as a Python float, or raise TypeError/ValueError.

    This function explicitly rejects several classes of input that a naive
    float(epsilon) would silently accept:

      * bool and np.bool_ — float(True) is 1.0 and would slip through
        unnoticed. Note that np.bool_ is *not* an instance of the builtin
        bool, so a plain isinstance(x, bool) check misses it.

      * str and bytes — float("1e-8") would silently coerce a string into
        a number.

      * non-finite floats — NaN or +/-inf would make the residual-mass
        comparison behave unpredictably. For example, epsilon = NaN makes
        every comparison evaluate False, silently forcing the fallback
        branch for every input.

      * negative floats — these turn "residual_mass > epsilon" into a
        division by a residual whose sum is zero, which produces NaNs.

    Valid numeric types (int, float, np.integer, np.floating) pass through
    unchanged.
    """
    if isinstance(epsilon, (bool, np.bool_)):
        raise TypeError("epsilon must be a real number, not bool")
    if isinstance(epsilon, (str, bytes)):
        raise TypeError(f"epsilon must be a real number, not {type(epsilon).__name__}")
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"epsilon must be a real number; got {type(epsilon).__name__}"
        ) from exc
    if not np.isfinite(epsilon_value):
        raise ValueError(f"epsilon must be finite; got {epsilon_value}")
    if epsilon_value < 0.0:
        raise ValueError(f"epsilon must be nonnegative; got {epsilon_value}")
    return epsilon_value


def _ensure_probability_row_is_valid(
    probability_row: np.ndarray, row_name: str
) -> np.ndarray:
    """Return `probability_row` as a float64 1-D array, renormalized to sum 1.

    Checks, in order:

      * 1-D shape.
      * Finite values (no NaN, no +/-inf).
      * Nonnegative entries.
      * Sum within PROBABILITY_ROW_SUM_TOLERANCE of 1.0.

    Rows that fail any check raise. Rows that pass the tolerance check are
    then divided by their sum, so the returned array is an exact probability
    distribution up to float64 round-off — not one that is merely close.

    The renormalization step is what makes the module's exactness claim
    hold for inputs that arrive with floating-point drift (for example,
    softmax output, or a caller-supplied row whose sum is 1 + 1e-9). Without
    it, downstream code would consume the raw row:
    _draw_one_index_from_probs scales by the row's total mass,
    acceptance_probability uses raw target[drafted] / draft[drafted]
    ratios, and residual correction would be off by up to the tolerance
    amount. Renormalization eliminates that drift in one step.
    """
    values = np.asarray(probability_row, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{row_name} must be 1-D; got ndim={values.ndim}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{row_name} must be finite (no NaN/Inf)")
    if np.any(values < 0.0):
        raise ValueError(f"{row_name} must be nonnegative")
    row_total = float(values.sum())
    if not np.isclose(row_total, 1.0, atol=PROBABILITY_ROW_SUM_TOLERANCE, rtol=0.0):
        raise ValueError(
            f"{row_name} must sum to 1 within "
            f"{PROBABILITY_ROW_SUM_TOLERANCE}; got {row_total}"
        )
    return values / row_total


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return row-wise softmax probabilities, computed stably.

    Every row is first shifted down by its maximum entry, then exponentiated
    and divided by the row sum. This is the standard max-subtraction trick:
    after subtracting the row maximum, the surviving entry becomes
    exp(0) = 1, so the denominator is always >= 1 and there is no risk of
    overflow.

    No log-sum-exp is needed here because every row has a nonempty positive
    term — the maximum entry contributes exactly 1 to the sum. That is the
    entire numerical-stability argument.

    Raises ValueError if `logits` is not 2-D, has an empty vocabulary
    dimension, or contains any NaN or +/-inf.
    """
    logit_matrix = np.asarray(logits, dtype=np.float64)
    if logit_matrix.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (rows, vocab); got ndim={logit_matrix.ndim}"
        )
    if logit_matrix.shape[1] < 1:
        raise ValueError("vocabulary dimension must be >= 1")
    if not np.all(np.isfinite(logit_matrix)):
        raise ValueError("logits must be finite (no NaN/Inf); refused")

    # Subtract the row maximum from every entry before exponentiating. After
    # this shift, the maximum entry becomes 0 and contributes exp(0) = 1 to
    # the row sum, which guarantees a positive denominator.
    centered_logits = logit_matrix - np.max(logit_matrix, axis=1, keepdims=True)
    exponentiated_logits = np.exp(centered_logits)
    return exponentiated_logits / np.sum(exponentiated_logits, axis=1, keepdims=True)


def acceptance_probability(
    target_probs_row: np.ndarray,
    draft_probs_row: np.ndarray,
    token: int,
) -> float:
    """Return min(1, p[token] / q[token]) for one drafted token.

    The clamp at 1.0 is the acceptance criterion itself, not a numerical
    guard. A drafted token the target likes *more* than the draft is
    accepted with probability 1, and the residual correction accounts for
    the surplus mass elsewhere.

    Raises ValueError if draft_probs_row[token] == 0. A token genuinely
    sampled from q cannot have zero q-mass — you cannot draw a
    zero-probability token — so zero mass means the caller's draft tokens
    and draft rows are inconsistent. Silently treating that as acceptance
    probability 1 would hide the bug and break the exactness proof.

    Note: this function re-validates its arguments but does *not*
    renormalize them. Normalization is enforced upstream by
    _ensure_step_inputs_are_valid and _ensure_probability_row_is_valid.
    Called directly with unnormalized rows, the per-token ratio still uses
    the raw values; the exactness claim of speculative_decode_step does not
    apply to such direct calls.
    """
    target_probs = np.asarray(target_probs_row, dtype=np.float64)
    draft_probs = np.asarray(draft_probs_row, dtype=np.float64)
    if (
        target_probs.ndim != 1
        or draft_probs.ndim != 1
        or target_probs.shape != draft_probs.shape
    ):
        raise ValueError(
            f"target_probs_row and draft_probs_row must be 1-D of equal "
            f"shape; got {target_probs.shape} and {draft_probs.shape}"
        )
    if not (np.all(np.isfinite(target_probs)) and np.all(np.isfinite(draft_probs))):
        raise ValueError("target_probs_row and draft_probs_row must be finite")
    if np.any(target_probs < 0.0) or np.any(draft_probs < 0.0):
        raise ValueError("target_probs_row and draft_probs_row must be nonnegative")
    if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
        raise TypeError(f"token must be an integer; got {type(token).__name__}")
    if token < 0 or token >= target_probs.size:
        raise ValueError(f"token {token} outside [0, {target_probs.size})")

    draft_prob_at_token = float(draft_probs[token])
    if draft_prob_at_token <= 0.0:
        raise ValueError(
            f"drafted token {token} has zero draft probability; a token "
            "with zero draft probability could not have been sampled from "
            "the draft distribution"
        )
    probability_ratio = float(target_probs[token]) / draft_prob_at_token
    # The ratio may overflow to +inf when the draft probability is subnormal;
    # in that case the correct value is the limit 1.0.
    return min(1.0, probability_ratio)


def residual_distribution(
    target_probs_row: np.ndarray,
    draft_probs_row: np.ndarray,
    eps: float = NEAR_ZERO_RESIDUAL_MASS,
) -> np.ndarray:
    """Return the rejection-correction distribution max(p - q, 0), normalized.

    Contract
    --------
      * target_probs_row and draft_probs_row must be 1-D, finite,
        nonnegative, and sum to 1 within PROBABILITY_ROW_SUM_TOLERANCE.
        Within-tolerance rows are renormalized before use, so the residual
        is computed against exact distributions.
      * eps must be a *numeric* finite nonnegative real. Strings, bytes,
        bool, and numpy bool are rejected explicitly before any coercion;
        NaN, +/-inf, and negative values are rejected too.

    Policy for numerically-zero residual mass
    -----------------------------------------
    When the residual mass (the sum of max(p - q, 0)) is at most `eps`, the
    function returns p itself, post-renormalization.

    Justification: the residual mass equals the total-variation distance
    between p and q. So mass <= eps means the two distributions agree to
    within eps — and the correction branch is reached with probability at
    most that same eps. Sampling p in that branch therefore perturbs the
    emitted distribution by O(eps). The alternative — dividing by a
    residual that is numerically ~0 — produces arbitrary garbage. When
    mass is exactly 0, the two rows are identical, so p is exact.
    """
    epsilon = _ensure_epsilon_is_valid(eps)
    target_probs = _ensure_probability_row_is_valid(
        target_probs_row, "target_probs_row"
    )
    draft_probs = _ensure_probability_row_is_valid(draft_probs_row, "draft_probs_row")
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            f"target_probs_row and draft_probs_row must have equal shape; "
            f"got {target_probs.shape} and {draft_probs.shape}"
        )
    positive_residual = np.maximum(target_probs - draft_probs, 0.0)
    residual_mass = float(positive_residual.sum())
    if residual_mass > epsilon:
        return positive_residual / residual_mass
    return target_probs.copy()


def _draw_one_index_from_probs(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Draw one index from `probabilities` (does not need to sum to exactly 1).

    Uses inverse-CDF sampling: one uniform draw plus a binary search. There
    is no per-vocabulary Python loop, and no dependence on rng.choice's
    internal normalization check. The final clamp absorbs summation-order
    differences between probabilities.sum() and the last cumulative-sum
    entry.

    Sampling is invariant to the total mass of `probabilities` (the draw is
    scaled by probabilities.sum()), so un-normalized weights are already
    handled correctly here.
    """
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("probabilities must be a nonempty 1-D array")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total_mass = float(weights.sum())
    if total_mass <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    cumulative_probs = np.cumsum(weights)
    uniform_draw = float(rng.random()) * total_mass
    selected_index = int(np.searchsorted(cumulative_probs, uniform_draw, side="right"))
    return min(selected_index, weights.size - 1)


def _ensure_numpy_generator(rng, caller_name: str) -> None:
    """Raise TypeError unless `rng` is a numpy.random.Generator.

    Shared RNG type check used by both public entry points. Ensures a
    caller who passes a wrong RNG type gets a TypeError at the public API
    boundary, not an AttributeError from rng.random() deep inside sampling.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"{caller_name}: rng must be a numpy.random.Generator; seed it "
            "explicitly, e.g. np.random.default_rng(seed)"
        )


def _ensure_step_inputs_are_valid(
    draft_tokens: Sequence[int],
    target_logits_sequence: np.ndarray,
    draft_probs_sequence: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Validate all four inputs to speculative_decode_step.

    Returns a tuple (validated_tokens, target_logit_matrix,
    renormalized_draft_probability_matrix). Raises TypeError or ValueError
    for malformed inputs, with messages that name the violating contract.
    Inputs are never mutated.
    """
    _ensure_numpy_generator(rng, "speculative_decode_step")

    target_logit_matrix = np.asarray(target_logits_sequence, dtype=np.float64)
    if target_logit_matrix.ndim != 2:
        raise ValueError(
            f"target_logits_sequence must be 2-D (K+1, V); "
            f"got ndim={target_logit_matrix.ndim}"
        )
    num_rows, vocab_size = target_logit_matrix.shape
    if vocab_size < 1:
        raise ValueError("vocabulary dimension must be >= 1")
    if num_rows < 1:
        raise ValueError("target_logits_sequence must contain at least the bonus row")
    num_draft_tokens = num_rows - 1

    draft_probability_matrix = np.asarray(draft_probs_sequence, dtype=np.float64)
    if draft_probability_matrix.shape != (num_draft_tokens, vocab_size):
        raise ValueError(
            f"draft_probs_sequence must have shape "
            f"{(num_draft_tokens, vocab_size)} to match the {num_rows} "
            f"target rows and vocabulary {vocab_size}; got "
            f"{draft_probability_matrix.shape}"
        )

    try:
        raw_token_list = list(draft_tokens)
    except TypeError as exc:
        raise TypeError("draft_tokens must be a sequence of integer token ids") from exc
    if len(raw_token_list) != num_draft_tokens:
        raise ValueError(
            f"draft_tokens has length {len(raw_token_list)} but "
            f"target_logits_sequence implies K={num_draft_tokens}"
        )
    validated_tokens: list[int] = []
    for position, token in enumerate(raw_token_list):
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise TypeError(
                f"draft_tokens[{position}] must be an integer; "
                f"got {type(token).__name__}"
            )
        token_id = int(token)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"draft_tokens[{position}]={token_id} outside [0, {vocab_size})"
            )
        validated_tokens.append(token_id)

    if not np.all(np.isfinite(target_logit_matrix)):
        raise ValueError("target logits must be finite (no NaN/Inf); refused")
    if not np.all(np.isfinite(draft_probability_matrix)):
        raise ValueError("draft probabilities must be finite")
    if np.any(draft_probability_matrix < 0.0):
        raise ValueError("draft probabilities must be nonnegative")
    row_sums = draft_probability_matrix.sum(axis=1)
    if row_sums.size and not np.allclose(
        row_sums, 1.0, atol=PROBABILITY_ROW_SUM_TOLERANCE, rtol=0.0
    ):
        raise ValueError(
            f"each draft probability row must sum to 1 within "
            f"{PROBABILITY_ROW_SUM_TOLERANCE}; row sums were "
            f"{row_sums.tolist()}"
        )
    # Renormalize the tolerance into exactness. Without this, downstream
    # would consume draft rows that are within 1e-6 of a distribution but
    # not on it; the module's exactness claim requires the stronger
    # guarantee.
    if draft_probability_matrix.size:
        draft_probability_matrix = draft_probability_matrix / row_sums[:, np.newaxis]
    return validated_tokens, target_logit_matrix, draft_probability_matrix


def speculative_decode_step(
    draft_tokens: Sequence[int],
    target_logits_sequence: np.ndarray,
    draft_probs_sequence: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Run one draft–verify step; return (accepted_count, final_token).

    Exactness claim
    ---------------
    If draft_tokens[i] really was sampled from draft_probs_sequence[i], and
    target_logits_sequence holds the target logits conditioned on the same
    prefix, then the emitted sequence — accepted prefix plus final_token —
    is distributed as target sampling, up to float64 rounding.

    "Up to float64 rounding" is enforced, not assumed.
    _ensure_step_inputs_are_valid renormalizes draft rows after the
    sum-to-1 tolerance check; softmax returns the canonical normalized
    form; and residual_distribution renormalizes its output. The tolerance
    PROBABILITY_ROW_SUM_TOLERANCE therefore admits floating-point drift,
    not approximate distributions.

    The precondition (draft_tokens[i] actually sampled from
    draft_probs_sequence[i]) is the caller's responsibility and is not
    checked. Distributional exactness is *not* tested end-to-end by this
    module's test suite; it rests on the standard speculative-sampling
    result plus the renormalization above.

    Branch behaviour
    ----------------
      * Position i accepts iff u < min(1, target[i][x_i]/draft[i][x_i])
        for u ~ Uniform[0, 1) drawn from `rng`. The comparison is strict,
        so an acceptance probability of 0 (target assigns zero mass to the
        drafted token) always rejects.

      * The first rejection ends the step. Later draft tokens are never
        examined and never counted.

      * K == 0 is valid and means "draw only the bonus token", from
        target_logits_sequence[0].

    Errors
    ------
    Raises ValueError or TypeError for malformed shapes, non-finite logits,
    unnormalized rows, out-of-range or non-integer tokens, a drafted token
    with zero draft probability, or an `rng` that is not a
    numpy.random.Generator. Inputs are never silently normalized and never
    mutated.
    """
    validated_tokens, target_logit_matrix, draft_probability_matrix = (
        _ensure_step_inputs_are_valid(
            draft_tokens, target_logits_sequence, draft_probs_sequence, rng
        )
    )
    num_draft_tokens = len(validated_tokens)
    target_probability_matrix = softmax(target_logit_matrix)

    for position in range(num_draft_tokens):
        drafted_token = validated_tokens[position]
        accept_prob = acceptance_probability(
            target_probability_matrix[position],
            draft_probability_matrix[position],
            drafted_token,
        )
        if float(rng.random()) < accept_prob:
            continue
        replacement_probs = residual_distribution(
            target_probability_matrix[position],
            draft_probability_matrix[position],
        )
        return position, _draw_one_index_from_probs(replacement_probs, rng)

    return num_draft_tokens, _draw_one_index_from_probs(
        target_probability_matrix[num_draft_tokens], rng
    )


def _ensure_context_is_valid(context: Sequence[int], vocab_size: int) -> list[int]:
    """Return `context` as a list of integer token ids, or raise.

    Checks: sequence-convertible, each entry an integer (not bool), each
    entry in [0, vocab_size).
    """
    try:
        raw_tokens = list(context)
    except TypeError as exc:
        raise TypeError("context must be a sequence of integer token ids") from exc
    validated_tokens: list[int] = []
    for position, token in enumerate(raw_tokens):
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
            raise TypeError(f"context[{position}] must be an integer")
        token_id = int(token)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"context[{position}]={token_id} outside [0, {vocab_size})"
            )
        validated_tokens.append(token_id)
    return validated_tokens


@dataclass(frozen=True)
class DraftTargetEnginePair:
    """Typed pairing of a draft-probability engine and a target-logit engine.

    Contract
    --------
    This is the *whole* contract — nothing here loads, quantizes, or
    executes a real model.

      * draft_probs_fn(context) -> float64 array of shape (vocab_size,).
        A normalized next-token probability distribution from the draft
        model.

      * target_logits_fn(context) -> float64 array of shape (vocab_size,).
        Raw, unnormalized logits for the next token from the target model.
        They are passed through softmax by speculative_decode_step.

      * vocab_size > 0 and draft_len >= 0.

    What the methods do
    -------------------
    sample_draft_tokens runs the draft engine autoregressively for
    `draft_len` steps, appending each sampled token to the context before
    the next call.

    step samples a draft, evaluates the target engine at each of the K+1
    positions (position i sees the original context plus the first i draft
    tokens), and returns speculative_decode_step's
    (accepted_count, final_token).

    A real target engine would batch all K+1 positions into one forward
    pass. This simulator calls the target callable K+1 times in series,
    which is the honest CPU analogue; do not read a latency result into
    either shape.
    """

    draft_probs_fn: Callable[[Sequence[int]], np.ndarray]
    target_logits_fn: Callable[[Sequence[int]], np.ndarray]
    vocab_size: int
    draft_len: int

    def __post_init__(self) -> None:
        if not callable(self.draft_probs_fn):
            raise TypeError("draft_probs_fn must be callable")
        if not callable(self.target_logits_fn):
            raise TypeError("target_logits_fn must be callable")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
            raise TypeError("vocab_size must be an int")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if isinstance(self.draft_len, bool) or not isinstance(self.draft_len, int):
            raise TypeError("draft_len must be an int")
        if self.draft_len < 0:
            raise ValueError("draft_len must be >= 0")

    def _draft_row(self, context: Sequence[int]) -> np.ndarray:
        """Return the draft engine's next-token distribution as a validated row."""
        row = np.asarray(self.draft_probs_fn(list(context)), dtype=np.float64)
        if row.shape != (self.vocab_size,):
            raise ValueError(
                f"draft_probs_fn returned shape {row.shape}, expected "
                f"({self.vocab_size},)"
            )
        if not np.all(np.isfinite(row)) or np.any(row < 0.0):
            raise ValueError(
                "draft_probs_fn must return finite nonnegative probabilities"
            )
        if not np.isclose(
            float(row.sum()),
            1.0,
            atol=PROBABILITY_ROW_SUM_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("draft_probs_fn must return a normalized distribution")
        return row

    def _target_row(self, context: Sequence[int]) -> np.ndarray:
        """Return the target engine's logits as a validated row."""
        row = np.asarray(self.target_logits_fn(list(context)), dtype=np.float64)
        if row.shape != (self.vocab_size,):
            raise ValueError(
                f"target_logits_fn returned shape {row.shape}, expected "
                f"({self.vocab_size},)"
            )
        if not np.all(np.isfinite(row)):
            raise ValueError("target_logits_fn returned non-finite logits")
        return row

    def sample_draft_tokens(
        self, context: Sequence[int], rng: np.random.Generator
    ) -> tuple[list[int], np.ndarray]:
        """Autoregressively sample `draft_len` tokens; return (tokens, q_rows).

        Validates the RNG before drawing. Without this check,
        pair.sample_draft_tokens([0], 123) would reach rng.random() deep
        inside _draw_one_index_from_probs and raise AttributeError instead
        of the documented TypeError.
        """
        _ensure_numpy_generator(rng, "DraftTargetEnginePair.sample_draft_tokens")
        context_tokens = _ensure_context_is_valid(context, self.vocab_size)
        drafted_tokens: list[int] = []
        draft_probability_rows: list[np.ndarray] = []
        for _ in range(self.draft_len):
            draft_row = self._draft_row(context_tokens)
            drafted_token = _draw_one_index_from_probs(draft_row, rng)
            drafted_tokens.append(drafted_token)
            draft_probability_rows.append(draft_row)
            context_tokens.append(drafted_token)
        if draft_probability_rows:
            draft_probability_matrix = np.stack(draft_probability_rows, axis=0)
        else:
            draft_probability_matrix = np.zeros((0, self.vocab_size), dtype=np.float64)
        return drafted_tokens, draft_probability_matrix

    def step(self, context: Sequence[int], rng: np.random.Generator) -> tuple[int, int]:
        """Sample a draft and run one verify step; return (accepted_count, final_token)."""
        _ensure_numpy_generator(rng, "DraftTargetEnginePair.step")
        context_tokens = _ensure_context_is_valid(context, self.vocab_size)
        drafted_tokens, draft_probability_matrix = self.sample_draft_tokens(
            context_tokens, rng
        )
        target_logit_rows = [
            self._target_row(context_tokens + drafted_tokens[:i])
            for i in range(self.draft_len + 1)
        ]
        target_logit_matrix = np.stack(target_logit_rows, axis=0)
        return speculative_decode_step(
            drafted_tokens,
            target_logit_matrix,
            draft_probability_matrix,
            rng,
        )
