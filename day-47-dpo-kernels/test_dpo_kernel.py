"""test_dpo_kernel.py

Verification suite for day47_dpo_kernel.py.

Each test docstring begins with [ROBUST], [CONTRACT], or [GUARD] and names the
wrong implementation it catches. Tests were not executed in the authoring
environment; they are designed to be run with `pytest -v test_dpo_kernel.py`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from day47_dpo_kernel import DPOLossKernel

# --------------------------------------------------------------------------- #
# Correctness of the DPO objective
# --------------------------------------------------------------------------- #


def test_zero_margin_policy_equals_reference_yields_ln2_and_zero_rewards() -> None:
    """[ROBUST] Fails if reference log-probs are not subtracted, because zero
    implicit rewards would not occur."""
    kernel = DPOLossKernel(beta=0.1)
    p_chosen = np.array([-2.0, -3.0], dtype=np.float64)
    p_rejected = np.array([-4.0, -5.0], dtype=np.float64)
    r_chosen = p_chosen.copy()
    r_rejected = p_rejected.copy()

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert math.isclose(loss, math.log(2.0), rel_tol=0.0, abs_tol=1e-12)
    assert np.array_equal(metrics["chosen_rewards"], np.zeros(2))
    assert np.array_equal(metrics["rejected_rewards"], np.zeros(2))
    assert np.array_equal(metrics["reward_margin"], np.zeros(2))
    assert metrics["reward_accuracy"] == 0.0


def test_swapped_chosen_and_rejected_subtraction_reverses_margin() -> None:
    """[ROBUST] Fails if chosen/rejected reward subtraction is swapped, because
    reward_margin sign and the loss change."""
    kernel = DPOLossKernel(beta=0.3)
    p_chosen = np.array([-1.0, -2.0], dtype=np.float64)
    p_rejected = np.array([-4.0, -5.0], dtype=np.float64)
    r_chosen = np.array([-3.0, -3.5], dtype=np.float64)
    r_rejected = np.array([-2.0, -1.0], dtype=np.float64)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    expected_chosen = p_chosen - r_chosen
    expected_rejected = p_rejected - r_rejected
    expected_margin = expected_chosen - expected_rejected

    assert np.allclose(metrics["chosen_rewards"], expected_chosen)
    assert np.allclose(metrics["rejected_rewards"], expected_rejected)
    assert np.allclose(metrics["reward_margin"], expected_margin)

    expected_loss = float(np.mean(np.logaddexp(0.0, -0.3 * expected_margin)))
    assert np.isclose(loss, expected_loss)

    swapped_margin = expected_rejected - expected_chosen
    assert not np.allclose(expected_margin, swapped_margin)


def test_missing_reference_subtraction_changes_implicit_rewards() -> None:
    """[ROBUST] Fails if implicit rewards use policy log-probs without
    subtracting reference log-probs."""
    kernel = DPOLossKernel(beta=0.5)
    p_chosen = np.array([-1.0], dtype=np.float64)
    p_rejected = np.array([-2.0], dtype=np.float64)
    r_chosen = np.array([-3.0], dtype=np.float64)
    r_rejected = np.array([-2.5], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert metrics["chosen_rewards"][0] == pytest.approx(2.0)
    assert metrics["rejected_rewards"][0] == pytest.approx(0.5)
    assert metrics["reward_margin"][0] == pytest.approx(1.5)


def test_reversed_loss_logit_sign_changes_loss() -> None:
    """[ROBUST] Fails if the DPO loss uses +logits instead of -logits, because
    the expected loss value becomes wrong."""
    kernel = DPOLossKernel(beta=0.7)
    p_chosen = np.array([-0.5, -1.0], dtype=np.float64)
    p_rejected = np.array([-2.0, -2.5], dtype=np.float64)
    r_chosen = np.array([-1.0, -1.5], dtype=np.float64)
    r_rejected = np.array([-1.0, -1.0], dtype=np.float64)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)
    margin = metrics["reward_margin"]

    expected = float(np.mean(np.logaddexp(0.0, -0.7 * margin)))
    wrong = float(np.mean(np.logaddexp(0.0, 0.7 * margin)))

    assert np.isclose(loss, expected)
    assert not np.isclose(loss, wrong)


def test_scalar_loss_is_mean_not_sum() -> None:
    """[ROBUST] Fails if per-example losses are summed instead of averaged,
    because the scalar loss would scale with batch size."""
    kernel = DPOLossKernel(beta=1.0)
    p_chosen = np.array([0.0, 0.0], dtype=np.float64)
    p_rejected = np.array([0.0, 0.0], dtype=np.float64)
    r_chosen = np.array([-1.0, -2.0], dtype=np.float64)
    r_rejected = np.array([0.0, 0.0], dtype=np.float64)

    loss, _ = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    per_example = np.logaddexp(0.0, -np.array([1.0, 2.0], dtype=np.float64))

    assert np.isclose(loss, float(np.mean(per_example)))
    assert not np.isclose(loss, float(np.sum(per_example)))


def test_single_example_batch_loss_matches_closed_form() -> None:
    """[ROBUST] Fails if any reduction or reshape assumes batch size > 1."""
    kernel = DPOLossKernel(beta=2.0)
    p_chosen = np.array([-0.3], dtype=np.float64)
    p_rejected = np.array([-0.9], dtype=np.float64)
    r_chosen = np.array([-0.5], dtype=np.float64)
    r_rejected = np.array([-0.4], dtype=np.float64)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    expected_loss = float(np.log1p(np.exp(-1.4)))
    assert np.isclose(loss, expected_loss)
    assert metrics["chosen_rewards"].shape == (1,)
    assert metrics["rejected_rewards"].shape == (1,)
    assert metrics["reward_margin"].shape == (1,)
    assert metrics["grad_policy_chosen"].shape == (1,)
    assert metrics["grad_policy_rejected"].shape == (1,)


def test_loss_equals_mean_of_stable_logaddexp_objective() -> None:
    """[ROBUST] Fails if per-example objective or mean reduction differs from
    the specified DPO objective."""
    kernel = DPOLossKernel(beta=0.3)
    p_chosen = np.array([-0.1, -5.0, 0.0], dtype=np.float64)
    p_rejected = np.array([-3.0, -2.0, -1.0], dtype=np.float64)
    r_chosen = np.array([-2.0, -5.0, -0.5], dtype=np.float64)
    r_rejected = np.array([-0.5, -1.0, -1.0], dtype=np.float64)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)
    expected = float(np.mean(np.logaddexp(0.0, -0.3 * metrics["reward_margin"])))

    assert np.isclose(loss, expected)


# --------------------------------------------------------------------------- #
# Gradients: magnitude, sign, and finite-difference checks
# --------------------------------------------------------------------------- #


def test_gradients_include_one_over_batch_mean_factor() -> None:
    """[ROBUST] Fails if returned gradients omit the 1/B factor from the
    batch-mean loss."""
    kernel = DPOLossKernel(beta=0.5)
    p_chosen = np.array([-1.0, -2.0], dtype=np.float64)
    p_rejected = np.array([-3.0, -4.0], dtype=np.float64)
    r_chosen = np.array([-2.0, -2.0], dtype=np.float64)
    r_rejected = np.array([-2.0, -2.0], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    margin = metrics["reward_margin"]
    z = 0.5 * margin
    sigmoid_neg_z = 1.0 / (1.0 + np.exp(z))
    batch_size = 2

    expected_chosen = -0.5 * sigmoid_neg_z / batch_size
    expected_rejected = 0.5 * sigmoid_neg_z / batch_size

    assert np.allclose(metrics["grad_policy_chosen"], expected_chosen)
    assert np.allclose(metrics["grad_policy_rejected"], expected_rejected)
    assert not np.allclose(metrics["grad_policy_chosen"], expected_chosen * batch_size)


def test_gradient_signs_and_magnitudes_are_opposite() -> None:
    """[ROBUST] Fails if either analytic policy gradient sign is reversed, or
    if the two gradient arrays are not exactly opposite per example."""
    kernel = DPOLossKernel(beta=0.2)
    p_chosen = np.array([-2.0, -3.0], dtype=np.float64)
    p_rejected = np.array([-4.0, -5.0], dtype=np.float64)
    r_chosen = np.array([-1.0, -1.0], dtype=np.float64)
    r_rejected = np.array([-1.0, -1.0], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert np.all(metrics["grad_policy_chosen"] < 0.0)
    assert np.all(metrics["grad_policy_rejected"] > 0.0)
    assert np.allclose(metrics["grad_policy_chosen"], -metrics["grad_policy_rejected"])
    assert np.all(np.abs(metrics["grad_policy_chosen"]) > 0.0)


def test_central_finite_difference_matches_chosen_policy_gradient() -> None:
    """[ROBUST] Fails if chosen policy gradient sign, beta factor, or 1/B
    factor is wrong. Central differences use the scalar loss returned by
    compute_loss."""
    kernel = DPOLossKernel(beta=0.8)
    p_chosen = np.array([-1.2, -0.7], dtype=np.float64)
    p_rejected = np.array([-2.5, -1.1], dtype=np.float64)
    r_chosen = np.array([-1.0, -0.5], dtype=np.float64)
    r_rejected = np.array([-1.5, -0.9], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)
    eps = 1e-6

    for i in range(p_chosen.size):
        plus = p_chosen.copy()
        minus = p_chosen.copy()
        plus[i] += eps
        minus[i] -= eps

        loss_plus, _ = kernel.compute_loss(plus, p_rejected, r_chosen, r_rejected)
        loss_minus, _ = kernel.compute_loss(minus, p_rejected, r_chosen, r_rejected)
        fd = (loss_plus - loss_minus) / (2.0 * eps)

        assert np.isclose(metrics["grad_policy_chosen"][i], fd, rtol=1e-5, atol=1e-7)


def test_central_finite_difference_matches_rejected_policy_gradient() -> None:
    """[ROBUST] Fails if rejected policy gradient sign, beta factor, or 1/B
    factor is wrong."""
    kernel = DPOLossKernel(beta=0.8)
    p_chosen = np.array([-1.2, -0.7], dtype=np.float64)
    p_rejected = np.array([-2.5, -1.1], dtype=np.float64)
    r_chosen = np.array([-1.0, -0.5], dtype=np.float64)
    r_rejected = np.array([-1.5, -0.9], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)
    eps = 1e-6

    for i in range(p_rejected.size):
        plus = p_rejected.copy()
        minus = p_rejected.copy()
        plus[i] += eps
        minus[i] -= eps

        loss_plus, _ = kernel.compute_loss(p_chosen, plus, r_chosen, r_rejected)
        loss_minus, _ = kernel.compute_loss(p_chosen, minus, r_chosen, r_rejected)
        fd = (loss_plus - loss_minus) / (2.0 * eps)

        assert np.isclose(metrics["grad_policy_rejected"][i], fd, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- #
# Reward accuracy
# --------------------------------------------------------------------------- #


def test_reward_accuracy_uses_margin_not_raw_policy_logps() -> None:
    """[ROBUST] Fails if reward_accuracy is computed from raw policy log-probs
    instead of reward margins. Here p_chosen > p_rejected but margin < 0."""
    kernel = DPOLossKernel(beta=0.2)
    p_chosen = np.array([-2.0], dtype=np.float64)
    p_rejected = np.array([-3.0], dtype=np.float64)
    r_chosen = np.array([-0.5], dtype=np.float64)
    r_rejected = np.array([-10.0], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert p_chosen[0] > p_rejected[0]
    assert metrics["reward_margin"][0] < 0.0
    assert metrics["reward_accuracy"] == 0.0


def test_reward_accuracy_counts_only_strictly_positive_margins() -> None:
    """[ROBUST] Fails if zero margin is counted as correct, e.g. by using
    >= instead of >."""
    kernel = DPOLossKernel(beta=0.5)
    p_chosen = np.array([-1.0, -1.0], dtype=np.float64)
    p_rejected = np.array([-1.0, -3.0], dtype=np.float64)
    r_chosen = np.array([-1.0, -1.0], dtype=np.float64)
    r_rejected = np.array([-1.0, -1.0], dtype=np.float64)

    _, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert np.allclose(metrics["reward_margin"], np.array([0.0, 2.0], dtype=np.float64))
    assert metrics["reward_accuracy"] == 0.5


# --------------------------------------------------------------------------- #
# Numerical stability
# --------------------------------------------------------------------------- #


def test_large_negative_logit_is_finite_and_not_overflow() -> None:
    """[ROBUST] Fails if log(1+exp(-z)) is evaluated naively and overflows
    for large negative logits."""
    kernel = DPOLossKernel(beta=1.0)
    p_chosen = np.array([-1000.0], dtype=np.float64)
    p_rejected = np.array([0.0], dtype=np.float64)
    r_chosen = np.array([0.0], dtype=np.float64)
    r_rejected = np.array([0.0], dtype=np.float64)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert np.isfinite(loss)
    assert loss == pytest.approx(1000.0, rel=1e-12, abs=1e-12)
    assert np.all(np.isfinite(metrics["grad_policy_chosen"]))
    assert np.all(np.isfinite(metrics["grad_policy_rejected"]))


def test_large_positive_logit_is_not_clipped_to_artificial_ceiling() -> None:
    """[ROBUST] Fails if logits are clipped at e.g. 100, because the loss for
    a large positive logit would then be nonzero instead of underflowing to 0."""
    kernel = DPOLossKernel(beta=1.0)
    p_chosen = np.array([0.0], dtype=np.float64)
    p_rejected = np.array([0.0], dtype=np.float64)
    r_chosen = np.array([-1000.0], dtype=np.float64)
    r_rejected = np.array([0.0], dtype=np.float64)

    loss, _ = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    # logit = 1000. logaddexp(0, -1000) underflows to 0.0 in float64.
    # A clip at 100 would give logaddexp(0, -100) > 0, so this catches it.
    assert loss == 0.0


def test_extreme_sigmoid_inputs_produce_finite_gradients() -> None:
    """[ROBUST] Fails if the sigmoid evaluates the unstable exp branch. A naive
    np.where sigmoid computes correct values here but emits a RuntimeWarning
    from the discarded exp(700) branch; simplefilter("error") converts that
    warning into a failure, so this test catches both the NaN-producing and
    the warning-producing variants."""
    kernel = DPOLossKernel(beta=1.0)

    # Force logits near +-700: margin = policy_chosen - ref_chosen -
    # (policy_rejected - ref_rejected). Set refs to zero, margin = chosen - rejected.
    p_chosen = np.array([700.0, -700.0], dtype=np.float64)
    p_rejected = np.array([0.0, 0.0], dtype=np.float64)
    r_chosen = np.array([0.0, 0.0], dtype=np.float64)
    r_rejected = np.array([0.0, 0.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert np.isfinite(loss)
    assert np.all(np.isfinite(metrics["grad_policy_chosen"]))
    assert np.all(np.isfinite(metrics["grad_policy_rejected"]))
    # At logit +700, sigmoid(-700) -> 0, so both gradients must be ~0.
    assert metrics["grad_policy_chosen"][0] == pytest.approx(0.0, abs=1e-300)
    assert metrics["grad_policy_rejected"][0] == pytest.approx(0.0, abs=1e-300)
    # At logit -700, sigmoid(700) -> 1, so gradient magnitude is beta / B.
    assert metrics["grad_policy_chosen"][1] == pytest.approx(-1.0 / 2)
    assert metrics["grad_policy_rejected"][1] == pytest.approx(1.0 / 2)


def test_reward_subtraction_overflow_raises_floating_point_error() -> None:
    """[ROBUST] Fails if float64 overflow in reward subtraction leaks as NaN/inf,
    or emits a RuntimeWarning instead of raising FloatingPointError. Uses
    policy_chosen = +max, ref_chosen = -max so chosen_rewards overflows."""
    kernel = DPOLossKernel(beta=0.1)
    huge = np.finfo(np.float64).max

    p_chosen = np.array([huge], dtype=np.float64)
    p_rejected = np.array([0.0], dtype=np.float64)
    r_chosen = np.array([-huge], dtype=np.float64)
    r_rejected = np.array([0.0], dtype=np.float64)

    with (
    pytest.raises(FloatingPointError, match="chosen_rewards"),
    warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", RuntimeWarning)
        kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)


def test_logit_multiplication_overflow_raises_floating_point_error() -> None:
    """[ROBUST] Fails if overflow from beta * reward_margin is not caught.
    Uses finite rewards whose product with beta exceeds float64 max."""
    kernel = DPOLossKernel(beta=1e308)
    p_chosen = np.array([1e10], dtype=np.float64)
    p_rejected = np.array([0.0], dtype=np.float64)
    r_chosen = np.array([0.0], dtype=np.float64)
    r_rejected = np.array([0.0], dtype=np.float64)

    with pytest.raises(FloatingPointError, match="logits"):
        kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)


def test_output_arrays_are_float64() -> None:
    """[ROBUST] Fails if the kernel returns float32 or downcast arrays, or if
    `loss` is a numpy scalar rather than a Python float. The documented contract
    pins output dtype to float64 and `loss` to Python float."""
    kernel = DPOLossKernel(beta=0.1)
    p_chosen = np.array([-1.0], dtype=np.float32)
    p_rejected = np.array([-2.0], dtype=np.float32)
    r_chosen = np.array([-1.0], dtype=np.float32)
    r_rejected = np.array([-2.0], dtype=np.float32)

    loss, metrics = kernel.compute_loss(p_chosen, p_rejected, r_chosen, r_rejected)

    assert type(loss) is float
    for key in (
        "chosen_rewards",
        "rejected_rewards",
        "reward_margin",
        "grad_policy_chosen",
        "grad_policy_rejected",
    ):
        assert metrics[key].dtype == np.float64, key


# --------------------------------------------------------------------------- #
# Validation and edge cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_beta", [0.0, -0.1, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_beta_rejected(bad_beta: float) -> None:
    """[GUARD] Fails if beta is accepted when non-finite or <= 0."""
    with pytest.raises(ValueError):
        DPOLossKernel(beta=bad_beta)


def test_non_numeric_beta_rejected_with_consistent_message() -> None:
    """[GUARD] Fails if a non-numeric beta leaks a raw TypeError instead of a
    kernel-specific ValueError."""
    with pytest.raises(ValueError, match="beta must be a real number"):
        DPOLossKernel(beta="not-a-number")  # type: ignore[arg-type]


def test_non_numeric_input_raises_consistent_value_error() -> None:
    """[GUARD] Fails if a non-convertible array element leaks a raw
    ValueError/TypeError without naming the offending argument."""
    kernel = DPOLossKernel()
    good = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="policy_chosen_logps"):
        kernel.compute_loss(["abc"], good, good, good)
    with pytest.raises(ValueError, match="policy_rejected_logps"):
        kernel.compute_loss(good, ["abc"], good, good)
    with pytest.raises(ValueError, match="ref_chosen_logps"):
        kernel.compute_loss(good, good, ["abc"], good)
    with pytest.raises(ValueError, match="ref_rejected_logps"):
        kernel.compute_loss(good, good, good, ["abc"])


def test_mismatched_shapes_raise_instead_of_broadcasting() -> None:
    """[GUARD] Fails if NumPy broadcasting silently accepts mismatched batch
    shapes."""
    kernel = DPOLossKernel(beta=0.1)

    with pytest.raises(ValueError, match="identical shapes"):
        kernel.compute_loss(
            np.zeros(2, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )


def test_scalar_two_dimensional_and_empty_inputs_rejected() -> None:
    """[GUARD] Fails if scalar, 2-D, or empty batch inputs are accepted."""
    kernel = DPOLossKernel()

    with pytest.raises(ValueError, match="1-D"):
        kernel.compute_loss(np.float64(0.0), np.zeros(1), np.zeros(1), np.zeros(1))

    with pytest.raises(ValueError, match="1-D"):
        kernel.compute_loss(np.zeros((1, 1)), np.zeros(1), np.zeros(1), np.zeros(1))

    with pytest.raises(ValueError, match="at least one"):
        kernel.compute_loss(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(bad: float) -> None:
    """[GUARD] Fails if NaN or infinity in any input array is silently
    propagated through the public API."""
    kernel = DPOLossKernel()
    bad_arr = np.array([bad], dtype=np.float64)
    good_arr = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="NaN or infinity"):
        kernel.compute_loss(bad_arr, good_arr, good_arr, good_arr)
    with pytest.raises(ValueError, match="NaN or infinity"):
        kernel.compute_loss(good_arr, bad_arr, good_arr, good_arr)
    with pytest.raises(ValueError, match="NaN or infinity"):
        kernel.compute_loss(good_arr, good_arr, bad_arr, good_arr)
    with pytest.raises(ValueError, match="NaN or infinity"):
        kernel.compute_loss(good_arr, good_arr, good_arr, bad_arr)


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #


def test_metrics_do_not_expose_reference_or_model_parameter_gradients() -> None:
    """[CONTRACT] Fails if metrics add reference gradients or mislabel
    policy-array gradients as model-parameter gradients. This test verifies
    the exact key set only; the numerical meaning of the two gradient arrays
    is verified separately by the finite-difference tests."""
    kernel = DPOLossKernel(beta=0.1)
    _, metrics = kernel.compute_loss(
        np.array([-1.0], dtype=np.float64),
        np.array([-2.0], dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        np.array([-2.0], dtype=np.float64),
    )

    assert set(metrics.keys()) == {
        "loss",
        "chosen_rewards",
        "rejected_rewards",
        "reward_margin",
        "reward_accuracy",
        "grad_policy_chosen",
        "grad_policy_rejected",
    }


# PROPERTY: For any finite 1-D arrays policy_chosen_logps, policy_rejected_logps,
# ref_chosen_logps, ref_rejected_logps of equal non-zero length B, and any finite
# beta > 0, compute_loss returns a finite scalar loss equal to
# mean(logaddexp(0, -beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)))),
# finite batch-shaped reward arrays, finite batch-shaped policy gradients, and
# reward_accuracy equal to mean(reward_margin > 0).
#
# Hypothesis strategy:
#   B = st.integers(min_value=1, max_value=64)
#   logp = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
#   beta = st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False)
#   arrays = st.lists(logp, min_size=B, max_size=B).map(np.asarray)
# Magnitudes are capped at 1e6 so the property test itself does not overflow;
# deterministic tests above separately exercise ~700-magnitude logits, 1e10
# rewards with beta=1e308, and float64.max subtraction overflow.

# Not tested (and why):
# - Full DPO training, model-parameter gradients, transformer forward passes,
#   tokenization, reward-model training, PPO, RLHF, and multi-GPU behavior:
#   this kernel only consumes precomputed sequence log-probabilities and
#   returns gradients with respect to those arrays.
# - Hypothesis property tests were not executed because Hypothesis is not a
#   required dependency and execution was unavailable in the authoring
#   environment; the strategy is documented above and in TEST_STRATEGY.md.
# - GPU execution is not tested; the kernel is pure NumPy CPU code.
# - Multi-dimensional sequence log-probabilities, per-token DPO, and
#   padding/masking semantics are out of scope.
# - Attribute mutation after construction (`kernel.beta = -1`) is not
#   validated; the public contract assumes construction-time validation only.
# - Non-float array-like objects (Decimal, Fraction) are converted via
#   np.asarray(..., dtype=np.float64); precision loss on exotic dtypes is
#   not tested.
# - The kernel rejects non-finite intermediates; there is no attempt to
#   compute mathematically-defined results that exceed float64 range.
