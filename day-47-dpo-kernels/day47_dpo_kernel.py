"""day47_dpo_kernel.py

Direct Preference Optimization (DPO) alignment loss kernel.

Problem
-------
Given precomputed sequence-level log-probabilities from a policy model and a
reference model for chosen/rejected response pairs, compute the scalar DPO
loss, implicit rewards, reward margins, reward accuracy, and analytic
gradients of the batch-mean loss with respect to the policy log-probabilities.

Array-like inputs are accepted and converted to float64 NumPy arrays. The
kernel does not run any model, tokenizer, reward model, or PPO loop.

Complexity
----------
For batch size B, the kernel performs a constant number of NumPy elementwise
operations over length-B arrays. Time is O(B) and auxiliary space is O(B),
because the implementation materializes chosen/rejected rewards, margins,
logits, per-example losses, sigmoid(-logits), and two gradient arrays, each
of length B. Upstream cost of computing log-probabilities (policy and
reference forward passes) is not included in this kernel's complexity.

What this is NOT
----------------
This is not full DPO training, RLHF, PPO, a reward model, a transformer, a
tokenizer, or a production LLM fine-tuning system. It does not backpropagate
through model parameters. It consumes only supplied log-probabilities and
returns gradients with respect to those supplied policy log-probabilities.
Reference log-probabilities are fixed inputs and receive no gradients.

Reproducibility
---------------
This module is deterministic; no seed handling is required. There is no
randomness, no RNG, and no hidden state beyond the validated `beta` value.

Output dtypes
-------------
All returned arrays (`chosen_rewards`, `rejected_rewards`, `reward_margin`,
`grad_policy_chosen`, `grad_policy_rejected`) are float64. `loss` and
`reward_accuracy` are Python floats (float64 semantics).

Numerical failure behavior
--------------------------
Intermediate arithmetic (reward subtraction, logit scaling, logit negation)
is evaluated under `np.errstate(over="ignore", invalid="ignore")` so that
unexpected float overflow does not surface as a `RuntimeWarning` on stderr.
Finiteness of every intermediate and every returned array is checked
explicitly; any non-finite value raises `FloatingPointError` with the name
of the offending quantity. Warnings are not the primary failure signal.

`beta` mutation contract
------------------------
`beta` is validated once, at construction. Assigning to `kernel.beta` after
construction is not re-validated and is unsupported: callers must not mutate
this attribute. If per-call beta variation is ever needed, construct a new
kernel or add an explicit `beta` argument to `compute_loss` and re-validate
there. This is a deliberate scope limit, not an oversight.

Implementation note
-------------------
The source blueprint contained a malformed dictionary entry:
    "rejected_rewards": rejected_rejected = rejected_rewards
That is invalid Python inside a dict literal. This implementation uses the
intended key/value mapping:
    "rejected_rewards": rejected_rewards
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class DPOMetrics(TypedDict):
    """Typed metrics returned by :meth:`DPOLossKernel.compute_loss`."""

    loss: float
    chosen_rewards: FloatArray
    rejected_rewards: FloatArray
    reward_margin: FloatArray
    reward_accuracy: float
    grad_policy_chosen: FloatArray
    grad_policy_rejected: FloatArray


class DPOLossKernel:
    """First-principles NumPy DPO loss kernel.

    The kernel is deterministic and stateless except for the validated `beta`
    hyperparameter. `beta` is validated once at construction and must not be
    mutated afterwards; see the module docstring's mutation contract.
    """

    def __init__(self, beta: float = 0.1) -> None:
        try:
            beta_float = float(beta)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"beta must be a real number, got {beta!r}") from exc
        if not math.isfinite(beta_float):
            raise ValueError(f"beta must be finite, got {beta!r}")
        if beta_float <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta_float!r}")
        self.beta = beta_float

    @staticmethod
    def _validate_batch(name: str, values: npt.ArrayLike) -> FloatArray:
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} could not be converted to float64: {exc}"
            ) from exc
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D batch array; got ndim={arr.ndim}, "
                f"shape={arr.shape}"
            )
        if arr.size == 0:
            raise ValueError(f"{name} must contain at least one example")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinity")
        return arr

    @staticmethod
    def _require_finite(name: str, values: FloatArray) -> None:
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"Non-finite intermediate {name}; arithmetic overflow likely."
            )

    @staticmethod
    def _sigmoid(x: FloatArray) -> FloatArray:
        """Branch-safe sigmoid.

        Masked assignment is used instead of ``np.where`` because NumPy may
        evaluate both branches of ``np.where`` before selecting, which can
        overflow ``exp`` on the discarded branch. Only the stable branch is
        evaluated per element here.
        """
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)

        positive = x >= 0.0
        negative = ~positive

        if np.any(positive):
            xp = x[positive]
            out[positive] = 1.0 / (1.0 + np.exp(-xp))

        if np.any(negative):
            xn = x[negative]
            exp_xn = np.exp(xn)
            out[negative] = exp_xn / (1.0 + exp_xn)

        return out

    def compute_loss(
        self,
        policy_chosen_logps: npt.ArrayLike,
        policy_rejected_logps: npt.ArrayLike,
        ref_chosen_logps: npt.ArrayLike,
        ref_rejected_logps: npt.ArrayLike,
    ) -> tuple[float, DPOMetrics]:
        """Compute DPO loss, metrics, and policy-log-probability gradients.

        All four inputs are converted to float64 and must be 1-D, equal-length,
        non-empty, and finite. `beta` is validated at construction time and is
        not re-validated here.

        Returns
        -------
        loss : float
            Batch-mean of ``logaddexp(0, -beta * reward_margin)``.
        metrics : DPOMetrics
            Typed dictionary. All arrays are float64. The two gradient arrays
            are derivatives of the batch-mean scalar loss with respect to the
            *supplied* policy sequence log-probabilities, not model parameters.

        Raises
        ------
        ValueError
            On non-convertible inputs, wrong dimensionality, empty batch,
            mismatched shapes, or non-finite input values.
        FloatingPointError
            If any intermediate or returned array becomes non-finite due to
            arithmetic overflow within float64.
        """
        p_chosen = self._validate_batch("policy_chosen_logps", policy_chosen_logps)
        p_rejected = self._validate_batch("policy_rejected_logps", policy_rejected_logps)
        r_chosen = self._validate_batch("ref_chosen_logps", ref_chosen_logps)
        r_rejected = self._validate_batch("ref_rejected_logps", ref_rejected_logps)

        if not (
            p_chosen.shape == p_rejected.shape == r_chosen.shape == r_rejected.shape
        ):
            raise ValueError("All log-probability arrays must have identical shapes.")

        # Intermediate arithmetic is evaluated under errstate so that expected
        # float overflow does not emit RuntimeWarnings on stderr. Explicit
        # finiteness checks below are the failure signal.
        with np.errstate(over="ignore", invalid="ignore"):
            chosen_rewards = p_chosen - r_chosen
            rejected_rewards = p_rejected - r_rejected
            reward_margin = chosen_rewards - rejected_rewards
            logits = self.beta * reward_margin

            self._require_finite("chosen_rewards", chosen_rewards)
            self._require_finite("rejected_rewards", rejected_rewards)
            self._require_finite("reward_margin", reward_margin)
            self._require_finite("logits", logits)

            # Stable softplus: log(1 + exp(-z)) == logaddexp(0, -z).
            # No clipping of z; clipping would change the objective for large |z|.
            losses = np.logaddexp(0.0, -logits)
            self._require_finite("per-example losses", losses)

        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise FloatingPointError("Non-finite scalar DPO loss.")

        batch_size = p_chosen.shape[0]
        sigmoid_neg_logits = self._sigmoid(-logits)
        self._require_finite("sigmoid(-logits)", sigmoid_neg_logits)

        grad_policy_chosen = (-self.beta / batch_size) * sigmoid_neg_logits
        grad_policy_rejected = (self.beta / batch_size) * sigmoid_neg_logits

        self._require_finite("grad_policy_chosen", grad_policy_chosen)
        self._require_finite("grad_policy_rejected", grad_policy_rejected)

        reward_accuracy = float(np.mean(reward_margin > 0.0))

        metrics: DPOMetrics = {
            "loss": mean_loss,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "reward_margin": reward_margin,
            "reward_accuracy": reward_accuracy,
            "grad_policy_chosen": grad_policy_chosen,
            "grad_policy_rejected": grad_policy_rejected,
        }

        return mean_loss, metrics


__all__ = ["DPOLossKernel", "DPOMetrics"]