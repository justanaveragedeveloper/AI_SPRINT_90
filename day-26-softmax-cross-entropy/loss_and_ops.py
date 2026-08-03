"""
loss_and_ops.py

Softmax activation and categorical cross‑entropy loss implemented on top of
the Day 24 autograd engine. All operations preserve the computational graph
and are numerically stable.
"""

import logging
import os
import sys

# Ensure Day 24 engine is importable (assumes sibling directory)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../day-24-autograd"))
)
from engine import Value  # type: ignore

logger = logging.getLogger(__name__)


def softmax(logits: list[Value]) -> list[Value]:
    """
    Compute numerically stable softmax probabilities over a list of Value logits.

    For each logit z_i:
        p_i = exp(z_i - max(logits)) / sum_j exp(z_j - max(logits))

    Args:
        logits: List of Value objects representing raw scores (C classes).

    Returns:
        List of Value probabilities, each in [0,1] and sum ≈ 1.0.

    Raises:
        ValueError: if logits is empty.
        TypeError: if any element is not a Value.
    """
    if not logits:
        raise ValueError("softmax: logits list cannot be empty")

    for z in logits:
        if not isinstance(z, Value):
            raise TypeError(f"softmax: expected Value, got {type(z).__name__}")

    # Numerical stabilisation: subtract max(logits)
    max_val = max(z.value for z in logits)  # Python float
    max_val_value = Value(max_val)  # wrap as Value
    # Use addition of negative value because __sub__ may not be implemented
    exps = [(z - max_val_value).exp() for z in logits]  # Value + Value
    total_exp = sum(exps)  # Value sum

    probs = [exp_val / total_exp for exp_val in exps]
    return probs


def categorical_cross_entropy(
    probs: list[Value], target_idx: int, epsilon: float = 1e-15
) -> Value:
    """
    Compute categorical cross‑entropy loss: L = -log(p_target + epsilon_protection).

    Args:
        probs: List of Value probabilities (output of softmax).
        target_idx: Integer index of the true class (0‑based).
        epsilon: Small float to avoid log(0) when p_target is extremely small.

    Returns:
        Value representing the scalar loss.

    Raises:
        ValueError: if probs is empty or target_idx is out of bounds.
        TypeError: if elements of probs are not Value.
    """
    if not probs:
        raise ValueError("categorical_cross_entropy: probs list cannot be empty")

    if target_idx < 0 or target_idx >= len(probs):
        raise IndexError(
            f"categorical_cross_entropy: target_idx {target_idx} out of range "
            f"[0, {len(probs)-1}]"
        )

    for p in probs:
        if not isinstance(p, Value):
            raise TypeError(
                f"categorical_cross_entropy: expected Value, got {type(p).__name__}"
            )

    p_target = probs[target_idx]

    # Epsilon protection: if probability is too close to zero, add epsilon.
    if p_target.value < epsilon:
        logger.warning(
            f"Target probability {p_target.value:.3e} below epsilon={epsilon}; "
            "adding epsilon for numerical safety."
        )
        eps_value = Value(epsilon)  # wrap epsilon as Value
        p_safe = p_target + eps_value  # Value + Value
    else:
        p_safe = p_target

    loss = -p_safe.log()
    return loss
