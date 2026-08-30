"""
Day 38: Loss Functions and Activations from First Principles.

This module implements the core components needed for a neural network's
decision-making layer:

1. Softmax activation (numerically stable via max-subtraction)
2. Categorical Cross-Entropy (CCE) loss with integrated gradient
   (using log-sum-exp for stability)
3. Mean Squared Error (MSE) for regression
4. Binary Cross-Entropy (BCE) for binary classification

All operations are vectorized using NumPy, with rigorous input validation,
numerical safeguards, and explicit backward closures.

The design follows the autograd pattern used in earlier days of the sprint:
    loss, backward = criterion.forward(logits, targets)
    grad = backward()   # dL/dlogits

This gradient can then be passed back through the CNN layers.
"""

import logging
from collections.abc import Callable

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper function for input validation (reduces code duplication)
# -----------------------------------------------------------------------------
def _validate_array(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Validate that `value` is a finite NumPy array.

    Args:
        value: The object to validate.
        name: Name of the variable (for error messages).
        allow_empty: If False, raise ValueError when size == 0.

    Returns:
        The validated array (unchanged).

    Raises:
        TypeError: If value is not a NumPy array.
        ValueError: If array is empty (and allow_empty=False) or contains inf/nan.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if not allow_empty and value.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains inf or nan")
    return value


# -----------------------------------------------------------------------------
# 1. Numerically Stable Softmax
# -----------------------------------------------------------------------------
def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of the input along the specified axis.

    Softmax converts logits into probabilities:
        p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))

    The max-subtraction prevents overflow for large positive logits.

    Args:
        logits: Input array (1D or ND). Must be non-empty and finite.
        axis: Axis along which to apply softmax. Default is the last axis.

    Returns:
        Probabilities with the same shape as logits, summing to 1 along `axis`.

    Raises:
        TypeError: If logits is not a numpy array.
        ValueError: If logits is empty, contains inf/nan, or axis is out of bounds.
    """
    logits = _validate_array(logits, "logits")

    # Ensure at least one dimension exists
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")

    # Validate axis
    if not -logits.ndim <= axis < logits.ndim:
        raise ValueError(
            f"axis {axis} out of bounds for array with {logits.ndim} dimensions"
        )

    # Max subtraction for numerical stability
    max_vals = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_vals
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals, axis=axis, keepdims=True)
    probs = exp_vals / sum_exp

    logger.debug(f"Softmax computed on shape {logits.shape}, axis={axis}")
    return probs


# -----------------------------------------------------------------------------
# 2. Categorical Cross‑Entropy Loss (Multiclass Classification)
# -----------------------------------------------------------------------------
class CrossEntropyLoss:
    """
    Categorical Cross‑Entropy loss with integrated Softmax gradient.

    This class combines the Softmax activation and the negative log-likelihood
    loss into one forward/backward pass. It uses the log‑sum‑exp trick for
    numerical stability and returns the direct gradient:
        dL/dz = (p - y) / N   (mean reduction over batch)

    This avoids the need to compute the full Jacobian of Softmax.
    """

    def forward(
        self, logits: np.ndarray, targets: int | np.ndarray
    ) -> tuple[float, Callable[[], np.ndarray]]:
        """
        Compute the mean CCE loss and return a closure for the gradient.

        Args:
            logits: Raw scores from the network. Shape (K,) for a single sample,
                    or (N, K) for a batch of N samples.
            targets: Target class index(es). For single sample, an integer.
                     For batch, a 1D array of integers of length N.

        Returns:
            loss: Scalar loss (mean over batch).
            backward: A callable that returns the gradient dL/dlogits,
                      with the same shape as the input logits.

        Raises:
            TypeError: If inputs have invalid types.
            ValueError: If logits are empty, contain inf/nan, targets are out of range,
                        or shapes are incompatible.
        """
        # --- Validate logits ---
        logits = _validate_array(logits, "logits")

        # --- Handle single sample vs batch ---
        if logits.ndim == 1:
            K = logits.shape[0]
            if not isinstance(targets, (int, np.integer)):
                raise TypeError("target must be an integer")
            if not (0 <= targets < K):
                raise ValueError(f"target index {targets} out of range [0, {K-1}]")
            # Reshape to (1, K) for unified batch processing
            logits = logits.reshape(1, -1)
            targets = np.array([targets], dtype=np.int64)
            single = True

        elif logits.ndim == 2:
            N, K = logits.shape
            if not isinstance(targets, np.ndarray) or targets.ndim != 1 or targets.shape[0] != N:
                raise ValueError("targets must be a 1D array of length batch size")
            if targets.dtype.kind not in "iu":
                raise TypeError("targets must be integer indices")
            if not np.all((0 <= targets) & (targets < K)):
                raise ValueError(f"target indices out of range [0, {K-1}]")
            single = False

        else:
            raise ValueError("logits must be 1D or 2D")

        N, K = logits.shape

        # --- Forward pass: log-sum-exp for numerical stability ---
        # Formula: loss = -z_y + log(sum_j exp(z_j))
        # We compute max for stability: log(sum exp(z)) = max + log(sum exp(z - max))
        max_vals = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_vals
        sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True)
        log_sum_exp = max_vals + np.log(sum_exp)  # shape (N, 1)

        # Extract logit for the target class
        z_target = logits[np.arange(N), targets]  # (N,)

        # Loss per sample = - (z_target - log_sum_exp)
        losses = -(z_target - log_sum_exp.reshape(N))
        loss = float(np.mean(losses))

        logger.debug(f"CCE forward: batch_size={N}, K={K}, loss={loss:.6f}")

        # --- Precompute probabilities for gradient ---
        probs = np.exp(shifted) / sum_exp  # (N, K)

        # --- Backward closure ---
        def backward() -> np.ndarray:
            """
            Return dL/dlogits.

            Since loss = mean over batch, gradient is (p - y) / N.
            For single sample (N=1), this reduces to p - y.
            """
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), targets] = 1.0
            grad = (probs - one_hot) / N
            if single:
                grad = grad.reshape(-1)  # back to 1D
            return grad

        return loss, backward


# -----------------------------------------------------------------------------
# 3. Mean Squared Error Loss (Regression)
# -----------------------------------------------------------------------------
class MSELoss:
    """
    Mean Squared Error loss for regression tasks.

    Reduction is mean over all elements (i.e., over both batch and feature dimensions).
    """

    def forward(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> tuple[float, Callable[[], np.ndarray]]:
        """
        Compute MSE and return a closure for the gradient.

        Args:
            predictions: Predicted values (any shape).
            targets: Target values (same shape as predictions).

        Returns:
            loss: Scalar MSE = mean((pred - target)^2).
            backward: Callable that returns dMSE/dpredictions (same shape).

        Raises:
            ValueError: If inputs are empty, contain inf/nan, or shapes mismatch.
        """
        predictions = _validate_array(predictions, "predictions")
        targets = _validate_array(targets, "targets")

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {predictions.shape} vs {targets.shape}"
            )

        diff = predictions - targets
        loss = float(np.mean(diff ** 2))
        N = predictions.size  # total number of elements

        def backward() -> np.ndarray:
            # dMSE/dpred = 2*(pred - target) / N
            return 2.0 * diff / N

        return loss, backward


# -----------------------------------------------------------------------------
# 4. Binary Cross‑Entropy Loss (Binary Classification)
# -----------------------------------------------------------------------------
class BCELoss:
    """
    Binary Cross‑Entropy loss with probability inputs.

    This loss expects probabilities (not logits) as input, and targets in [0,1].
    It clamps probabilities to [eps, 1‑eps] to prevent log(0) and log(1).

    The backward gradient is the derivative of the composite function
    BCE(clip(pred_prob)), so for inputs that are clipped (outside [eps, 1-eps]),
    the gradient is zero. This is mathematically consistent with the forward pass.
    """

    def __init__(self, eps: float = 1e-15):
        """
        Args:
            eps: Small value used for clamping probabilities.
                 Must be in (0, 0.5).
        """
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must be in (0, 0.5)")
        self.eps = eps

    def forward(
        self, pred_prob: np.ndarray, target: np.ndarray
    ) -> tuple[float, Callable[[], np.ndarray]]:
        """
        Compute mean BCE and return a closure for the gradient.

        Args:
            pred_prob: Predicted probabilities, in [0,1] (any shape).
            target: Binary targets, also in [0,1] (same shape).

        Returns:
            loss: Scalar BCE = -mean(y*log(p) + (1-y)*log(1-p)).
            backward: Callable returning dBCE/dpred_prob.

        Raises:
            ValueError: If inputs are empty, contain inf/nan, shapes mismatch,
                        or values are outside [0,1].
        """
        pred_prob = _validate_array(pred_prob, "pred_prob")
        target = _validate_array(target, "target")

        if pred_prob.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: {pred_prob.shape} vs {target.shape}"
            )

        if np.any((pred_prob < 0.0) | (pred_prob > 1.0)):
            raise ValueError("pred_prob values must be in [0,1]")
        if np.any((target < 0.0) | (target > 1.0)):
            raise ValueError("target values must be in [0,1]")

        # Clip to avoid log(0)
        p = np.clip(pred_prob, self.eps, 1.0 - self.eps)

        # BCE loss
        loss = -np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
        N = pred_prob.size

        def backward() -> np.ndarray:
            # Derivative of BCE w.r.t. the clipped probability p:
            #   dBCE/dp = (p - y) / (p*(1-p)) / N
            denom = p * (1.0 - p)
            grad = (p - target) / denom / N

            # Multiply by derivative of the clip function:
            # derivative = 1 where eps < pred_prob < 1-eps, else 0.
            mask = (pred_prob > self.eps) & (pred_prob < 1.0 - self.eps)
            grad = grad * mask
            return grad

        return float(loss), backward