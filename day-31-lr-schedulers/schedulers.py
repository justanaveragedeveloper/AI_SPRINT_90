"""
Day 31: Learning Rate Schedulers and Adaptive Decay Mechanics

This module provides production‑ready learning rate schedulers that work with
any optimizer exposing a mutable `lr` attribute. All hyperparameters are
validated for finiteness and range.

The schedulers implement:
  - Step decay (StepLR)
  - Exponential decay (ExponentialLR)
  - Cosine annealing (CosineAnnealingLR)
  - Warmup + cosine decay (WarmupCosineLR)
  - Reduce on plateau (ReduceLROnPlateau)

Each scheduler maintains its own state and updates the optimizer's learning rate
via the `step()` method.
"""

import math
from typing import Optional

# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _validate_finite_non_negative(
    name: str,
    value: object,
    *,
    allow_zero: bool = True,
) -> float:
    """
    Ensure `value` is a finite number and, if `allow_zero` is True, non‑negative.

    Raises:
        TypeError: if value is a boolean or not numeric.
        ValueError: if value is infinite, NaN, or out of the allowed range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric value.")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    if allow_zero:
        if numeric_value < 0:
            raise ValueError(f"{name} must be non‑negative.")
    else:
        if numeric_value <= 0:
            raise ValueError(f"{name} must be positive.")

    return numeric_value


def _validate_positive_integer(
    name: str,
    value: object,
    *,
    minimum: int = 1,
) -> int:
    """
    Ensure `value` is an integer >= `minimum`.

    Raises:
        TypeError: if value is a boolean or not an integer.
        ValueError: if value is less than `minimum`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non‑negative.")
        raise ValueError(f"{name} must be positive.")

    return value


def _validate_optimizer(optimizer) -> float:
    """
    Verify that the optimizer has a numeric `lr` attribute and return its value.
    """
    if not hasattr(optimizer, "lr"):
        raise AttributeError("Optimizer must have an 'lr' attribute.")

    return _validate_finite_non_negative(
        "Optimizer.lr",
        optimizer.lr,
        allow_zero=False,  # learning rate must be > 0
    )


# -----------------------------------------------------------------------------
# Base Scheduler
# -----------------------------------------------------------------------------


class BaseLRScheduler:
    """
    Abstract base class for all epoch‑based learning rate schedulers.

    Semantics:
        - `last_epoch = -1` (default): the constructor initialises the scheduler
          at epoch 0 (i.e., the first `step()` call will move to epoch 1).
        - `last_epoch >= 0`: the constructor sets the learning rate for that
          specific epoch without advancing the counter.

    After construction, calling `step()` advances the epoch by one and updates
    the optimizer's learning rate.
    """

    def __init__(self, optimizer, last_epoch: int = -1):
        # Capture and validate the initial learning rate.
        self.base_lr = _validate_optimizer(optimizer)

        # Store the current epoch counter.
        self.last_epoch = _validate_positive_integer(
            "last_epoch",
            last_epoch,
            minimum=-1,  # allow -1 as a special "before epoch 0" value
        )

        self.optimizer = optimizer

        # If last_epoch == -1, we treat this as "start at epoch 0".
        if self.last_epoch == -1:
            self.step()
        else:
            # Otherwise, set the LR directly for the given epoch.
            self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        """Compute the learning rate for the current epoch."""
        raise NotImplementedError("Subclasses must implement get_lr().")

    def step(self, epoch: Optional[int] = None) -> float:
        """
        Advance the scheduler by one epoch (or set a specific epoch) and
        update the optimizer's learning rate.

        Args:
            epoch: Optional explicit epoch number (must be >= 0). If not given,
                   the internal counter is incremented by 1.

        Returns:
            The new learning rate.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = _validate_positive_integer(
                "epoch",
                epoch,
                minimum=0,
            )

        new_lr = self.get_lr()

        # Guard against numerical errors (should never happen with correct inputs).
        if not math.isfinite(new_lr) or new_lr < 0:
            raise RuntimeError("Scheduler produced an invalid learning rate.")

        self.optimizer.lr = new_lr
        return new_lr


# -----------------------------------------------------------------------------
# Step Decay Scheduler
# -----------------------------------------------------------------------------


class StepLR(BaseLRScheduler):
    """
    Step decay scheduler.

    Formula:
        η_t = η_0 * γ^(⌊t / step_size⌋)

    where:
        η_0 = base learning rate
        γ   = decay factor (0 < γ ≤ 1)
        t   = current epoch
        step_size = number of epochs between drops
    """

    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.step_size = _validate_positive_integer("step_size", step_size, minimum=1)
        self.gamma = _validate_finite_non_negative("gamma", gamma, allow_zero=True)

        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0.0, 1.0].")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        # Determine how many full intervals have passed.
        intervals = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma**intervals)


# -----------------------------------------------------------------------------
# Exponential Decay Scheduler
# -----------------------------------------------------------------------------


class ExponentialLR(BaseLRScheduler):
    """
    Exponential decay scheduler.

    Formula:
        η_t = η_0 * γ^t

    where:
        η_0 = base learning rate
        γ   = decay factor (0 < γ ≤ 1)
        t   = current epoch

    Important: The scheduler always computes the LR from the base learning rate,
               never compounding from the current LR.
    """

    def __init__(
        self,
        optimizer,
        gamma: float,
        last_epoch: int = -1,
    ):
        self.gamma = _validate_finite_non_negative("gamma", gamma, allow_zero=True)

        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0.0, 1.0].")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma**self.last_epoch)


# -----------------------------------------------------------------------------
# Cosine Annealing Scheduler
# -----------------------------------------------------------------------------


class CosineAnnealingLR(BaseLRScheduler):
    """
    Cosine annealing scheduler.

    Formula:
        η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T_max))

    where:
        η_max = base learning rate
        η_min = minimum learning rate (default 0)
        T_max = total number of epochs for the annealing
        t     = current epoch

    After T_max, the learning rate stays at η_min.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_max = _validate_positive_integer("T_max", T_max, minimum=1)
        self.eta_min = _validate_finite_non_negative(
            "eta_min", eta_min, allow_zero=True
        )

        # Check that eta_min does not exceed the base LR.
        base_lr = _validate_optimizer(optimizer)
        if self.eta_min > base_lr:
            raise ValueError("eta_min must be <= base_lr.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        # Clamp the epoch to T_max to keep LR stable after the cycle.
        clamped_epoch = min(self.last_epoch, self.T_max)

        cosine_term = 1.0 + math.cos(math.pi * clamped_epoch / self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * cosine_term


# -----------------------------------------------------------------------------
# Warmup + Cosine Decay Scheduler
# -----------------------------------------------------------------------------


class WarmupCosineLR(BaseLRScheduler):
    """
    Warmup followed by cosine decay.

    Two phases:

    1. Warmup (t < warmup_steps):
           η_t = η_min + (η_max - η_min) * t / warmup_steps

    2. Cosine decay (t ≥ warmup_steps):
           progress = (t - warmup_steps) / (total_steps - warmup_steps)
           η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * progress))

    where:
        η_max = base learning rate
        η_min = minimum learning rate (default 0)
        warmup_steps = number of epochs for linear warmup
        total_steps = total epochs until the cosine cycle ends

    After total_steps, the learning rate stays at η_min.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = _validate_positive_integer(
            "warmup_steps",
            warmup_steps,
            minimum=0,  # allow zero warmup
        )

        self.total_steps = _validate_positive_integer(
            "total_steps",
            total_steps,
            minimum=1,
        )

        if self.total_steps <= self.warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps.")

        self.eta_min = _validate_finite_non_negative(
            "eta_min", eta_min, allow_zero=True
        )

        base_lr = _validate_optimizer(optimizer)
        if self.eta_min > base_lr:
            raise ValueError("eta_min must be <= base_lr.")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        # Clamp epoch to total_steps to keep LR stable afterwards.
        current_epoch = min(self.last_epoch, self.total_steps)

        if current_epoch < self.warmup_steps:
            # ---------- Warmup phase ----------
            # Avoid division by zero when warmup_steps == 0.
            if self.warmup_steps == 0:
                return self.base_lr

            linear_progress = current_epoch / self.warmup_steps
            return self.eta_min + linear_progress * (self.base_lr - self.eta_min)

        else:
            # ---------- Cosine decay phase ----------
            cosine_progress = (current_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            # Clamp progress to [0, 1] just in case.
            cosine_progress = min(1.0, max(0.0, cosine_progress))

            cosine_term = 1.0 + math.cos(math.pi * cosine_progress)
            return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * cosine_term


# -----------------------------------------------------------------------------
# Reduce on Plateau Scheduler (Metric‑Driven)
# -----------------------------------------------------------------------------


class ReduceLROnPlateau:
    """
    Reduce learning rate when a monitored metric stops improving.

    The scheduler tracks the best metric seen so far and counts how many
    consecutive epochs the metric has failed to improve.

    Improvement is defined as:
        mode='min': current < best - threshold
        mode='max': current > best + threshold

    When the number of bad epochs exceeds `patience`, the learning rate is
    scaled down by `factor` (capped at `min_lr`) and the bad‑epoch counter is
    reset.

    Reduction occurs after `patience+1` bad epochs (strictly greater than
    patience). This is consistent with PyTorch's implementation.
    """

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-8,
    ):
        # Validate optimizer and capture base LR.
        base_lr = _validate_optimizer(optimizer)

        # Validate mode.
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        # Validate factor.
        self.factor = _validate_finite_non_negative("factor", factor, allow_zero=True)
        if not (0.0 < self.factor < 1.0):
            raise ValueError("factor must be in (0, 1).")

        self.patience = _validate_positive_integer("patience", patience, minimum=0)
        self.threshold = _validate_finite_non_negative(
            "threshold",
            threshold,
            allow_zero=True,
        )
        self.min_lr = _validate_finite_non_negative("min_lr", min_lr, allow_zero=True)

        # Ensure we don't accidentally increase LR.
        if self.min_lr > base_lr:
            raise ValueError("min_lr must be <= optimizer.lr.")

        self.optimizer = optimizer

        # Initialise tracking state.
        self.best_metric = math.inf if mode == "min" else -math.inf
        self.consecutive_bad_epochs = 0
        self.last_lr = float(optimizer.lr)

    def step(self, metric: float) -> float:
        """
        Update the scheduler with a new metric value.

        Args:
            metric: The current validation metric (e.g., loss or accuracy).

        Returns:
            The current learning rate after any possible reduction.
        """
        # Ensure metric is finite.
        metric = _validate_finite_non_negative("metric", metric)

        if self._is_improvement(metric):
            # Improvement: update best metric and reset bad counter.
            self.best_metric = metric
            self.consecutive_bad_epochs = 0
        else:
            # No improvement: increment bad counter.
            self.consecutive_bad_epochs += 1

        # Check if patience has been exceeded.
        if self.consecutive_bad_epochs > self.patience:
            self._reduce_lr()
            self.consecutive_bad_epochs = 0  # reset after reduction

        self.last_lr = float(self.optimizer.lr)
        return self.last_lr

    def _is_improvement(self, current: float) -> bool:
        """
        Determine whether `current` is better than the best metric seen so far,
        respecting the threshold.
        """
        if self.mode == "min":
            return current < self.best_metric - self.threshold
        else:  # mode == "max"
            return current > self.best_metric + self.threshold

    def _reduce_lr(self) -> None:
        """
        Apply the reduction: new_lr = max(old_lr * factor, min_lr).
        """
        new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
        self.optimizer.lr = new_lr
