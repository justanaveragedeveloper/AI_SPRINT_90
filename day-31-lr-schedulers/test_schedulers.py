"""
Unit tests for the learning rate schedulers.
All tests are written with descriptive names and inline comments.
"""

import math

import pytest
from schedulers import (
    BaseLRScheduler,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
    WarmupCosineLR,
)

# -----------------------------------------------------------------------------
# Mock Optimizer for testing
# -----------------------------------------------------------------------------


class MockOptimizer:
    """Simple optimizer stub that only stores a learning rate."""

    def __init__(self, lr=0.1):
        self.lr = lr


# -----------------------------------------------------------------------------
# StepLR tests
# -----------------------------------------------------------------------------


def test_steplr_boundaries():
    """Check exact values at transition boundaries."""
    opt = MockOptimizer(0.1)
    scheduler = StepLR(opt, step_size=2, gamma=0.5)

    # Expected sequence: 0.1, 0.1, 0.05, 0.05, 0.025, 0.025
    expected = [0.1, 0.1, 0.05, 0.05, 0.025, 0.025]
    for i, expected_lr in enumerate(expected):
        if i:
            scheduler.step()
        assert math.isclose(opt.lr, expected_lr, abs_tol=1e-12)


def test_steplr_step_size_one():
    """When step_size=1, decay happens every epoch."""
    opt = MockOptimizer(0.1)
    scheduler = StepLR(opt, step_size=1, gamma=0.5)
    expected = [0.1, 0.05, 0.025, 0.0125]
    for i, expected_lr in enumerate(expected):
        if i:
            scheduler.step()
        assert math.isclose(opt.lr, expected_lr, abs_tol=1e-12)


def test_steplr_explicit_epoch():
    """Setting a specific epoch via step(epoch=...) works correctly."""
    opt = MockOptimizer(0.1)
    scheduler = StepLR(opt, step_size=2, gamma=0.5)
    scheduler.step(epoch=4)  # 4//2 = 2 → 0.1 * 0.5² = 0.025
    assert scheduler.last_epoch == 4
    assert math.isclose(opt.lr, 0.025, abs_tol=1e-12)


def test_steplr_validation():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        StepLR(MockOptimizer(), step_size=0)

    with pytest.raises(ValueError):
        StepLR(MockOptimizer(), step_size=1, gamma=0.0)  # gamma must be > 0

    with pytest.raises(ValueError):
        StepLR(MockOptimizer(), step_size=1, gamma=1.1)  # gamma must be ≤ 1


# -----------------------------------------------------------------------------
# ExponentialLR tests
# -----------------------------------------------------------------------------


def test_exponentiallr_formula():
    """Check exact exponential decay values."""
    opt = MockOptimizer(1.0)
    scheduler = ExponentialLR(opt, gamma=0.9)
    expected = [1.0, 0.9, 0.81, 0.729]
    for i, expected_lr in enumerate(expected):
        if i:
            scheduler.step()
        assert math.isclose(opt.lr, expected_lr, abs_tol=1e-12)


def test_exponentiallr_gamma_one():
    """gamma=1.0 means no decay."""
    opt = MockOptimizer(0.1)
    scheduler = ExponentialLR(opt, gamma=1.0)
    for _ in range(5):
        scheduler.step()
    assert math.isclose(opt.lr, 0.1)


def test_exponentiallr_uses_base_lr():
    """Ensure the scheduler does not compound from the current LR."""
    opt = MockOptimizer(1.0)
    scheduler = ExponentialLR(opt, gamma=0.9)

    # Advance 3 epochs → LR = 0.9³ = 0.729
    for _ in range(3):
        scheduler.step()

    # Manually change the optimizer's LR – this should NOT affect the scheduler.
    opt.lr = 0.5
    scheduler.step()  # should compute 0.9⁴ from base_lr, not from 0.5
    assert math.isclose(opt.lr, 0.9**4, abs_tol=1e-12)


def test_exponentiallr_validation():
    """Gamma must be in (0, 1]."""
    with pytest.raises(ValueError):
        ExponentialLR(MockOptimizer(), gamma=0.0)
    with pytest.raises(ValueError):
        ExponentialLR(MockOptimizer(), gamma=1.1)


# -----------------------------------------------------------------------------
# CosineAnnealingLR tests
# -----------------------------------------------------------------------------


def test_cosine_endpoints():
    """Check start, midpoint, and end of the cosine cycle."""
    opt = MockOptimizer(0.1)
    scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.01)

    assert math.isclose(opt.lr, 0.1)  # t=0 → η_max

    scheduler.step(epoch=5)  # midpoint
    assert math.isclose(opt.lr, 0.055, abs_tol=1e-12)

    scheduler.step(epoch=10)  # end → η_min
    assert math.isclose(opt.lr, 0.01, abs_tol=1e-12)


def test_cosine_after_tmax():
    """After T_max, LR stays at eta_min."""
    opt = MockOptimizer(0.1)
    scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.01)
    scheduler.step(epoch=100)
    assert math.isclose(opt.lr, 0.01)


def test_cosine_tmax_one():
    """When T_max=1, LR drops to eta_min after one step."""
    opt = MockOptimizer(0.1)
    scheduler = CosineAnnealingLR(opt, T_max=1, eta_min=0.0)
    assert math.isclose(opt.lr, 0.1)
    scheduler.step()
    assert math.isclose(opt.lr, 0.0)


def test_cosine_validation():
    """Invalid arguments raise appropriate errors."""
    with pytest.raises(ValueError):
        CosineAnnealingLR(MockOptimizer(), T_max=0)

    with pytest.raises(ValueError):
        CosineAnnealingLR(MockOptimizer(), T_max=10, eta_min=-0.1)

    with pytest.raises(ValueError):
        CosineAnnealingLR(MockOptimizer(0.1), T_max=10, eta_min=0.2)  # > base_lr


# -----------------------------------------------------------------------------
# WarmupCosineLR tests
# -----------------------------------------------------------------------------


def test_warmup_values():
    """Check linear warmup values."""
    opt = MockOptimizer(0.1)
    scheduler = WarmupCosineLR(opt, warmup_steps=5, total_steps=15, eta_min=0.0)
    expected = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]  # t=0..5
    for i, expected_lr in enumerate(expected):
        if i:
            scheduler.step()
        assert math.isclose(opt.lr, expected_lr, abs_tol=1e-12)


def test_warmup_cosine_transition():
    """Ensure smooth transition between warmup and cosine phases."""
    opt = MockOptimizer(0.1)
    scheduler = WarmupCosineLR(opt, warmup_steps=5, total_steps=15, eta_min=0.0)

    # At t=5, warmup ends and cosine begins with same value.
    scheduler.step(epoch=5)
    assert math.isclose(opt.lr, 0.1, abs_tol=1e-12)

    # Next step (t=6) follows cosine.
    scheduler.step()
    expected = 0.05 * (1 + math.cos(math.pi / 10))
    assert math.isclose(opt.lr, expected, abs_tol=1e-12)


def test_warmup_endpoint():
    """At total_steps, LR reaches eta_min and stays there."""
    opt = MockOptimizer(0.1)
    scheduler = WarmupCosineLR(opt, warmup_steps=5, total_steps=15, eta_min=0.01)
    scheduler.step(epoch=15)
    assert math.isclose(opt.lr, 0.01, abs_tol=1e-12)
    scheduler.step()  # beyond total_steps
    assert math.isclose(opt.lr, 0.01, abs_tol=1e-12)


def test_warmup_zero_steps():
    """warmup_steps=0 means immediate cosine decay."""
    opt = MockOptimizer(0.1)
    scheduler = WarmupCosineLR(opt, warmup_steps=0, total_steps=10, eta_min=0.01)
    assert math.isclose(opt.lr, 0.1)  # t=0 → base_lr

    scheduler.step()
    expected = 0.01 + 0.5 * 0.09 * (1 + math.cos(math.pi * 0.1))
    assert math.isclose(opt.lr, expected, abs_tol=1e-12)


def test_warmup_validation():
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        WarmupCosineLR(MockOptimizer(), warmup_steps=-1, total_steps=10)

    with pytest.raises(ValueError):
        WarmupCosineLR(MockOptimizer(), warmup_steps=10, total_steps=10)  # not greater

    with pytest.raises(ValueError):
        WarmupCosineLR(MockOptimizer(0.1), warmup_steps=5, total_steps=10, eta_min=0.2)


# -----------------------------------------------------------------------------
# ReduceLROnPlateau tests
# -----------------------------------------------------------------------------


def test_plateau_patience_boundary():
    """Reduction occurs after patience+1 bad epochs."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    scheduler.step(1.0)  # best
    scheduler.step(1.1)  # bad=1
    assert math.isclose(opt.lr, 0.1)

    scheduler.step(1.2)  # bad=2
    assert math.isclose(opt.lr, 0.1)

    scheduler.step(1.3)  # bad=3 > patience → reduce
    assert math.isclose(opt.lr, 0.05)
    assert scheduler.consecutive_bad_epochs == 0  # counter reset


def test_plateau_patience_zero():
    """With patience=0, the first bad epoch triggers reduction."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(opt, patience=0, factor=0.5)

    scheduler.step(1.0)
    scheduler.step(1.1)  # bad=1 > 0 → reduce
    assert math.isclose(opt.lr, 0.05)


def test_plateau_improvement_resets():
    """An improvement resets the bad‑epoch counter."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(opt, patience=2, factor=0.5)

    scheduler.step(1.0)
    scheduler.step(1.1)  # bad=1
    scheduler.step(0.9)  # improvement → reset
    assert scheduler.consecutive_bad_epochs == 0
    assert scheduler.best_metric == 0.9


def test_plateau_threshold():
    """Small improvements within threshold are not counted as improvement."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(
        opt, mode="min", patience=1, threshold=0.01, factor=0.5
    )

    scheduler.step(1.0)
    # 0.995 is not < 0.99 → bad=1 (patience is 1, so no reduction yet)
    scheduler.step(0.995)
    assert math.isclose(opt.lr, 0.1)

    # 0.99 is also not < 0.99 → bad=2 > patience → reduce
    scheduler.step(0.99)
    assert math.isclose(opt.lr, 0.05)


def test_plateau_equality_not_improvement():
    """Exact equality (within threshold=0) is not an improvement."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(
        opt, mode="min", patience=1, threshold=0.0, factor=0.5
    )

    scheduler.step(1.0)
    scheduler.step(1.0)  # equality → bad=1
    assert math.isclose(opt.lr, 0.1)

    scheduler.step(1.0)  # bad=2 > patience → reduce
    assert math.isclose(opt.lr, 0.05)


def test_plateau_min_lr_floor():
    """LR is never reduced below min_lr."""
    opt = MockOptimizer(0.1)
    scheduler = ReduceLROnPlateau(opt, patience=1, factor=0.1, min_lr=0.01)

    scheduler.step(1.0)
    scheduler.step(1.1)  # bad=1 (patience=1, so no reduction yet)
    assert math.isclose(opt.lr, 0.1)

    scheduler.step(1.2)  # bad=2 > 1 → reduce to max(0.1*0.1, 0.01) = 0.01
    assert math.isclose(opt.lr, 0.01)

    scheduler.step(1.3)  # further bad epochs keep LR at min_lr
    assert math.isclose(opt.lr, 0.01)


def test_plateau_validation():
    """Invalid constructor arguments raise ValueError."""
    opt = MockOptimizer(0.1)

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, mode="invalid")

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, factor=0.0)

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, factor=1.0)

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, patience=-1)

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, threshold=-0.1)

    with pytest.raises(ValueError):
        ReduceLROnPlateau(opt, min_lr=0.2)  # > base_lr


# -----------------------------------------------------------------------------
# Validation for NaNs and Infs
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_lr", [0.0, -0.1, math.nan, math.inf, -math.inf])
def test_optimizer_lr_must_be_positive_finite(bad_lr):
    """Optimizer LR must be a finite positive number."""
    with pytest.raises((TypeError, ValueError)):
        StepLR(MockOptimizer(bad_lr), step_size=1)


@pytest.mark.parametrize("bad_metric", [math.nan, math.inf, -math.inf])
def test_plateau_rejects_nonfinite_metric(bad_metric):
    """ReduceLROnPlateau must reject non‑finite metrics."""
    scheduler = ReduceLROnPlateau(MockOptimizer())
    with pytest.raises(ValueError):
        scheduler.step(bad_metric)


def test_reject_boolean_as_numeric():
    """Boolean values are not valid numeric hyperparameters."""
    with pytest.raises(TypeError):
        StepLR(MockOptimizer(0.1), step_size=1, gamma=True)

    with pytest.raises(TypeError):
        ExponentialLR(MockOptimizer(0.1), gamma=False)


# -----------------------------------------------------------------------------
# Base scheduler tests
# -----------------------------------------------------------------------------


def test_base_scheduler_constructor_semantics():
    """Verify that last_epoch=-1 initialises at epoch 0, and explicit epoch works."""

    class DummyScheduler(BaseLRScheduler):
        def get_lr(self):
            return self.base_lr

    opt = MockOptimizer(0.1)

    # Default behaviour: last_epoch = -1 → auto‑step to epoch 0
    scheduler = DummyScheduler(opt)
    assert scheduler.last_epoch == 0
    assert opt.lr == 0.1

    # Explicit last_epoch → directly set LR for that epoch
    scheduler = DummyScheduler(opt, last_epoch=5)
    assert scheduler.last_epoch == 5
    assert opt.lr == 0.1

    scheduler.step()
    assert scheduler.last_epoch == 6


def test_explicit_epoch_must_be_non_negative():
    """Calling step(epoch) with a negative value should raise an error."""
    scheduler = StepLR(MockOptimizer(), step_size=1)
    with pytest.raises(ValueError):
        scheduler.step(epoch=-1)


def test_invalid_optimizer():
    """Optimizer must have an 'lr' attribute."""

    class BadOpt:
        pass

    with pytest.raises(AttributeError):
        StepLR(BadOpt(), step_size=1)


def test_scheduler_does_not_corrupt_optimizer_state():
    """Schedulers should only modify the 'lr' attribute of the optimizer."""
    opt = MockOptimizer(0.1)
    opt.momentum = 0.9  # extra attribute
    scheduler = StepLR(opt, step_size=1)
    scheduler.step()
    assert opt.momentum == 0.9  # unchanged


# -----------------------------------------------------------------------------
# Full Optimizer Integration Test
# -----------------------------------------------------------------------------


def test_scheduler_controls_optimizer_update():
    """
    Verify that changing optimizer.lr via the scheduler actually affects
    the parameter updates performed by the optimizer.
    """

    class SimpleOptimizer:
        def __init__(self):
            self.lr = 0.1
            self.parameter = 1.0
            self.grad = 1.0

        def step(self):
            # Simple SGD update: parameter -= lr * gradient
            self.parameter -= self.lr * self.grad

    opt = SimpleOptimizer()
    scheduler = StepLR(opt, step_size=1, gamma=0.5)

    # First update with LR=0.1
    opt.step()
    assert math.isclose(opt.parameter, 0.9)

    # Scheduler steps to epoch 1 → LR becomes 0.05
    scheduler.step()
    opt.step()
    assert math.isclose(opt.parameter, 0.85)

    # Scheduler steps to epoch 2 → LR becomes 0.025
    scheduler.step()
    opt.step()
    assert math.isclose(opt.parameter, 0.825)
