"""
Tests for the optimizers – checking they work as expected.

We test:
- Exact math (analytical tests) – we compute by hand and compare.
- State persistence (velocity, squared averages, moments) stays correct.
- Behaviour with zero gradients (state should still decay).
- Numerical stability (tiny/large gradients don't break things).
- Invalid hyperparameters are rejected.
- Convergence on a simple problem: f(x)=x².
"""

import math
import sys

import pytest

sys.path.append("../day-24-autograd")
sys.path.append("../day-25-nn-from-scratch")

from engine import Value  # our autograd scalar
from optimizers import SGD, RMSprop, Adam

# ---------- Helper ----------


def assert_close(a, b, tol=1e-6):
    """Check that two numbers (or lists of numbers) are nearly equal."""
    if isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            assert_close(x, y, tol)
    else:
        assert abs(a - b) < tol, f"Expected {b}, got {a}"


# ---------- SGD tests ----------


def test_sgd_vanilla():
    """Plain SGD: θ ← θ - lr * grad. Starting at 1.0 with grad=2.0 should become 0.8."""
    p = Value(1.0)
    p.grad = 2.0
    SGD([p], lr=0.1).step()
    assert_close(p.value, 0.8)


def test_sgd_momentum_update_and_persistence():
    """Check momentum velocity updates correctly and persists across steps."""
    p = Value(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)

    # Step 1: grad=2.0
    p.grad = 2.0
    opt.step()
    v1 = 0.2  # (1-0.9)*2
    assert_close(opt.v[0], v1)
    assert_close(p.value, 1.0 - 0.1 * v1)

    # Step 2: grad=3.0
    p.grad = 3.0
    opt.step()
    v2 = 0.9 * v1 + 0.1 * 3.0  # 0.9*0.2 + 0.3 = 0.48
    assert_close(opt.v[0], v2)
    assert_close(p.value, 1.0 - 0.1 * v1 - 0.1 * v2)


def test_sgd_vanilla_vs_momentum_zero():
    """With momentum=0, SGD should behave exactly like vanilla."""
    p1, p2 = Value(5.0), Value(5.0)
    opt1 = SGD([p1], lr=0.1)
    opt2 = SGD([p2], lr=0.1, momentum=0.0)
    p1.grad = p2.grad = 2.0
    opt1.step()
    opt2.step()
    assert_close(p1.value, p2.value)


# ---------- RMSProp tests ----------


def test_rmsprop_update_and_persistence():
    """Check that RMSProp's squared average and parameter update are correct."""
    p = Value(1.0)
    opt = RMSprop([p], lr=0.1, alpha=0.9, eps=1e-8)

    p.grad = 2.0
    opt.step()
    s1 = 0.1 * 4.0  # (1-0.9)*4 = 0.4
    assert_close(opt.s[0], s1)
    expected = 1.0 - 0.1 * 2.0 / (math.sqrt(s1) + 1e-8)
    assert_close(p.value, expected)

    p.grad = 3.0
    opt.step()
    s2 = 0.9 * s1 + 0.1 * 9.0
    assert_close(opt.s[0], s2)


def test_rmsprop_analytical_two_steps():
    """Full analytical two-step check for RMSProp."""
    p = Value(10.0)
    lr, alpha, eps = 0.1, 0.9, 1e-8
    opt = RMSprop([p], lr=lr, alpha=alpha, eps=eps)

    # Step 1
    p.grad = 2.0
    opt.step()
    s1 = 0.1 * 4.0
    expected1 = 10.0 - lr * 2.0 / (math.sqrt(s1) + eps)
    assert_close(opt.s[0], s1)
    assert_close(p.value, expected1)

    # Step 2
    p.grad = 3.0
    opt.step()
    s2 = alpha * s1 + (1 - alpha) * 9.0
    expected2 = expected1 - lr * 3.0 / (math.sqrt(s2) + eps)
    assert_close(opt.s[0], s2)
    assert_close(p.value, expected2)


# ---------- Adam tests ----------


def test_adam_bias_correction_and_timestep():
    """Check Adam's first step: bias correction makes m_hat=2, v_hat=4, so update = 0.1."""
    p = Value(1.0)
    opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    assert opt.t == 0
    p.grad = 2.0
    opt.step()
    # After first step, parameter should be 0.9 (since update = 0.1)
    assert_close(p.value, 0.9)
    assert opt.t == 1
    assert_close(opt.m[0], 0.2)
    assert_close(opt.v[0], 0.004)


def test_adam_state_persistence():
    """Check that moments update correctly across steps."""
    p = Value(1.0)
    opt = Adam([p], lr=0.01, beta1=0.9, beta2=0.999)
    p.grad = 2.0
    opt.step()
    m1, v1 = opt.m[0], opt.v[0]
    p.grad = 3.0
    opt.step()
    assert_close(opt.m[0], 0.9 * m1 + 0.1 * 3.0)
    assert_close(opt.v[0], 0.999 * v1 + 0.001 * 9.0)


def test_adam_analytical_two_steps():
    """Full analytical two-step check for Adam."""
    p = Value(10.0)
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    opt = Adam([p], lr=lr, beta1=b1, beta2=b2, eps=eps)

    # Step 1
    p.grad = 2.0
    opt.step()
    m1 = 0.2
    v1 = 0.004
    mhat1 = m1 / (1 - b1)
    vhat1 = v1 / (1 - b2)
    expected1 = 10.0 - lr * mhat1 / (math.sqrt(vhat1) + eps)
    assert_close(opt.m[0], m1)
    assert_close(opt.v[0], v1)
    assert_close(p.value, expected1)

    # Step 2
    p.grad = 3.0
    opt.step()
    m2 = b1 * m1 + (1 - b1) * 3.0
    v2 = b2 * v1 + (1 - b2) * 9.0
    mhat2 = m2 / (1 - b1**2)
    vhat2 = v2 / (1 - b2**2)
    expected2 = expected1 - lr * mhat2 / (math.sqrt(vhat2) + eps)
    assert_close(opt.m[0], m2)
    assert_close(opt.v[0], v2)
    assert_close(p.value, expected2)


# ---------- zero_grad ----------


def test_zero_grad():
    """Calling zero_grad() should set the gradient to exactly 0."""
    p = Value(1.0)
    p.grad = 5.0
    SGD([p]).zero_grad()
    assert p.grad == 0.0


# ---------- Zero-gradient state decay ----------


def test_zero_grad_state_decay_sgd():
    """When gradient is zero, momentum velocity should decay by beta each step."""
    p = Value(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    v_before = opt.v[0]
    p.grad = 0.0
    opt.step()
    assert_close(opt.v[0], 0.9 * v_before)


def test_zero_grad_state_decay_rmsprop():
    """When gradient is zero, RMSProp's squared average decays by alpha."""
    p = Value(1.0)
    opt = RMSprop([p], lr=0.1, alpha=0.9)
    p.grad = 2.0
    opt.step()
    s_before = opt.s[0]
    p.grad = 0.0
    opt.step()
    assert_close(opt.s[0], 0.9 * s_before)


def test_zero_grad_state_decay_adam():
    """When gradient is zero, Adam's moments decay by their respective beta."""
    p = Value(1.0)
    opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999)
    p.grad = 2.0
    opt.step()
    m_before, v_before = opt.m[0], opt.v[0]
    p.grad = 0.0
    opt.step()
    assert_close(opt.m[0], 0.9 * m_before)
    assert_close(opt.v[0], 0.999 * v_before)


# ---------- Zero-gradient parameter movement ----------


def test_zero_gradient_parameter_still_moves_momentum():
    """
    Even with zero gradient, momentum carries the parameter forward.
    This is because velocity decays but still exists.
    """
    p = Value(10.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    # After first step: velocity=0.2, parameter=9.98
    assert_close(opt.v[0], 0.2)
    assert_close(p.value, 9.98)

    p.grad = 0.0
    opt.step()
    # velocity decays to 0.9*0.2 = 0.18, then parameter moves by 0.1*0.18 = 0.018
    expected_v = 0.9 * 0.2
    expected_p = 9.98 - 0.1 * expected_v
    assert_close(opt.v[0], expected_v)
    assert_close(p.value, expected_p)


def test_zero_gradient_parameter_still_moves_rmsprop():
    """
    RMSProp with zero gradient: the parameter does NOT move because
    the update term is α * 0 / (√s+ε) = 0.
    The state s still decays though.
    """
    p = Value(10.0)
    opt = RMSprop([p], lr=0.1, alpha=0.9, eps=1e-8)
    p.grad = 2.0
    opt.step()
    s1 = 0.1 * 4.0
    expected1 = 10.0 - 0.1 * 2.0 / (math.sqrt(s1) + 1e-8)
    assert_close(opt.s[0], s1)
    assert_close(p.value, expected1)

    p.grad = 0.0
    opt.step()
    s2 = 0.9 * s1
    # parameter update is zero because grad=0
    expected2 = expected1  # stays same
    assert_close(opt.s[0], s2)
    assert_close(p.value, expected2)


def test_zero_gradient_parameter_still_moves_adam():
    """
    Adam: when gradient is zero, the first moment decays but is still non-zero,
    so the parameter continues to move (but with smaller magnitude).
    """
    p = Value(10.0)
    opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    p.grad = 2.0
    opt.step()
    # first step computed earlier
    mhat1 = 0.2 / (1 - 0.9)
    vhat1 = 0.004 / (1 - 0.999)
    expected1 = 10.0 - 0.1 * mhat1 / (math.sqrt(vhat1) + 1e-8)
    assert_close(p.value, expected1)

    p.grad = 0.0
    opt.step()
    m2 = 0.9 * 0.2
    v2 = 0.999 * 0.004
    mhat2 = m2 / (1 - 0.9**2)
    vhat2 = v2 / (1 - 0.999**2)
    expected2 = expected1 - 0.1 * mhat2 / (math.sqrt(vhat2) + 1e-8)
    assert_close(p.value, expected2)  # parameter still changes


# ---------- State isolation ----------


def test_optimizer_state_is_per_parameter():
    """
    Ensure that each parameter has its own independent state.
    We use Adam with two parameters: only the first gets a gradient initially.
    """
    p1 = Value(1.0)
    p2 = Value(1.0)
    opt = Adam([p1, p2], lr=0.1)

    p1.grad = 2.0
    p2.grad = 0.0
    opt.step()

    # p1's moments should be non-zero; p2's should still be zero
    assert opt.m[0] != 0.0
    assert opt.m[1] == 0.0
    assert opt.v[0] != 0.0
    assert opt.v[1] == 0.0

    # Now switch: only p2 gets gradient
    p1.grad = 0.0
    p2.grad = 3.0
    opt.step()

    # p1's moment decays (since grad=0); p2's moment becomes non-zero
    assert_close(opt.m[0], 0.9 * 0.2)  # from previous step
    assert_close(opt.m[1], (1 - 0.9) * 3.0)  # new moment


# ---------- Sign-changing gradients ----------


def test_momentum_dampens_oscillation():
    """
    When gradient flips sign, momentum reduces the oscillation.
    Starting at 10, with +2 then -2, vanilla SGD would go back to 10,
    but momentum keeps the value closer to 10 (less overshoot).
    """
    p = Value(10.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    p.grad = -2.0
    opt.step()
    # The net change should be small (less than 0.5 from 10)
    assert abs(p.value - 10.0) < 0.5


# ---------- Numerical stability ----------


def test_numerical_stability_tiny_gradients():
    """Very small gradients should not cause NaNs or overflows."""
    p = Value(1.0)
    p.grad = 1e-30
    RMSprop([p]).step()
    assert math.isfinite(p.value)
    p2 = Value(1.0)
    p2.grad = 1e-30
    Adam([p2]).step()
    assert math.isfinite(p2.value)


def test_numerical_stability_large_gradients():
    """Large but finite gradients should not cause infinities for SGD."""
    p = Value(1.0)
    p.grad = 1e20
    SGD([p]).step()
    assert math.isfinite(p.value)


def test_overflow_gradient_raises():
    """
    If a gradient is so large that squaring it overflows, we raise an error
    to prevent silent corruption of RMSProp/Adam state.
    """
    p = Value(1.0)
    p.grad = 1e200  # this will overflow when squared
    with pytest.raises(ValueError, match="overflows"):
        RMSprop([p]).step()
    with pytest.raises(ValueError, match="overflows"):
        Adam([p]).step()
    # SGD doesn't square, so it's fine
    SGD([p]).step()
    assert math.isfinite(p.value)


# ---------- Invalid hyperparameters ----------


def test_invalid_hyperparams_sgd_rmsprop():
    """Check that we reject bad values (negative, NaN, Inf, out-of-range)."""
    p = Value(1.0)
    bad = [
        (SGD, {"lr": -0.1}),
        (SGD, {"lr": float("nan")}),
        (SGD, {"lr": float("inf")}),
        (SGD, {"momentum": 1.0}),  # now invalid (was allowed before)
        (SGD, {"momentum": float("nan")}),
        (RMSprop, {"lr": -0.1}),
        (RMSprop, {"lr": float("nan")}),
        (RMSprop, {"alpha": 1.5}),
        (RMSprop, {"alpha": float("nan")}),
        (RMSprop, {"eps": -1e-8}),
        (RMSprop, {"eps": float("nan")}),
    ]
    for cls, kwargs in bad:
        with pytest.raises(ValueError):
            cls([p], **kwargs)


def test_invalid_hyperparams_adam():
    p = Value(1.0)
    bad = [
        (Adam, {"lr": -0.1}),
        (Adam, {"lr": float("nan")}),
        (Adam, {"lr": float("inf")}),
        (Adam, {"beta1": 1.0}),
        (Adam, {"beta1": float("nan")}),
        (Adam, {"beta2": 1.0}),
        (Adam, {"beta2": float("nan")}),
        (Adam, {"eps": -1e-8}),
        (Adam, {"eps": float("nan")}),
    ]
    for cls, kwargs in bad:
        with pytest.raises(ValueError):
            cls([p], **kwargs)


# ---------- Convergence on f(x)=x² ----------


@pytest.mark.parametrize(
    "cls,kwargs,steps",
    [
        (SGD, {"lr": 0.1}, 200),
        (SGD, {"lr": 0.1, "momentum": 0.9}, 200),
        (RMSprop, {"lr": 0.1}, 200),
        (Adam, {"lr": 0.1}, 200),
    ],
)
def test_convergence_parabola(cls, kwargs, steps):
    """
    All optimizers should minimise f(x)=x² from x=5 down to near zero.
    We use analytical gradient (g=2x) and check final |x| < 1e-2.
    """
    x = Value(5.0)
    opt = cls([x], **kwargs)
    for _ in range(steps):
        # Manual gradient: derivative of x² is 2x
        x.grad = 2 * x.value
        opt.step()
    assert abs(x.value) < 1e-2
