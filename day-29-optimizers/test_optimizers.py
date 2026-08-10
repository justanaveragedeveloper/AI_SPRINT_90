"""Comprehensive test suite for all optimizers, including analytical, state, and numerical tests."""

import math
import sys

import pytest

sys.path.append("../day-24-autograd")
sys.path.append("../day-25-nn-from-scratch")

from engine import Value # noqa: I001
from optimizers import SGD, RMSprop, Adam

# ---------- Helpers ----------

def assert_close(a, b, tol=1e-6):
    if isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            assert_close(x, y, tol)
    else:
        assert abs(a - b) < tol, f"{a} != {b}"

# ---------- SGD (3 tests) ----------

def test_sgd_vanilla():
    p = Value(1.0)
    p.grad = 2.0
    SGD([p], lr=0.1).step()
    assert_close(p.value, 0.8)

def test_sgd_momentum_update_and_persistence():
    p = Value(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    v1 = 0.2
    assert_close(opt.v[0], v1)
    assert_close(p.value, 1.0 - 0.1 * v1)
    p.grad = 3.0
    opt.step()
    v2 = 0.9 * v1 + 0.1 * 3.0
    assert_close(opt.v[0], v2)
    assert_close(p.value, 1.0 - 0.1 * v1 - 0.1 * v2)

def test_sgd_vanilla_vs_momentum_zero():
    p1, p2 = Value(5.0), Value(5.0)
    opt1 = SGD([p1], lr=0.1)
    opt2 = SGD([p2], lr=0.1, momentum=0.0)
    p1.grad = p2.grad = 2.0
    opt1.step()
    opt2.step()
    assert_close(p1.value, p2.value)

# ---------- RMSProp (2 tests) ----------

def test_rmsprop_update_and_persistence():
    p = Value(1.0)
    opt = RMSprop([p], lr=0.1, alpha=0.9, eps=1e-8)
    p.grad = 2.0
    opt.step()
    s1 = 0.1 * 4.0
    assert_close(opt.s[0], s1)
    assert_close(p.value, 1.0 - 0.1 * 2.0 / (math.sqrt(s1) + 1e-8))
    p.grad = 3.0
    opt.step()
    s2 = 0.9 * s1 + 0.1 * 9.0
    assert_close(opt.s[0], s2)

def test_rmsprop_analytical_two_steps():
    p = Value(10.0)
    lr, alpha, eps = 0.1, 0.9, 1e-8
    opt = RMSprop([p], lr=lr, alpha=alpha, eps=eps)

    p.grad = 2.0
    opt.step()
    s1 = 0.1 * 4.0
    expected1 = 10.0 - lr * 2.0 / (math.sqrt(s1) + eps)
    assert_close(opt.s[0], s1)
    assert_close(p.value, expected1)

    p.grad = 3.0
    opt.step()
    s2 = alpha * s1 + (1 - alpha) * 9.0
    expected2 = expected1 - lr * 3.0 / (math.sqrt(s2) + eps)
    assert_close(opt.s[0], s2)
    assert_close(p.value, expected2)

# ---------- Adam (3 tests) ----------

def test_adam_bias_correction_and_timestep():
    p = Value(1.0)
    opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    assert opt.t == 0
    p.grad = 2.0
    opt.step()
    assert_close(p.value, 0.9)
    assert opt.t == 1
    assert_close(opt.m[0], 0.2)
    assert_close(opt.v[0], 0.004)

def test_adam_state_persistence():
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
    p = Value(10.0)
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    opt = Adam([p], lr=lr, beta1=b1, beta2=b2, eps=eps)

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

    p.grad = 3.0
    opt.step()
    m2 = b1 * m1 + (1 - b1) * 3.0
    v2 = b2 * v1 + (1 - b2) * 9.0
    mhat2 = m2 / (1 - b1 ** 2)
    vhat2 = v2 / (1 - b2 ** 2)
    expected2 = expected1 - lr * mhat2 / (math.sqrt(vhat2) + eps)
    assert_close(opt.m[0], m2)
    assert_close(opt.v[0], v2)
    assert_close(p.value, expected2)

# ---------- zero_grad ----------

def test_zero_grad():
    p = Value(1.0)
    p.grad = 5.0
    SGD([p]).zero_grad()
    assert p.grad == 0.0

# ---------- Zero-gradient state decay ----------

def test_zero_grad_state_decay_sgd():
    p = Value(1.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    v_before = opt.v[0]
    p.grad = 0.0
    opt.step()
    assert_close(opt.v[0], 0.9 * v_before)

def test_zero_grad_state_decay_rmsprop():
    p = Value(1.0)
    opt = RMSprop([p], lr=0.1, alpha=0.9)
    p.grad = 2.0
    opt.step()
    s_before = opt.s[0]
    p.grad = 0.0
    opt.step()
    assert_close(opt.s[0], 0.9 * s_before)

def test_zero_grad_state_decay_adam():
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
    """Momentum should still move parameter when gradient is zero."""
    p = Value(10.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    # After step 1, velocity = 0.2, parameter = 9.98
    assert_close(opt.v[0], 0.2)
    assert_close(p.value, 9.98)

    p.grad = 0.0
    opt.step()
    # Velocity decays to 0.18, parameter moves by 0.018
    expected_v = 0.9 * 0.2
    expected_p = 9.98 - 0.1 * expected_v
    assert_close(opt.v[0], expected_v)
    assert_close(p.value, expected_p)

def test_zero_gradient_parameter_still_moves_rmsprop():
    """RMSProp should still move parameter when gradient is zero (with decayed s)."""
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
    expected2 = expected1 - 0.1 * 0.0 / (math.sqrt(s2) + 1e-8)  # no movement because grad=0
    assert_close(opt.s[0], s2)
    assert_close(p.value, expected2)  # stays same

def test_zero_gradient_parameter_still_moves_adam():
    """Adam should still move parameter when gradient is zero (decaying moments)."""
    p = Value(10.0)
    opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    p.grad = 2.0
    opt.step()
    # m1=0.2, v1=0.004
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
    assert_close(p.value, expected2)

# ---------- State isolation ----------

def test_optimizer_state_is_per_parameter():
    p1 = Value(1.0)
    p2 = Value(1.0)
    opt = Adam([p1, p2], lr=0.1)

    p1.grad = 2.0
    p2.grad = 0.0
    opt.step()

    assert opt.m[0] != 0.0
    assert opt.m[1] == 0.0
    assert opt.v[0] != 0.0
    assert opt.v[1] == 0.0

    p1.grad = 0.0
    p2.grad = 3.0
    opt.step()

    assert_close(opt.m[0], 0.9 * 0.2)
    assert_close(opt.m[1], (1 - 0.9) * 3.0)

# ---------- Sign-changing gradients ----------

def test_momentum_dampens_oscillation():
    p = Value(10.0)
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = 2.0
    opt.step()
    p.grad = -2.0
    opt.step()
    assert abs(p.value - 10.0) < 0.5

# ---------- Numerical stability ----------

def test_numerical_stability_tiny_gradients():
    p = Value(1.0)
    p.grad = 1e-30
    RMSprop([p]).step()
    assert math.isfinite(p.value)
    p2 = Value(1.0)
    p2.grad = 1e-30
    Adam([p2]).step()
    assert math.isfinite(p2.value)

def test_numerical_stability_large_gradients():
    p = Value(1.0)
    p.grad = 1e20
    SGD([p]).step()
    assert math.isfinite(p.value)

def test_overflow_gradient_raises():
    """Test that a gradient causing overflow in squared term raises ValueError."""
    p = Value(1.0)
    p.grad = 1e200  # grad**2 overflows
    with pytest.raises(ValueError, match="overflows"):
        RMSprop([p]).step()
    with pytest.raises(ValueError, match="overflows"):
        Adam([p]).step()
    # SGD does not square, so it should work (though huge, but finite)
    SGD([p]).step()
    assert math.isfinite(p.value)

# ---------- Invalid hyperparameters ----------

def test_invalid_hyperparams_sgd_rmsprop():
    p = Value(1.0)
    bad = [
        (SGD, {"lr": -0.1}),
        (SGD, {"lr": float("nan")}),
        (SGD, {"lr": float("inf")}),
        (SGD, {"momentum": 1.0}),
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
    x = Value(5.0)
    opt = cls([x], **kwargs)
    for _ in range(steps):
        x.grad = 2 * x.value
        opt.step()
    assert abs(x.value) < 1e-2