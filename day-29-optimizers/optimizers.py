"""
First‑principles optimizers for the autograd engine.

Implements:
- Base Optimizer with zero_grad()
- SGD (vanilla and with momentum)
- RMSProp (with √s + ε)
- Adam (with bias correction and timestep)

All optimizers operate on Value parameters from the Day‑24 autograd engine.
"""

import math
import sys

sys.path.append("../day-24-autograd")


class Optimizer:
    """Base class for all optimizers. Manages parameter collection and zero_grad."""

    def __init__(self, params):
        self.params = list(params)
        if not self.params:
            raise ValueError("Optimizer received an empty parameter list.")

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.params:
            p.grad = 0.0

    def step(self) -> None:
        """Perform a single optimization step. Must be overridden."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Vanilla:  θ ← θ - αg
    Momentum: v_t = β v_{t-1} + (1-β) g_t
              θ_t = θ_{t-1} - α v_t

    If momentum == 0, behaves as vanilla SGD.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
    ):
        super().__init__(params)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be finite and > 0, got {lr}")
        if not math.isfinite(momentum) or not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be finite and in [0, 1), got {momentum}")

        self.lr = lr
        self.momentum = momentum
        self.v: list[float] = [0.0] * len(self.params)

    def step(self) -> None:
        """Update parameters using SGD (with optional momentum)."""
        # Pre-validate all gradients to avoid partial updates
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")

        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if self.momentum > 0.0:
                self.v[i] = self.momentum * self.v[i] + (1.0 - self.momentum) * grad
                p.value -= self.lr * self.v[i]
            else:
                p.value -= self.lr * grad


class RMSprop(Optimizer):
    """
    RMSProp optimizer.

    s_t = β s_{t-1} + (1-β) g_t²
    θ_t = θ_{t-1} - α * g_t / (√s_t + ε)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__(params)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be finite and > 0, got {lr}")
        if not math.isfinite(alpha) or not (0.0 <= alpha < 1.0):
            raise ValueError(f"Alpha must be finite and in [0, 1), got {alpha}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"Epsilon must be finite and > 0, got {eps}")

        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.s: list[float] = [0.0] * len(self.params)

    def step(self) -> None:
        """Update parameters using RMSProp."""
        # Pre-validate all gradients and ensure grad**2 is finite
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")
            if not math.isfinite(grad * grad):
                raise ValueError(f"Gradient squared overflows for parameter {i}: {grad}")

        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            self.s[i] = self.alpha * self.s[i] + (1.0 - self.alpha) * (grad * grad)
            p.value -= self.lr * grad / (math.sqrt(self.s[i]) + self.eps)


class Adam(Optimizer):
    """
    Adam optimizer.

    m_t = β1 m_{t-1} + (1-β1) g_t
    v_t = β2 v_{t-1} + (1-β2) g_t²
    m_hat = m_t / (1 - β1^t)
    v_hat = v_t / (1 - β2^t)
    θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)
    """

    def __init__(
        self,
        params,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params)
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be finite and > 0, got {lr}")
        if not math.isfinite(beta1) or not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be finite and in [0, 1), got {beta1}")
        if not math.isfinite(beta2) or not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be finite and in [0, 1), got {beta2}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"Epsilon must be finite and > 0, got {eps}")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m: list[float] = [0.0] * len(self.params)
        self.v: list[float] = [0.0] * len(self.params)
        self.t: int = 0

    def step(self) -> None:
        """Update parameters using Adam with bias correction."""
        # Pre-validate all gradients and check for overflow in grad**2
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")
            if not math.isfinite(grad * grad):
                raise ValueError(f"Gradient squared overflows for parameter {i}: {grad}")

        self.t += 1
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[i] / (1.0 - beta1_t)
            v_hat = self.v[i] / (1.0 - beta2_t)

            p.value -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)