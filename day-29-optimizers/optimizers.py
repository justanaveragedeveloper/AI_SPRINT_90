"""
Optimizers from scratch – for training neural networks.

This file implements:
- A base Optimizer class (handles parameters and zeroing gradients)
- SGD (with optional momentum)
- RMSProp (scales learning rates per parameter)
- Adam (combines momentum and RMSProp, with bias correction)

All optimizers work with the 'Value' objects from Day 24's autograd engine.
"""

import math
import sys

sys.path.append("../day-24-autograd")  # so we can import the autograd engine


class Optimizer:
    """
    The foundation for all optimizers.

    It holds the parameters and provides a method to reset gradients to zero.
    You must override the `step()` method in subclasses.
    """

    def __init__(self, params):
        # Store parameters as a list so we can index them (important for state)
        self.params = list(params)
        # If there are no parameters, training can't happen – raise an error
        if not self.params:
            raise ValueError("Optimizer received an empty parameter list.")

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.
        Always call this before calling `.backward()` to avoid accumulating old gradients.
        """
        for p in self.params:
            p.grad = 0.0

    def step(self) -> None:
        """Perform one optimization step. Must be overridden by subclasses."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Simple SGD with optional momentum.

    Vanilla SGD:  θ ← θ - α * g
    Momentum:     v_t = β * v_{t-1} + (1-β) * g_t
                  θ_t = θ_{t-1} - α * v_t

    If momentum = 0, this is plain SGD (no velocity).
    Momentum helps smooth updates and escape local minima.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
    ):
        super().__init__(params)

        # Validate learning rate: must be positive and not NaN/infinite
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be finite and > 0, got {lr}")

        # Momentum should be between 0 and 1 (exclusive of 1, because 1 would freeze velocity)
        if not math.isfinite(momentum) or not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be finite and in [0, 1), got {momentum}")

        self.lr = lr
        self.momentum = momentum
        # Velocity buffer (one per parameter) – we keep it as a list of floats
        self.v: list[float] = [0.0] * len(self.params)

    def step(self) -> None:
        """
        Perform one update step.

        Steps:
        1. Check all gradients are finite (not NaN/Inf) to avoid silent failures.
        2. For each parameter, update velocity (if momentum>0) and then the parameter value.
        """
        # First, validate all gradients. This prevents partial updates if one gradient is bad.
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")

        # Now apply the updates
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)

            if self.momentum > 0.0:
                # Update velocity: accumulate momentum and current gradient
                self.v[i] = self.momentum * self.v[i] + (1.0 - self.momentum) * grad
                # Apply velocity to parameter
                p.value -= self.lr * self.v[i]
            else:
                # Plain SGD: just subtract learning rate * gradient
                p.value -= self.lr * grad


class RMSprop(Optimizer):
    """
    RMSProp – adapts learning rate per parameter using a moving average of squared gradients.

    Formula:
        s_t = β * s_{t-1} + (1-β) * g_t²
        θ_t = θ_{t-1} - α * g_t / (√s_t + ε)

    This helps when gradients vary a lot across parameters (ill-conditioned problems).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__(params)

        # Validate all hyperparameters
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"Learning rate must be finite and > 0, got {lr}")
        if not math.isfinite(alpha) or not (0.0 <= alpha < 1.0):
            raise ValueError(f"Alpha must be finite and in [0, 1), got {alpha}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"Epsilon must be finite and > 0, got {eps}")

        self.lr = lr
        self.alpha = alpha   # decay rate for squared gradient average
        self.eps = eps       # small constant to avoid division by zero
        # Squared gradient accumulator (one per parameter)
        self.s: list[float] = [0.0] * len(self.params)

    def step(self) -> None:
        """Perform one RMSProp step."""
        # Validate all gradients and also check if gradient² overflows
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")
            # Prevent extreme gradients that would cause overflow in grad²
            if not math.isfinite(grad * grad):
                raise ValueError(f"Gradient squared overflows for parameter {i}: {grad}")

        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            # Update moving average of squared gradient
            self.s[i] = self.alpha * self.s[i] + (1.0 - self.alpha) * (grad * grad)
            # Scale the learning rate inversely to the sqrt of the average
            p.value -= self.lr * grad / (math.sqrt(self.s[i]) + self.eps)


class Adam(Optimizer):
    """
    Adam – combines momentum and RMSProp with bias correction.

    Equations:
        m_t = β1 * m_{t-1} + (1-β1) * g_t          (first moment, like momentum)
        v_t = β2 * v_{t-1} + (1-β2) * g_t²         (second moment, like RMSProp)
        m_hat = m_t / (1 - β1^t)                   (bias-corrected)
        v_hat = v_t / (1 - β2^t)
        θ_t = θ_{t-1} - α * m_hat / (√v_hat + ε)

    Adam is often the default choice because it works well on many problems.
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

        # Validate everything
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

        # First and second moment buffers (one per parameter)
        self.m: list[float] = [0.0] * len(self.params)
        self.v: list[float] = [0.0] * len(self.params)
        # Timestep starts at 0 and is incremented each step() call
        self.t: int = 0

    def step(self) -> None:
        """Perform one Adam step with bias correction."""
        # Validate all gradients and overflow checks
        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)
            if not math.isfinite(grad):
                raise ValueError(f"Non-finite gradient detected for parameter {i}: {grad}")
            if not math.isfinite(grad * grad):
                raise ValueError(f"Gradient squared overflows for parameter {i}: {grad}")

        # Increment timestep
        self.t += 1
        # Precompute powers for bias correction denominators
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for i, p in enumerate(self.params):
            grad = getattr(p, "grad", 0.0)

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)

            # Bias-corrected estimates
            m_hat = self.m[i] / (1.0 - beta1_t)
            v_hat = self.v[i] / (1.0 - beta2_t)

            # Parameter update
            p.value -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)