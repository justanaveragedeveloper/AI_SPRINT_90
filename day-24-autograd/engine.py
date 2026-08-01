"""engine.py – Automatic Differentiation Engine

Implements a scalar `Value` node that builds a dynamic computational graph
(DAG) with reverse‑mode autograd. Supports addition, multiplication,
power, tanh, and ReLU operations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Value:
    """
    A node in a computational graph that holds a scalar value and its gradient.

    Attributes:
        value (float): The forward‑pass scalar value.
        gradient (float): The accumulated gradient (∂L/∂self).
        _children (set[Value]): Child nodes that produced this node.
        _operation (str): Operator name used to create this node (for debugging).
        _backward_fn (Callable[[], None]): Local gradient closure.
        label (str): Optional label for debugging.
    """

    def __init__(
        self,
        value: float,
        children: tuple[Value, ...] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        self.value = float(value)
        self.gradient = 0.0
        self._children: set[Value] = set(children)
        self._operation = operation
        self._backward_fn: Callable[[], None] = lambda: None
        self.label = label
        logger.debug(f"Created Value(value={self.value:.4f}, op='{operation}')")

    def __repr__(self) -> str:
        return (
            f"Value(value={self.value:.4f}, gradient={self.gradient:.4f}"
            f"{', op=' + self._operation if self._operation else ''})"
        )

    def __add__(self, other: Value | float) -> Value:
        """Addition operator overload."""
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.value + other.value, (self, other), "+")
        result._backward_fn = self._make_add_backward_fn(other, result)
        logger.debug(f"{self} + {other} -> {result}")
        return result

    def __radd__(self, other: float) -> Value:
        """Reverse addition (e.g., 2 + Value)."""
        return self.__add__(other)

    def __mul__(self, other: Value | float) -> Value:
        """Multiplication operator overload."""
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.value * other.value, (self, other), "*")
        result._backward_fn = self._make_mul_backward_fn(other, result)
        logger.debug(f"{self} * {other} -> {result}")
        return result

    def __rmul__(self, other: float) -> Value:
        """Reverse multiplication (e.g., 2 * Value)."""
        return self.__mul__(other)

    def __pow__(self, exponent: float) -> Value:
        """
        Power operator overload (only for numeric exponent).

        Raises:
            AssertionError: if exponent is not int/float.
        """
        assert isinstance(
            exponent, (int, float)
        ), "Power only supports numeric exponents"
        result = Value(self.value**exponent, (self,), f"**{exponent}")
        result._backward_fn = self._make_pow_backward_fn(exponent, result)
        logger.debug(f"{self} ** {exponent} -> {result}")
        return result

    def tanh(self) -> Value:
        """Hyperbolic tangent activation."""
        tanh_value = math.tanh(self.value)
        result = Value(tanh_value, (self,), "tanh")
        result._backward_fn = self._make_tanh_backward_fn(tanh_value, result)
        logger.debug(f"tanh({self}) -> {result}")
        return result

    def relu(self) -> Value:
        """Rectified Linear Unit activation."""
        result = Value(max(0.0, self.value), (self,), "relu")
        result._backward_fn = self._make_relu_backward_fn(result)
        logger.debug(f"relu({self}) -> {result}")
        return result

    def backward(self) -> None:
        """
        Execute reverse‑mode automatic differentiation.

        Builds a topological order of the entire graph via DFS and propagates
        gradients from the root (self) back to all leaves.
        """
        # 1. Topological sort (children before parents)
        topo_order: list[Value] = []
        visited: set[Value] = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo_order.append(node)

        build_topo(self)
        logger.debug(f"Topological order: {[v.label or str(v) for v in topo_order]}")

        # 2. Seed gradient at root
        self.gradient = 1.0

        # 3. Traverse in reverse order and call each node's backward closure
        for node in reversed(topo_order):
            logger.debug(f"Applying backward on {node}")
            node._backward_fn()

    # ---------- Backward closure factories ----------

    def _make_add_backward_fn(self, other: Value, result: Value) -> Callable[[], None]:
        def _backward() -> None:
            self.gradient += result.gradient
            other.gradient += result.gradient

        return _backward

    def _make_mul_backward_fn(self, other: Value, result: Value) -> Callable[[], None]:
        def _backward() -> None:
            self.gradient += other.value * result.gradient
            other.gradient += self.value * result.gradient

        return _backward

    def _make_pow_backward_fn(
        self, exponent: float, result: Value
    ) -> Callable[[], None]:
        def _backward() -> None:
            # Handle edge cases to avoid NaN/Inf for x = 0 and non‑positive exponent
            if self.value == 0.0:
                if exponent == 0:
                    # derivative of constant 1 is 0
                    self.gradient += 0.0
                elif exponent < 0:
                    raise ValueError(
                        f"Derivative undefined for 0^{exponent} (exponent < 0)"
                    )
                else:
                    # exponent > 0: derivative = exponent * 0^(exponent-1)
                    # For exponent == 1, derivative = 1 (since x' = 1)
                    if exponent == 1:
                        self.gradient += 1.0 * result.gradient
                    else:
                        self.gradient += 0.0  # for exponent > 1, derivative is 0 at x=0
            else:
                self.gradient += (
                    exponent * (self.value ** (exponent - 1)) * result.gradient
                )

        return _backward

    def _make_tanh_backward_fn(
        self, tanh_value: float, result: Value
    ) -> Callable[[], None]:
        def _backward() -> None:
            # derivative: 1 - tanh^2
            self.gradient += (1.0 - tanh_value * tanh_value) * result.gradient

        return _backward

    def _make_relu_backward_fn(self, result: Value) -> Callable[[], None]:
        def _backward() -> None:
            self.gradient += (result.value > 0.0) * result.gradient

        return _backward
