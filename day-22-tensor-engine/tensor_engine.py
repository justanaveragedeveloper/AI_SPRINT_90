from __future__ import annotations

from typing import Callable, Iterable

import numpy as np


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Reduce a broadcasted gradient back to the original tensor shape.

    During the forward pass, NumPy may automatically broadcast tensors of
    different shapes. During backpropagation, the gradient must be reduced
    to match the original tensor's shape.

    Example
    -------
    Tensor A : (2, 3)
    Tensor B : (1, 3)

    A + B -> (2, 3)

    The gradient flowing back to B must be summed along axis 0 to recover
    shape (1, 3).

    Parameters
    ----------
    grad : np.ndarray
        Incoming gradient from the output tensor.

    shape : tuple[int, ...]
        Original shape of the tensor before broadcasting.

    Returns
    -------
    np.ndarray
        Gradient reshaped to match the original tensor.
    """

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, (grad_dim, target_dim) in enumerate(zip(grad.shape, shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


class Tensor:
    """
    Lightweight tensor supporting reverse-mode automatic differentiation.

    Each Tensor represents one node inside a dynamic computational graph.

    Attributes
    ----------
    data : np.ndarray
        Numerical values stored in this tensor.

    grad : np.ndarray
        Gradient of the final loss with respect to this tensor.

    _prev : set[Tensor]
        Parent tensors that produced this tensor.

    _backward : Callable[[], None]
        Function responsible for propagating gradients to parent tensors.
    """

    def __init__(
        self,
        data: float | int | list | np.ndarray,
        _children: Iterable["Tensor"] = (),
    ) -> None:
        """
        Create a Tensor.

        Parameters
        ----------
        data
            Scalar, list, or NumPy array.

        _children
            Parent tensors used to construct this tensor.
        """

        try:
            self.data = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Tensor data must be a numeric scalar, list, or NumPy array."
            ) from exc

        self.grad: np.ndarray = np.zeros_like(self.data)

        self._prev: set[Tensor] = set(_children)

        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """
        Return a readable representation of the tensor.
        """
        return (
            f"Tensor(shape={self.data.shape}, "
            f"data={self.data}, "
            f"grad={self.grad})"
        )

    def zero_grad(self) -> None:
        """
        Reset gradients to zero.

        This should typically be called before running another
        backward pass during training.
        """
        self.grad.fill(0.0)

    # ---------------------------------------------------
    # Addition
    # ---------------------------------------------------

    def __add__(self, other: float | int | "Tensor") -> "Tensor":
        """
        Element-wise tensor addition.

        Supports NumPy broadcasting.

        Parameters
        ----------
        other
            Tensor or numeric value.

        Returns
        -------
        Tensor
            Result of the addition.
        """

        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data + other.data, (self, other))

        def _backward() -> None:
            self.grad += _sum_to_shape(out.grad, self.data.shape)
            other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: float | int | "Tensor") -> "Tensor":
        """
        Support scalar + Tensor.
        """
        return self + other

    # ---------------------------------------------------
    # Matrix Multiplication
    # ---------------------------------------------------

    def __matmul__(self, other: float | int | "Tensor") -> "Tensor":
        """
        Matrix multiplication.

        Parameters
        ----------
        other
            Tensor or numeric value.

        Returns
        -------
        Tensor
            Matrix multiplication result.
        """

        if not isinstance(other, Tensor):
            other = Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                "Matrix multiplication requires both tensors to be 2-dimensional."
            )

        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                f"Incompatible matrix shapes: "
                f"{self.data.shape} and {other.data.shape}"
            )

        out = Tensor(self.data @ other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ---------------------------------------------------
    # ReLU
    # ---------------------------------------------------

    def relu(self) -> "Tensor":
        """
        Apply the ReLU activation function.

        Returns
        -------
        Tensor
            Activated tensor.
        """

        out = Tensor(np.maximum(0.0, self.data), (self,))

        def _backward() -> None:
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    # ---------------------------------------------------
    # Backward Pass
    # ---------------------------------------------------

    def backward(self) -> None:
        """
        Execute reverse-mode automatic differentiation.

        The computational graph is first traversed using a depth-first search
        to build a topological ordering. Gradients are then propagated in
        reverse topological order.

        Notes
        -----
        This educational engine only supports implicit backward on scalar
        outputs, similar to PyTorch.
        """

        if self.data.size != 1:
            raise RuntimeError(
                "backward() can only be called implicitly on scalar outputs. "
                "Reduce the tensor to a scalar before calling backward()."
            )

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(node: "Tensor") -> None:
            if node in visited:
                return


            for parent in node._prev:
                build(parent)

            topo.append(node)

        build(self)

        self.grad.fill(1.0)

        for node in reversed(topo):
            node._backward()
