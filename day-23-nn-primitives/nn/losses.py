import numpy as np
import warnings

from .module import Module
from tensor_engine import Tensor

# Epsilon for numerical stability in BCE loss
EPS = 1e-15


class MSELoss(Module):
    """
    Mean Squared Error loss.

    ℒ = (1 / N) · Σ (y_pred - y_true)²

    where N is the total number of elements (batch size × output dimension).

    This implementation manually defines the forward and backward passes,
    so it does not rely on Tensor having __mul__, __pow__, or sum().
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the MSE loss.

        Parameters
        ----------
        pred : Tensor
            Predicted values (any shape).
        target : Tensor
            Ground‑truth values (same shape as pred).

        Returns
        -------
        Tensor
            Scalar loss value (as a Tensor) with a custom backward hook.

        Raises
        ------
        TypeError
            If pred or target are not Tensors.
        ValueError
            If pred and target do not have the same shape.
        """
        # Validate input types
        if not isinstance(pred, Tensor):
            raise TypeError(f"Expected Tensor for pred, got {type(pred).__name__}")
        if not isinstance(target, Tensor):
            raise TypeError(f"Expected Tensor for target, got {type(target).__name__}")

        # Validate shapes
        if pred.data.shape != target.data.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.data.shape} vs "
                f"target shape {target.data.shape}"
            )

        # ---- Forward pass using NumPy ----
        diff = pred.data - target.data
        num_elements = diff.size
        loss_data = np.mean(diff**2)  # scalar

        # ---- Build output Tensor ----
        out = Tensor(loss_data, _children=(pred, target))

        # ---- Define custom backward ----
        def _backward() -> None:
            # Gradient of MSE w.r.t pred:
            #   dL/dpred = (2 / N) * (pred - target)
            if pred.requires_grad:
                pred.grad += (2.0 / num_elements) * (pred.data - target.data) * out.grad
            # target does not require gradients (typically), but we could add it
            # if target.requires_grad:
            #     target.grad += (2.0 / num_elements) * (target.data - pred.data) * out.grad

        out._backward = _backward
        return out


class BCELoss(Module):
    """
    Binary Cross‑Entropy loss (for binary classification).

    ℒ = -(1 / N) · Σ [ y·log(ŷ + ε) + (1‑y)·log(1‑ŷ + ε) ]

    where ε = 1e‑15 is used for numerical safety (clamping the predictions).

    This implementation manually defines the forward and backward passes,
    so it does not rely on Tensor having __mul__, log, or sum().

    Note: Predictions should be probabilities in [0, 1]. Values outside this range
    will raise a ValueError to avoid silent clipping.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the BCE loss.

        Parameters
        ----------
        pred : Tensor
            Predicted probabilities (should be in [0, 1]).
        target : Tensor
            Binary ground‑truth labels (0 or 1).

        Returns
        -------
        Tensor
            Scalar loss value (as a Tensor) with a custom backward hook.

        Raises
        ------
        TypeError
            If pred or target are not Tensors.
        ValueError
            If pred and target do not have the same shape, or if pred contains
            values outside the [0, 1] range.
        """
        # Validate input types
        if not isinstance(pred, Tensor):
            raise TypeError(f"Expected Tensor for pred, got {type(pred).__name__}")
        if not isinstance(target, Tensor):
            raise TypeError(f"Expected Tensor for target, got {type(target).__name__}")

        # Validate shapes
        if pred.data.shape != target.data.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.data.shape} vs "
                f"target shape {target.data.shape}"
            )

        # Validate that predictions are probabilities in [0, 1]
        if np.any(pred.data < 0) or np.any(pred.data > 1):
            raise ValueError(
                "BCELoss expects predictions in [0, 1]. "
                "Consider using a sigmoid activation before the loss."
            )

        # ---- Clamp predictions to avoid log(0) ----
        pred_clipped = np.clip(pred.data, EPS, 1.0 - EPS)

        # ---- Forward pass using NumPy ----
        num_elements = pred_clipped.size
        loss_data = -np.mean(
            target.data * np.log(pred_clipped)
            + (1.0 - target.data) * np.log(1.0 - pred_clipped)
        )

        # ---- Build output Tensor ----
        out = Tensor(loss_data, _children=(pred, target))

        # ---- Define custom backward ----
        def _backward() -> None:
            # Derivative of BCE w.r.t pred (after clipping):
            #   dL/dpred = (pred - y) / (pred * (1 - pred))   (averaged over N)
            # To avoid division by zero, add a tiny epsilon to the denominator.
            denom = pred_clipped * (1.0 - pred_clipped) + EPS
            grad_pred = (pred_clipped - target.data) / denom
            grad_pred = grad_pred / num_elements

            if pred.requires_grad:
                pred.grad += grad_pred * out.grad

        out._backward = _backward
        return out
