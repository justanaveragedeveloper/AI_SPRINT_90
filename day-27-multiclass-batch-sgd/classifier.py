# -- classifier.py --

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../day-24-autograd"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../day-25-nn-from-scratch")
    )
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../day-26-softmax-cross-entropy")
    )
)

from engine import Value
from loss_and_ops import categorical_cross_entropy, softmax
from nn import MLP


def _get_value(v: Value) -> float:
    """Extract the numeric value from a Value object, supporting both 'data' and 'value'."""
    if hasattr(v, "data"):
        return v.data
    if hasattr(v, "value"):
        return v.value
    raise AttributeError(f"Value object has no 'data' or 'value' attribute: {v}")


def _set_value(v: Value, new_val: float) -> None:
    """Set the numeric value of a Value object, supporting both 'data' and 'value'."""
    if hasattr(v, "data"):
        v.data = new_val
    elif hasattr(v, "value"):
        v.value = new_val
    else:
        raise AttributeError(f"Value object has no 'data' or 'value' attribute: {v}")


def _get_grad(v: Value) -> float:
    """Extract the gradient from a Value object, supporting 'grad', 'gradient', etc."""
    if hasattr(v, "grad"):
        return v.grad
    if hasattr(v, "gradient"):
        return v.gradient
    raise AttributeError(f"Value object has no 'grad' or 'gradient' attribute: {v}")


class MultiClassClassifier:
    """
    Multi‑class classifier combining an MLP backbone with softmax output
    and mini‑batch SGD training.
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], num_classes: int
    ) -> None:
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("input_dim and num_classes must be positive")
        if not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError(
                "hidden_dims must be a non‑empty list of positive integers"
            )
        self.model = MLP(input_dim, hidden_dims + [num_classes])
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, x: list[float]) -> list[Value]:
        """
        Compute class probabilities for a single input sample.
        Returns a list of Value objects summing to 1.
        """
        if len(x) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {len(x)}")
        val_x = [v if isinstance(v, Value) else Value(float(v)) for v in x]
        logits = self.model(val_x)
        if not isinstance(logits, list):
            logits = [logits]
        return softmax(logits)

    def predict(self, x: list[float]) -> int:
        """
        Return the predicted class index (argmax of probabilities).
        No gradient is recorded.
        """
        probs = self.forward(x)
        prob_values = [_get_value(p) for p in probs]
        return prob_values.index(max(prob_values))

    def train_step(
        self,
        batch_x: list[list[float]],
        batch_y: list[int],
        learning_rate: float = 0.01,
    ) -> float:
        """
        Perform one mini‑batch update:
        1. Forward pass and loss accumulation
        2. Mean loss
        3. Backward
        4. SGD parameter update
        Returns the mean loss as a Python float.
        """
        # ---- Input validation ----
        if not batch_x or not batch_y:
            raise ValueError("Batch cannot be empty")
        if len(batch_x) != len(batch_y):
            raise ValueError(
                f"Mismatched batch sizes: x has {len(batch_x)}, y has {len(batch_y)}"
            )
        for x, y in zip(batch_x, batch_y):
            if len(x) != self.input_dim:
                raise ValueError(
                    f"Input dimension mismatch: expected {self.input_dim}, got {len(x)}"
                )
            if not isinstance(y, int) or not (0 <= y < self.num_classes):
                raise ValueError(
                    f"Invalid label {y}; must be int in [0, {self.num_classes-1}]"
                )

        # ---- Forward ----
        self.model.zero_grad()
        total_loss = Value(0.0)

        for x, y in zip(batch_x, batch_y):
            probs = self.forward(x)
            loss = categorical_cross_entropy(probs, y)
            total_loss = total_loss + loss

        # Mean loss
        batch_size = len(batch_x)
        mean_loss = total_loss * (1.0 / batch_size)

        # ---- Backward ----
        mean_loss.backward()

        # ---- Update ----
        for p in self.model.parameters():
            current = _get_value(p)
            grad = _get_grad(p)
            new_val = current - learning_rate * grad
            _set_value(p, new_val)

        return _get_value(mean_loss)
