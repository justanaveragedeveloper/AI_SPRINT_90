# -- classifier.py --
# Multi‑Class Classifier with Batch SGD
# Integrates custom autograd (Value), MLP, softmax, and cross‑entropy.

import os
import sys

# Add sibling directories to Python path so we can import previous modules.
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

# -------------------------------------------------------------------
# Helper functions to work with Value objects agnostically
# (so the code works whether the attribute is called 'data' or 'value')
# -------------------------------------------------------------------


def _get_value(v: Value) -> float:
    """
    Retrieve the numeric value from a Value object.
    Supports both 'data' and 'value' attribute names.
    """
    if hasattr(v, "data"):
        return v.data
    if hasattr(v, "value"):
        return v.value
    raise AttributeError(f"Value object has no 'data' or 'value' attribute: {v}")


def _set_value(v: Value, new_val: float) -> None:
    """
    Set the numeric value of a Value object.
    Works with 'data' or 'value' attribute names.
    """
    if hasattr(v, "data"):
        v.data = new_val
    elif hasattr(v, "value"):
        v.value = new_val
    else:
        raise AttributeError(f"Value object has no 'data' or 'value' attribute: {v}")


def _get_grad(v: Value) -> float:
    """
    Retrieve the gradient from a Value object.
    Supports 'grad' or 'gradient' attribute names.
    """
    if hasattr(v, "grad"):
        return v.grad
    if hasattr(v, "gradient"):
        return v.gradient
    raise AttributeError(f"Value object has no 'grad' or 'gradient' attribute: {v}")


# -------------------------------------------------------------------
# Main Classifier
# -------------------------------------------------------------------


class MultiClassClassifier:
    """
    A multi‑class classifier that wraps an MLP, applies softmax,
    and provides mini‑batch training with SGD.

    Responsibilities:
        - Hold the MLP model.
        - Compute forward pass (logits → softmax probabilities).
        - Predict the class with highest probability.
        - Perform one training step on a mini‑batch.
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], num_classes: int
    ) -> None:
        """
        Build the classifier.

        Args:
            input_dim:  Number of features in each input sample.
            hidden_dims: List of neuron counts for each hidden layer.
            num_classes: Number of output classes (must be >= 2).
        """
        # Validate dimensions
        if input_dim <= 0 or num_classes <= 0:
            raise ValueError("input_dim and num_classes must be positive")
        if not hidden_dims or any(d <= 0 for d in hidden_dims):
            raise ValueError(
                "hidden_dims must be a non‑empty list of positive integers"
            )

        # Create the MLP: input → hidden layers → output layer (num_classes)
        self.model = MLP(input_dim, hidden_dims + [num_classes])
        self.num_classes = num_classes
        self.input_dim = input_dim

    def forward(self, x: list[float]) -> list[Value]:
        """
        Run a single input through the network and return softmax probabilities.

        Steps:
            1. Convert raw numbers to Value objects (if not already).
            2. Pass through the MLP to get logits (unnormalised scores).
            3. Apply softmax to get a probability distribution.

        Returns:
            A list of Value objects, each representing the probability of a class.
            The probabilities sum to ~1.0.
        """
        # Validate input length
        if len(x) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {len(x)}")

        # Ensure every element is a Value (so the autograd graph is built)
        val_x = [v if isinstance(v, Value) else Value(float(v)) for v in x]

        # Forward pass through MLP → logits (may be a single Value or a list)
        logits = self.model(val_x)
        if not isinstance(logits, list):
            logits = [logits]

        # Softmax normalisation → probabilities
        return softmax(logits)

    def predict(self, x: list[float]) -> int:
        """
        Predict the class label for a single input sample.

        Uses the forward pass to get probabilities, then returns the index
        of the highest probability. This does not modify the computational graph.
        """
        probs = self.forward(x)
        # Extract numeric values from Value objects
        prob_values = [_get_value(p) for p in probs]
        # Argmax: index of the largest probability
        return prob_values.index(max(prob_values))

    def train_step(
        self,
        batch_x: list[list[float]],
        batch_y: list[int],
        learning_rate: float = 0.01,
    ) -> float:
        """
        Perform one mini‑batch SGD training step.

        Steps:
            1. Validate batch inputs.
            2. Zero out previous gradients.
            3. Forward pass for each sample → accumulate total loss.
            4. Compute mean loss over the batch.
            5. Backward pass to compute gradients.
            6. Update all parameters using SGD: p = p - lr * grad(p).

        Returns:
            The mean loss (as a Python float) for this batch.
        """
        # ---- Input Validation ----
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

        # ---- Forward Pass ----
        # Clear old gradients so they don't accumulate from previous steps
        self.model.zero_grad()

        # Accumulate total loss over the batch
        total_loss = Value(0.0)  # start with zero as a Value
        for x, y in zip(batch_x, batch_y):
            probs = self.forward(x)  # get probabilities
            loss = categorical_cross_entropy(probs, y)  # per‑sample loss
            total_loss = total_loss + loss  # sum losses

        # Average loss over the batch
        batch_size = len(batch_x)
        mean_loss = total_loss * (1.0 / batch_size)

        # ---- Backward Pass ----
        # Compute gradients for all parameters in the computational graph
        mean_loss.backward()

        # ---- Parameter Update (SGD) ----
        # For each parameter, apply: new_value = old_value - learning_rate * gradient
        for param in self.model.parameters():
            current_val = _get_value(param)
            gradient = _get_grad(param)
            updated_val = current_val - learning_rate * gradient
            _set_value(param, updated_val)

        # Return the loss value as a Python float
        return _get_value(mean_loss)
