"""
Day 39: End-to-End CNN Trainer & Optimizer from First Principles.

This module brings together the CNN, DataLoader, and Loss functions to build
a complete training loop. It implements SGD with Momentum, gradient accumulation,
batch averaging, and defensive numerical checks – all using explicit Python lists
to expose the underlying mathematics.
"""

import math
from typing import Any


class SGDMomentum:
    """
    Stochastic Gradient Descent optimizer with momentum.

    For each trainable parameter (weight or bias), we keep a velocity tensor
    of the same shape. The update rule is:

        v_new = momentum * v_old + learning_rate * gradient
        param_new = param_old - v_new

    This is the exact convention used in the project specification.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        """
        Create an optimizer instance.

        Args:
            lr: learning rate (must be > 0)
            momentum: momentum coefficient (0.0 <= momentum < 1.0)
        """
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("Momentum must be in range [0.0, 1.0).")

        self.lr = lr
        self.momentum = momentum

        # Dictionary: parameter identifier (e.g., "fc1.weight") -> velocity tensor
        self.velocities: dict[str, Any] = {}

    # ---------- shape and value helpers ----------

    @staticmethod
    def _ensure_velocity_shape(velocity: Any, param: Any) -> None:
        """
        Recursively check that the velocity structure matches the parameter structure.
        Raises ValueError if shapes differ.
        """
        if isinstance(param, list):
            if not isinstance(velocity, list) or len(velocity) != len(param):
                raise ValueError("Velocity shape mismatch.")
            for v, p in zip(velocity, param):
                SGDMomentum._ensure_velocity_shape(v, p)
        # For non-list, we assume it's a scalar – no further check needed.

    @staticmethod
    def _same_shape(a: Any, b: Any) -> bool:
        """
        Recursively compare the shapes of two nested lists.
        Returns True if they have identical structure, False otherwise.
        """
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(SGDMomentum._same_shape(x, y) for x, y in zip(a, b))
        # Both scalars or both lists? If one is list and the other not, mismatch.
        return not isinstance(a, list) and not isinstance(b, list)

    @staticmethod
    def _zeros_like(param: Any) -> Any:
        """Create a nested list of zeros with the same shape as `param`."""
        if isinstance(param, list):
            return [SGDMomentum._zeros_like(item) for item in param]
        return 0.0

    @staticmethod
    def _is_finite(value: Any) -> bool:
        """
        Recursively check that every element in a nested list is finite (not NaN/Inf).
        Returns True only if all elements are finite numbers.
        """
        if isinstance(value, list):
            return all(SGDMomentum._is_finite(x) for x in value)
        return math.isfinite(value)

    # ---------- core update ----------

    def _update_tensor(self, param: Any, grad: Any, key: str) -> None:
        """
        Update a single parameter tensor (weight or bias) using SGD with momentum.

        The parameter and gradient are nested lists. We recursively traverse them,
        compute new values, and then assign the result back in place.

        Steps:
            1. Validate shape of gradient vs parameter.
            2. Create velocity if not already present.
            3. Check that parameter, gradient, and velocity are all finite.
            4. Recursively compute new parameter and new velocity.
            5. Verify the results are finite.
            6. Mutate the original lists in place.
        """
        # 1. Shape check
        if not self._same_shape(param, grad):
            raise ValueError(f"Gradient shape mismatch for parameter {key}.")

        # 2. Initialise velocity if missing
        if key not in self.velocities:
            self.velocities[key] = self._zeros_like(param)

        velocity = self.velocities[key]
        self._ensure_velocity_shape(velocity, param)

        # 3. Pre‑flight finite check
        if not self._is_finite(param):
            raise ValueError(f"Parameter {key} contains non‑finite values.")
        if not self._is_finite(grad):
            raise ValueError(f"Gradient {key} contains non‑finite values.")
        if not self._is_finite(velocity):
            raise ValueError(f"Velocity {key} contains non‑finite values.")

        # 4. Recursive update
        def _update_rec(p, g, v):
            """Return (new_param, new_velocity) for this node."""
            if isinstance(p, list):
                # Recurse into each element
                new_p = []
                new_v = []
                for i in range(len(p)):
                    np, nv = _update_rec(p[i], g[i], v[i])
                    new_p.append(np)
                    new_v.append(nv)
                return new_p, new_v
            else:
                # Leaf: scalar update
                v_new = self.momentum * v + self.lr * g
                if not math.isfinite(v_new):
                    raise ValueError(f"Non‑finite velocity update for {key}.")
                p_new = p - v_new
                if not math.isfinite(p_new):
                    raise ValueError(f"Non‑finite parameter update for {key}.")
                return p_new, v_new

        new_param, new_velocity = _update_rec(param, grad, velocity)

        # 5. Post‑update finite check (defensive)
        if not self._is_finite(new_param):
            raise ValueError(f"Updated parameter {key} contains non‑finite values.")
        if not self._is_finite(new_velocity):
            raise ValueError(f"Updated velocity {key} contains non‑finite values.")

        # 6. Assign back in place – this mutates the original lists
        param[:] = new_param
        velocity[:] = new_velocity

    # ---------- public step method ----------

    def step(self, model) -> None:
        """
        Perform one optimisation step for all trainable parameters in the model.

        We automatically discover layers by looking for attributes that have
        `weight` and `weight_grad` (or `bias` and `bias_grad`). This avoids
        hard‑coding names like `conv1` or `fc1`, making the optimizer more flexible.
        """
        # Loop over all instance attributes (skip private ones)
        for attr_name in vars(model):
            if attr_name.startswith('_'):
                continue
            layer = getattr(model, attr_name)

            # If the layer has a weight parameter with a gradient, update it.
            if hasattr(layer, "weight") and hasattr(layer, "weight_grad"):
                self._update_tensor(layer.weight, layer.weight_grad, f"{attr_name}.weight")

            # Similarly for bias.
            if hasattr(layer, "bias") and hasattr(layer, "bias_grad"):
                self._update_tensor(layer.bias, layer.bias_grad, f"{attr_name}.bias")


# ---------- gradient scaling ----------

def _scale_tensor_inplace(tensor, scale: float):
    """
    Recursively multiply every element in a nested list by `scale`.
    The operation is performed in place to avoid creating new lists.
    """
    if isinstance(tensor, list):
        for i, item in enumerate(tensor):
            tensor[i] = _scale_tensor_inplace(item, scale)
        return tensor
    else:
        return tensor * scale


def scale_gradients(model, scale: float) -> None:
    """
    Scale all parameter gradients in the model by `scale`.

    This is used to average gradients over the batch size:
        scale = 1.0 / batch_size
    """
    for attr_name in vars(model):
        if attr_name.startswith('_'):
            continue
        layer = getattr(model, attr_name)
        if hasattr(layer, "weight_grad"):
            _scale_tensor_inplace(layer.weight_grad, scale)
        if hasattr(layer, "bias_grad"):
            _scale_tensor_inplace(layer.bias_grad, scale)


# ---------- training epoch ----------

def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer: SGDMomentum,
) -> tuple[float, float]:
    """
    Run one complete epoch of training over the given DataLoader.

    For each mini‑batch:
        1. Zero gradients (we want to accumulate only within this batch).
        2. For each sample in the batch:
            - Forward pass -> logits and closures.
            - Compute loss and its gradient w.r.t. logits.
            - Compute prediction for accuracy.
            - Backward pass: accumulates gradients in the model.
        3. Average the accumulated gradients by dividing by batch size.
        4. Tell the optimizer to update all parameters using the averaged gradients.
        5. Add the batch loss to the running total.

    Returns:
        (average_loss_per_sample, accuracy) for this epoch.
    """
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_imgs, batch_labels in dataloader:
        batch_size = len(batch_imgs)

        # Skip empty batches (should not happen, but safe).
        if batch_size == 0:
            continue

        # Safety check: ensure images and labels are aligned.
        if len(batch_labels) != batch_size:
            raise ValueError(
                f"Batch images and labels have mismatched lengths: "
                f"{batch_size} vs {len(batch_labels)}"
            )

        # 1. Zero gradients once per batch.
        model.zero_grad()

        batch_loss_sum = 0.0

        # 2. Process each sample in the batch.
        for img, label in zip(batch_imgs, batch_labels):
            # Forward pass: get logits and the closures needed for backprop.
            logits, closures = model.forward(img)

            # Compute loss and the function that returns dL/dlogits.
            loss, loss_backward = criterion.forward(logits, label)
            batch_loss_sum += loss

            # Determine predicted class (argmax of logits) for accuracy.
            pred_class = max(range(len(logits)), key=lambda i: logits[i])
            if pred_class == label:
                correct += 1
            total_samples += 1

            # Get the gradient of loss w.r.t. logits: p - y (for Cross‑Entropy).
            dL_dlogits = loss_backward()

            # Backpropagate through the model; this accumulates gradients.
            model.backward(dL_dlogits, closures)

        # 3. Average the accumulated gradients by batch size.
        scale_gradients(model, 1.0 / batch_size)

        # 4. Update parameters once using the averaged gradients.
        optimizer.step(model)

        # 5. Add this batch's total loss (sum of per‑sample losses) to running total.
        total_loss += batch_loss_sum

    # After all batches, compute averages.
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy