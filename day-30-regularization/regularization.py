"""
Day 30: Regularization (L1, L2, Dropout) & Gradient Clipping

This module provides the fundamental building blocks to help neural networks
generalise better and train more stably. It implements:

- L1 (Lasso) and L2 (Ridge) weight penalties – to keep weights small and sparse.
- Inverted dropout – to prevent neurons from co‑adapting too much.
- Gradient clipping – to stop gradients from exploding during training.

Everything works with plain NumPy arrays – no autograd or computational graphs.
"""

import numpy as np

# ----------------------------------------------------------------------
#  Helper functions for checking inputs
# ----------------------------------------------------------------------


def _check_scalar(value, name, allow_zero=True):
    """
    Make sure a value is a real number, finite, and not negative.
    If `allow_zero` is False, we also require it to be strictly positive.
    """
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value)}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    if not allow_zero and value == 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _check_array(arr, name, require_float=True):
    """Verify that `arr` is a NumPy array (and optionally a float array)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(arr)}")
    if require_float and not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must have floating‑point dtype, got {arr.dtype}")


def _check_all_finite(arr, name):
    """Raise an error if `arr` contains NaN or infinity."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non‑finite (NaN/Inf) values")


# ----------------------------------------------------------------------
#  Inverted Dropout
# ----------------------------------------------------------------------


class Dropout:
    """
    Inverted Dropout layer – used during training to randomly "drop" neurons.

    During training, each activation is multiplied by a random mask:
        mask = 1 with probability (1 - drop_rate)  (keep_prob)
        mask = 0 otherwise

    To keep the expected output the same as during evaluation, we **scale up**
    the surviving activations by 1 / keep_prob. This is called "inverted dropout".

    During evaluation, the layer simply passes the input through unchanged.
    """

    def __init__(self, drop_rate=0.5):
        """
        Parameters
        ----------
        drop_rate : float, between 0 and 1 (default 0.5)
            Probability that a neuron is dropped (zeroed out).
        """
        _check_scalar(drop_rate, "drop_rate", allow_zero=True)
        if drop_rate >= 1.0:
            raise ValueError(f"drop_rate must be < 1, got {drop_rate}")
        self.drop_rate = drop_rate  # the dropout probability
        self.mask = None  # the mask used in the last forward pass
        self.training = True  # are we in training mode?
        self._mask_is_valid = False  # has a training forward been run?

    def train(self):
        """Switch to training mode – dropout will be active."""
        self.training = True
        # Any old mask from a previous forward pass is now stale.
        self._mask_is_valid = False
        self.mask = None

    def eval(self):
        """Switch to evaluation mode – dropout will be turned off."""
        self.training = False

    def forward(self, x):
        """
        Run a forward pass through the dropout layer.

        In training mode, we generate a random mask and scale it.
        In evaluation mode, we just return x unchanged.

        Returns
        -------
        y : numpy array
            The result after applying dropout (or identity).
        """
        _check_array(x, "x")
        _check_all_finite(x, "Input")

        # If we're not training or drop_rate is zero, just pass through.
        if not self.training or self.drop_rate == 0.0:
            return x

        keep_prob = 1.0 - self.drop_rate
        # Create a mask: each entry is 1/keep_prob with probability keep_prob,
        # otherwise 0. This automatically does both the dropping and the scaling.
        self.mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
        self._mask_is_valid = True
        return x * self.mask

    def backward(self, dout):
        """
        Backward pass: propagate gradients through the dropout layer.

        We use the **same** mask that was used during the forward pass.
        This is critical – if we generated a new mask, the gradients would be wrong.

        Returns
        -------
        dx : numpy array
            Gradient with respect to the input of this layer.
        """
        _check_array(dout, "dout")
        _check_all_finite(dout, "Upstream gradient")

        # If dropout was inactive during forward, gradients pass through unchanged.
        if not self.training or self.drop_rate == 0.0:
            return dout

        # We must have a valid mask from a forward pass.
        if not self._mask_is_valid:
            raise RuntimeError(
                "backward() called without a valid training forward pass."
            )

        # The mask shape must match the incoming gradient shape.
        if dout.shape != self.mask.shape:
            raise ValueError(
                f"dout shape {dout.shape} does not match mask shape {self.mask.shape}"
            )

        # Multiply the upstream gradient by the same mask to backpropagate.
        return dout * self.mask


# ----------------------------------------------------------------------
#  L1 and L2 Regularisation Penalties
# ----------------------------------------------------------------------


def compute_l1_penalty(params, l1_lambda):
    """
    L1 regularisation penalty = λ * sum(|w_i|) over all parameters.

    This penalty encourages sparsity – many weights become exactly zero.
    """
    if not isinstance(params, (list, tuple)):
        raise TypeError("params must be a list or tuple")
    _check_scalar(l1_lambda, "l1_lambda", allow_zero=True)

    if l1_lambda == 0.0:
        return 0.0

    total = 0.0
    for i, w in enumerate(params):
        _check_array(w, f"params[{i}]")
        _check_all_finite(w, f"params[{i}]")
        total += np.sum(np.abs(w))
    return l1_lambda * total


def compute_l2_penalty(params, l2_lambda):
    """
    L2 regularisation penalty = (λ/2) * sum(w_i²) over all parameters.

    This penalty keeps weights small, preventing any single weight from becoming
    too large. The 1/2 factor makes the gradient simply λ * w.
    """
    if not isinstance(params, (list, tuple)):
        raise TypeError("params must be a list or tuple")
    _check_scalar(l2_lambda, "l2_lambda", allow_zero=True)

    if l2_lambda == 0.0:
        return 0.0

    total = 0.0
    for i, w in enumerate(params):
        _check_array(w, f"params[{i}]")
        _check_all_finite(w, f"params[{i}]")
        total += np.sum(w * w)
    return 0.5 * l2_lambda * total


def apply_l1_l2_gradients(params, grads, l1_lambda=0.0, l2_lambda=0.0):
    """
    Add the gradient contributions from L1 and L2 regularisation to the
    existing parameter gradients (in‑place).

    L1 gradient:  λ * sign(w)
    L2 gradient:  λ * w

    This is usually called right after computing the normal loss gradients,
    before the optimiser updates the weights.
    """
    if not isinstance(params, (list, tuple)):
        raise TypeError("params must be a list or tuple")
    if not isinstance(grads, (list, tuple)):
        raise TypeError("grads must be a list or tuple")

    _check_scalar(l1_lambda, "l1_lambda", allow_zero=True)
    _check_scalar(l2_lambda, "l2_lambda", allow_zero=True)

    if len(params) != len(grads):
        raise ValueError(f"params length {len(params)} != grads length {len(grads)}")

    for i, (w, dw) in enumerate(zip(params, grads)):
        _check_array(w, f"params[{i}]")
        _check_array(dw, f"grads[{i}]")
        _check_all_finite(w, f"params[{i}]")
        _check_all_finite(dw, f"grads[{i}]")

        if w.shape != dw.shape:
            raise ValueError(
                f"param[{i}] shape {w.shape} != grad[{i}] shape {dw.shape}"
            )

        if l2_lambda != 0.0:
            dw += l2_lambda * w  # L2 gradient
        if l1_lambda != 0.0:
            dw += l1_lambda * np.sign(w)  # L1 gradient (sign is 0 at zero)


# ----------------------------------------------------------------------
#  Gradient Clipping
# ----------------------------------------------------------------------


def clip_gradients_value(grads, clip_value):
    """
    Clip each element of every gradient array to the range [-clip_value, clip_value].

    This is a simple element‑wise clamp that prevents any single gradient
    component from becoming too large. `clip_value` must be strictly positive.
    """
    if not isinstance(grads, (list, tuple)):
        raise TypeError("grads must be a list or tuple")
    _check_scalar(clip_value, "clip_value", allow_zero=False)  # must be > 0

    for i, dw in enumerate(grads):
        _check_array(dw, f"grads[{i}]")
        _check_all_finite(dw, f"grads[{i}]")
        # In‑place clipping
        np.clip(dw, -clip_value, clip_value, out=dw)


def _stable_l2_norm(grads):
    """
    Compute the L2 norm of all gradients combined, but in a way that avoids
    overflow if any gradient value is extremely large (near the float64 limit).

    The trick: find the largest absolute value `m`, then compute:
        norm = m * sqrt( sum( (g_i / m)² ) )
    This way we never square a number larger than 1, so overflow is impossible.
    """
    # Find the maximum absolute value across all arrays
    max_abs = 0.0
    for dw in grads:
        if dw.size > 0:
            max_abs = max(max_abs, float(np.max(np.abs(dw))))

    if max_abs == 0.0:
        return 0.0

    # Scale down, compute norm on the scaled values, then scale back up
    scaled_sq = 0.0
    for dw in grads:
        if dw.size > 0:
            scaled = dw / max_abs
            scaled_sq += np.sum(scaled * scaled)
    return max_abs * np.sqrt(scaled_sq)


def clip_gradients_norm(grads, max_norm):
    """
    Apply global norm clipping.

    If the total L2 norm of all gradients exceeds `max_norm`, we scale every
    gradient array by the same factor: max_norm / original_norm.
    This preserves the direction of the gradient vector while ensuring its
    magnitude is bounded.

    Returns
    -------
    original_norm : float
        The L2 norm of the gradients before any clipping.
    """
    if not isinstance(grads, (list, tuple)):
        raise TypeError("grads must be a list or tuple")
    _check_scalar(max_norm, "max_norm", allow_zero=False)  # must be > 0

    for i, dw in enumerate(grads):
        _check_array(dw, f"grads[{i}]")
        _check_all_finite(dw, f"grads[{i}]")

    original_norm = _stable_l2_norm(grads)

    if original_norm > max_norm:
        scale = max_norm / original_norm
        for dw in grads:
            dw *= scale

    return original_norm
