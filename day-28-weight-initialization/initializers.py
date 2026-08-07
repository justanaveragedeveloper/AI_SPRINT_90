"""
Day 28: Weight Initialization Strategies (Xavier / He)
=======================================================

Provides statistically sound weight initializers and scaled versions of
Neuron, Layer, and MLP that preserve variance across deep networks.
Fully compatible with the autograd engine (Value) and base NN classes
from Days 24–25.
"""

import math
import os
import random
import sys
from collections.abc import Callable

# Add paths to previous days' modules (adjust if needed)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../day-24-autograd"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../day-25-nn-from-scratch")
    )
)

from engine import Value  # noqa: I001
from nn import Layer, MLP, Neuron

# -------------------------------------------------------------------
# Validation & Constants
# -------------------------------------------------------------------

_VALID_ACTIVATIONS = {"tanh", "sigmoid", "relu", "linear"}


def validate_activation(act: str) -> None:
    """Raise ValueError if activation name is not recognised."""
    if act.lower() not in _VALID_ACTIVATIONS:
        raise ValueError(f"Invalid activation '{act}'. Supported: {_VALID_ACTIVATIONS}")


# -------------------------------------------------------------------
# Weight Initialisation Factory
# -------------------------------------------------------------------


def initialize_weights(
    mode: str = "xavier_uniform", nin: int = 1, nouts: int | None = None
) -> Callable[[], float] | None:
    """
    Return a callable that generates a single weight value from the
    specified initialisation distribution.

    Parameters
    ----------
    mode : str
        One of: 'default', 'xavier_uniform', 'xavier_normal',
        'he_uniform', 'he_normal'.
    nin : int
        Number of inputs to the neuron (fan‑in).
    nouts : int, optional
        Number of outputs from the neuron (fan‑out). Required for Xavier
        modes; defaults to 1 if not given.

    Returns
    -------
    callable or None
        A zero‑argument function that returns a random weight sample.
        Returns None when mode == 'default', indicating that the parent
        class's default uniform[-1,1] initialisation should be used.

    Raises
    ------
    ValueError
        If mode is unknown or dimensions are non‑positive.
    """
    if nin <= 0:
        raise ValueError(f"Input dimension nin must be positive, got {nin}")
    if nouts is not None and nouts <= 0:
        raise ValueError(f"Output dimension nouts must be positive, got {nouts}")

    n_out = nouts if nouts is not None else 1

    if mode == "default":
        return None

    if mode == "xavier_uniform":
        # Uniform(-a, a) with a = sqrt(6 / (nin + nout))
        limit = math.sqrt(6.0 / (nin + n_out))
        return lambda: random.uniform(-limit, limit)

    if mode == "xavier_normal":
        # Normal(0, std) with std = sqrt(2 / (nin + nout))
        std = math.sqrt(2.0 / (nin + n_out))
        return lambda: random.gauss(0.0, std)

    if mode == "he_uniform":
        # Uniform(-a, a) with a = sqrt(6 / nin)
        limit = math.sqrt(6.0 / nin)
        return lambda: random.uniform(-limit, limit)

    if mode == "he_normal":
        # Normal(0, std) with std = sqrt(2 / nin)
        std = math.sqrt(2.0 / nin)
        return lambda: random.gauss(0.0, std)

    raise ValueError(f"Unknown initialisation mode: {mode}")


# -------------------------------------------------------------------
# Scaled Neuron, Layer, MLP
# -------------------------------------------------------------------


class ScaledNeuron(Neuron):
    """
    Neuron that applies an explicit weight initialisation strategy.

    The parent class's default uniform[-1,1] initialisation is replaced
    if a sampler is provided. Bias remains zero.
    """

    def __init__(
        self,
        nin: int,
        nonlin: bool = True,
        activation: str = "tanh",
        init_mode: str = "xavier_uniform",
        nout: int = 1,
    ):
        validate_activation(activation)
        # Parent expects (num_inputs, activation) – pass the string, not nonlin
        super().__init__(nin, activation)

        # Override weights with the chosen initialisation
        sampler = initialize_weights(init_mode, nin=nin, nouts=nout)
        if sampler is not None:
            self.w = [Value(sampler()) for _ in range(nin)]
            self.b = Value(0.0)


class ScaledLayer(Layer):
    """
    Layer composed of ScaledNeurons, each initialised with the chosen
    strategy. Supports the same keyword arguments as Layer.
    """

    def __init__(
        self, nin: int, nout: int, init_mode: str = "xavier_uniform", **kwargs
    ):
        if nin <= 0 or nout <= 0:
            raise ValueError(
                f"Layer dimensions must be positive: nin={nin}, nout={nout}"
            )
        # Set attributes required by parent's __call__
        self.input_size = nin
        self.output_size = nout

        # Override the neurons list with scaled versions
        self.neurons = [
            ScaledNeuron(nin, init_mode=init_mode, nout=nout, **kwargs)
            for _ in range(nout)
        ]


class ScaledMLP(MLP):
    """
    Multi‑layer perceptron with layer‑by‑layer weight initialisation.

    If init_mode == 'auto', the method is chosen automatically:
      - He normal for ReLU activations
      - Xavier normal for tanh, sigmoid, or linear activations

    All other features (forward pass, parameters, etc.) are inherited.
    """

    def __init__(
        self,
        nin: int,
        nouts: list[int],
        init_mode: str = "xavier_uniform",
        activations: list[str] | None = None,
    ):
        if not nouts:
            raise ValueError("nouts cannot be empty")
        if nin <= 0:
            raise ValueError(f"Input dimension nin must be positive, got {nin}")

        # Validate activations
        if activations is not None:
            if len(activations) != len(nouts):
                raise ValueError(
                    f"Length of activations ({len(activations)}) must match "
                    f"number of layers ({len(nouts)})"
                )
            for act in activations:
                validate_activation(act)
        else:
            # Default: tanh for all layers (including output, because no activation list given)
            activations = ["tanh"] * len(nouts)

        sz = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            act = activations[i]

            # Auto‑selection logic
            current_mode = init_mode
            if init_mode == "auto":
                current_mode = "he_normal" if act.lower() == "relu" else "xavier_normal"

            # Decide nonlin:
            # If activations was explicitly provided, all layers are non-linear.
            # Otherwise, the last layer is linear (no activation).
            if activations is not None:
                nonlin = True
            else:
                nonlin = (i != len(nouts) - 1)

            self.layers.append(
                ScaledLayer(
                    sz[i],
                    sz[i + 1],
                    nonlin=nonlin,
                    activation=act,
                    init_mode=current_mode,
                )
            )

        # Ensure the parent's input_size is set (required by MLP.__call__)
        self.input_size = nin