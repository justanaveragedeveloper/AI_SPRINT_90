"""
Neural Network framework built on top of the scalar Autograd Engine (Day 24).

This module defines the fundamental building blocks: Module, Neuron, Layer, MLP.
It provides parameter management, forward pass, and compatibility with reverse-mode
autodiff. All operations preserve the computational graph.

=============================================================================
DESIGN VERIFICATION CHECKLIST (matches the specification)
=============================================================================
✓ Module exposes parameters()
✓ Module exposes zero_grad()
✓ Neuron stores weights as Value objects
✓ Neuron stores bias as Value object
✓ Neuron computes Σ(w*x) + b using Value operations
✓ Neuron supports 'tanh', 'relu', and 'linear' activations
✓ Layer manages multiple neurons and forwards vector inputs
✓ MLP composes multiple Layers sequentially
✓ Every module returns a flat list of all trainable parameters
✓ zero_grad() resets all parameter gradients to zero
✓ Training loop: forward → loss → zero_grad → backward → SGD update
✓ All arithmetic preserves the computational graph
✓ Input validation for dimensions and activation names
✓ No graph‑breaking operations
=============================================================================
"""

import os
import random
import sys

# ----------------------------------------------------------------------
# Import Day 24 Autograd Engine kernel.
# We add the sibling folder "day-24-autograd" to sys.path so we can
# import engine.py directly.
# ----------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_autograd_dir = os.path.abspath(os.path.join(_current_dir, "..", "day-24-autograd"))
if _autograd_dir not in sys.path:
    sys.path.insert(0, _autograd_dir)

# Suppress Pylance warning with `# type: ignore`
try:
    from engine import Value  # type: ignore
except ImportError as e:
    raise ImportError(
        f"Could not import 'Value' from the Day 24 autograd engine. "
        f"Expected to find engine.py in: {_autograd_dir}\n"
        "Please make sure the folder exists and contains engine.py."
    ) from e


class Module:
    """
    Base class for all neural network modules.

    Provides common interface for parameter collection and gradient resetting.
    """

    def zero_grad(self) -> None:
        """
        Reset gradients of all trainable parameters to zero.

        This must be called before each backward pass to prevent accumulation
        across training steps.
        """
        for param in self.parameters():
            # engine.py uses 'gradient' attribute, not 'grad'
            param.gradient = 0.0

    def parameters(self) -> list["Value"]:
        """
        Return a flat list of all trainable Value parameters.

        Subclasses must override this to return their specific parameters.
        """
        return []


class Neuron(Module):
    """
    A single artificial neuron with weights, bias, and configurable activation.

    Computes: y = f(Σ(w_i * x_i) + b), where f is one of 'tanh', 'relu', or 'linear'.
    All weights and biases are stored as Value objects, preserving the autograd graph.
    """

    def __init__(self, num_inputs: int, activation: str = "tanh") -> None:
        """
        Initialize neuron with random weights and zero bias.

        Args:
            num_inputs: Number of input features (must be positive).
            activation: Activation function to apply. Must be one of 'tanh', 'relu', 'linear'.
                       Default is 'tanh'.

        Raises:
            ValueError: If num_inputs <= 0 or activation is not supported.
        """
        if num_inputs <= 0:
            raise ValueError(f"Number of inputs must be positive, got {num_inputs}")
        valid_activations = {"tanh", "relu", "linear"}
        if activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got {activation}"
            )

        self.num_inputs = num_inputs
        self.activation = activation

        # Weights initialized uniformly in [-1, 1]
        self.weights = [Value(random.uniform(-1.0, 1.0)) for _ in range(num_inputs)]
        self.bias = Value(0.0)  # bias initialised to 0

    def __call__(self, inputs: list[Value]) -> Value:
        """
        Perform forward pass: sum(weights * inputs) + bias, then activation.

        Args:
            inputs: List of input Value objects. Length must match num_inputs.

        Returns:
            Output Value (scalar).

        Raises:
            TypeError: If inputs is not a list or not all elements are Value.
            ValueError: If length mismatch.
        """
        if not isinstance(inputs, list):
            raise TypeError(f"Input must be a list, got {type(inputs)}")
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        if not all(isinstance(v, Value) for v in inputs):
            raise TypeError("All inputs must be Value objects")

        # Compute weighted sum: sum(weights_i * inputs_i) + bias
        weighted_sum = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)

        # Apply activation
        if self.activation == "tanh":
            return weighted_sum.tanh()
        if self.activation == "relu":
            return weighted_sum.relu()
        # 'linear'
        return weighted_sum

    def parameters(self) -> list[Value]:
        """Return all trainable parameters (weights and bias)."""
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f"{self.activation.capitalize()}Neuron({self.num_inputs})"


class Layer(Module):
    """
    A layer consisting of multiple independent neurons.

    Takes an input vector of size input_size and produces an output vector of size
    output_size. If output_size == 1, returns a single Value (scalar) instead of a list.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        """
        Create a layer with output_size neurons, each with input_size inputs.

        Args:
            input_size: Number of input features per neuron (must be positive).
            output_size: Number of neurons in the layer (must be positive).
            **kwargs: Additional arguments passed to each Neuron (e.g., activation).

        Raises:
            ValueError: If input_size or output_size <= 0.
        """
        if input_size <= 0:
            raise ValueError(f"Number of inputs must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"Number of neurons must be positive, got {output_size}")

        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size, **kwargs) for _ in range(output_size)]

    def __call__(self, inputs: list[Value]) -> Value | list[Value]:
        """
        Forward pass: feed inputs to all neurons.

        Args:
            inputs: Input list of Value objects. Length must match input_size.

        Returns:
            If output_size == 1, a single Value; otherwise a list of Value outputs.

        Raises:
            TypeError: If inputs is not a list or not all elements are Value.
            ValueError: If length mismatch.
        """
        if not isinstance(inputs, list):
            raise TypeError(f"Input must be a list, got {type(inputs)}")
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        if not all(isinstance(v, Value) for v in inputs):
            raise TypeError("All inputs must be Value objects")

        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        """Flatten and return all parameters from all neurons in the layer."""
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Multi-Layer Perceptron composed of a sequence of Layers.

    The output of each layer is fed as input to the next.
    By default, hidden layers use 'tanh' activation, and the last layer uses 'linear'.
    You can override activation per layer by passing `activation` in **kwargs
    (but note it will apply to all layers uniformly if provided).
    """

    def __init__(self, input_size: int, layer_sizes: list[int], **kwargs) -> None:
        """
        Build an MLP with specified layer sizes.

        Args:
            input_size: Input dimension (must be positive).
            layer_sizes: List of output sizes for each layer. Each element must be positive.
            **kwargs: Additional arguments passed to each Layer (e.g., activation).
                     If not specified, hidden layers use 'tanh' and the last uses 'linear'.

        Raises:
            ValueError: If input_size <= 0 or any layer_sizes <= 0 or layer_sizes empty.
        """
        if input_size <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_size}")
        if not layer_sizes:
            raise ValueError("Layer sizes list cannot be empty")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive")

        # Store for validation
        self.input_size = input_size
        self.layer_sizes = layer_sizes

        # Build layers: each layer takes the previous layer's output size as input
        sizes = [input_size] + layer_sizes

        # Determine activation for each layer.
        # If kwargs contains 'activation', it will be applied to all layers.
        # Otherwise, we use 'tanh' for all but the last, which is 'linear'.
        if "activation" in kwargs:
            # Apply the same activation to every layer (user override)
            self.layers = [
                Layer(sizes[i], sizes[i + 1], **kwargs) for i in range(len(layer_sizes))
            ]
        else:
            # Default: tanh for hidden, linear for output
            self.layers = [
                Layer(
                    sizes[i],
                    sizes[i + 1],
                    activation=("tanh" if i != len(layer_sizes) - 1 else "linear"),
                )
                for i in range(len(layer_sizes))
            ]

    def __call__(self, inputs: list[Value]) -> Value | list[Value]:
        """
        Forward pass through all layers sequentially.

        Args:
            inputs: Input list of Value objects. Length must match input dimension.

        Returns:
            The final output (Value if last layer has one neuron, else list).

        Raises:
            TypeError: If inputs is not a list or not all elements are Value.
            ValueError: If length mismatch.
        """
        if not isinstance(inputs, list):
            raise TypeError(f"Input must be a list, got {type(inputs)}")
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        if not all(isinstance(v, Value) for v in inputs):
            raise TypeError("All inputs must be Value objects")

        # Forward through layers
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self) -> list[Value]:
        """Collect all parameters from all layers recursively."""
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"