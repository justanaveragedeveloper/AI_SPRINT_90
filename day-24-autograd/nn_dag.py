"""nn_dag.py – Neural network primitives built on the Value graph.

Implements a simple MLP with configurable neurons, layers, and activations.
All parameters are `Value` objects and participate in autograd.
"""

from __future__ import annotations

import random

from engine import Value

# Set seed for reproducibility
random.seed(1337)


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> list[Value]:
        """Return a list of all trainable parameters."""
        return []


class Neuron(Module):
    """
    A single neuron with weights, bias, and an activation function.

    Args:
        num_inputs (int): Number of input connections.
        activation (str): Activation function – 'tanh', 'relu', or 'none'.
    """

    _SUPPORTED_ACTIVATIONS = ("tanh", "relu", "none")  # immutable tuple

    def __init__(self, num_inputs: int, activation: str = "tanh") -> None:
        if activation not in self._SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Supported: {self._SUPPORTED_ACTIVATIONS}"
            )
        self.weights: list[Value] = [
            Value(random.uniform(-1.0, 1.0)) for _ in range(num_inputs)
        ]
        self.bias: Value = Value(0.0)
        self.activation = activation

    def __call__(self, inputs: list[Value]) -> Value:
        """
        Forward pass: computes sum(weights_i * inputs_i) + bias and applies activation.

        Raises:
            ValueError: if len(inputs) != number of weights.
        """
        if len(inputs) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(inputs)}")
        # Sum over inputs
        pre_activation = self.bias
        for weight, input_value in zip(self.weights, inputs):
            pre_activation += weight * input_value

        if self.activation == "tanh":
            return pre_activation.tanh()
        elif self.activation == "relu":
            return pre_activation.relu()
        else:  # 'none'
            return pre_activation

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


class Layer(Module):
    """
    A layer consisting of multiple neurons.

    Args:
        num_inputs (int): Number of inputs.
        num_outputs (int): Number of neurons.
        activation (str): Activation function for all neurons.
    """

    def __init__(
        self, num_inputs: int, num_outputs: int, activation: str = "tanh"
    ) -> None:
        self.neurons: list[Neuron] = [
            Neuron(num_inputs, activation) for _ in range(num_outputs)
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        """Forward pass through all neurons in the layer."""
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP(Module):
    """
    Multi‑Layer Perceptron with multiple hidden layers.

    Args:
        num_inputs (int): Input size.
        output_sizes (list[int]): List of output sizes for each layer.
        activation (str): Activation function (applied to all hidden layers).
    """

    def __init__(
        self, num_inputs: int, output_sizes: list[int], activation: str = "tanh"
    ) -> None:
        sizes = [num_inputs] + output_sizes
        self.layers: list[Layer] = [
            Layer(sizes[i], sizes[i + 1], activation) for i in range(len(output_sizes))
        ]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        """Forward pass through all layers."""
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]
