# ruff: noqa: I001
"""
PyTest suite for the neural network framework (Day 25).

Covers:
- Forward pass correctness for all activation types
- Parameter counting and recursive collection
- zero_grad() functionality
- SGD update step
- Overfitting on tiny dataset with both tanh and relu (using a fixed seed)
- Input validation and error handling
- Edge cases (single neuron, empty MLP, etc.)
"""

import os
import random
import sys

import pytest

# ----------------------------------------------------------------------
# Add the sibling folder "day-24-autograd" to sys.path so we can
# import engine.py. This must come before importing 'engine'.
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
        f"Expected to find engine.py in: {_autograd_dir}"
    ) from e

from nn import Layer, MLP, Module, Neuron

# ----------------------------------------------------------------------
# Helper to get the numeric value from a Value object
# engine.py uses 'value' attribute, not 'data'
# ----------------------------------------------------------------------
def get_param_value(param: Value):
    """Return the scalar value of a Value parameter."""
    return param.value

# ----------------------------------------------------------------------
# Set a fixed seed for reproducible tests (overfitting tests)
# ----------------------------------------------------------------------
random.seed(42)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def all_grads_are_zero(module: Module) -> bool:
    """Return True if all parameters have gradient == 0.0."""
    return all(param.gradient == 0.0 for param in module.parameters())


def all_grads_are_nonzero(module: Module) -> bool:
    """Return True if all parameters have gradient != 0.0."""
    return all(param.gradient != 0.0 for param in module.parameters())


# ----------------------------------------------------------------------
# Tests for Neuron
# ----------------------------------------------------------------------


def test_neuron_forward_and_params():
    neuron = Neuron(num_inputs=3, activation="tanh")
    inputs = [Value(1.0), Value(-2.0), Value(0.5)]
    output = neuron(inputs)
    assert isinstance(output, Value)
    assert len(neuron.parameters()) == 4  # 3 weights + 1 bias

    # Test linear neuron
    linear_neuron = Neuron(num_inputs=2, activation="linear")
    linear_inputs = [Value(0.5), Value(-0.5)]
    linear_output = linear_neuron(linear_inputs)
    assert isinstance(linear_output, Value)
    # Check that backward exists (graph built)
    assert hasattr(linear_output, 'backward'), "Value object missing 'backward' method"

    # Test ReLU neuron
    relu_neuron = Neuron(num_inputs=2, activation="relu")
    relu_inputs = [Value(0.5), Value(-0.5)]
    relu_output = relu_neuron(relu_inputs)
    assert isinstance(relu_output, Value)
    assert hasattr(relu_output, 'backward')


def test_neuron_invalid_input():
    neuron = Neuron(num_inputs=2)
    with pytest.raises(ValueError, match="Expected 2 inputs, got 3"):
        neuron([Value(1.0), Value(2.0), Value(3.0)])
    with pytest.raises(TypeError, match="must be Value objects"):
        neuron([1.0, 2.0])
    with pytest.raises(TypeError, match="must be a list"):
        neuron(Value(1.0))  # type: ignore


def test_neuron_invalid_activation():
    with pytest.raises(ValueError, match="activation must be one of"):
        Neuron(num_inputs=2, activation="sigmoid")


# ----------------------------------------------------------------------
# Tests for Layer
# ----------------------------------------------------------------------


def test_layer_forward_and_params():
    layer = Layer(input_size=3, output_size=4, activation="tanh")
    inputs = [Value(1.0), Value(-2.0), Value(0.5)]
    outputs = layer(inputs)
    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert all(isinstance(v, Value) for v in outputs)
    # Parameters: each neuron has 3 weights + 1 bias = 4; 4 neurons => 16
    assert len(layer.parameters()) == 16

    # Layer with output_size=1 returns scalar
    single_layer = Layer(input_size=2, output_size=1, activation="linear")
    single_inputs = [Value(0.5), Value(-0.5)]
    single_output = single_layer(single_inputs)
    assert isinstance(single_output, Value)


def test_layer_invalid_input():
    layer = Layer(input_size=2, output_size=3)
    with pytest.raises(ValueError, match="Expected 2 inputs, got 1"):
        layer([Value(1.0)])
    with pytest.raises(TypeError, match="must be Value objects"):
        layer([1.0, 2.0])


# ----------------------------------------------------------------------
# Tests for MLP
# ----------------------------------------------------------------------


def test_mlp_forward_and_params():
    model = MLP(input_size=3, layer_sizes=[4, 4, 1])
    inputs = [Value(2.0), Value(3.0), Value(-1.0)]
    output = model(inputs)
    assert isinstance(output, Value)

    # Parameter count: (3*4 + 4) + (4*4 + 4) + (4*1 + 1) = 16 + 20 + 5 = 41
    assert len(model.parameters()) == 41

    # MLP with multiple output neurons (last layer size > 1) returns list
    model2 = MLP(input_size=2, layer_sizes=[3, 2])
    inputs2 = [Value(0.1), Value(0.2)]
    outputs2 = model2(inputs2)
    assert isinstance(outputs2, list)
    assert len(outputs2) == 2

    # MLP with custom activation (all layers use ReLU)
    model_relu = MLP(input_size=2, layer_sizes=[3, 1], activation="relu")
    out_relu = model_relu([Value(0.1), Value(0.2)])
    assert isinstance(out_relu, Value)


def test_mlp_invalid_input():
    model = MLP(input_size=2, layer_sizes=[2, 1])
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        model([Value(1.0)])
    with pytest.raises(TypeError, match="must be Value objects"):
        model([1.0, 2.0])


# ----------------------------------------------------------------------
# zero_grad test
# ----------------------------------------------------------------------


def test_zero_grad():
    model = MLP(input_size=2, layer_sizes=[3, 1])
    inputs = [Value(0.5), Value(-0.5)]
    target = Value(1.0)

    prediction = model(inputs)
    # Use addition with -1 to avoid relying on __sub__ (not implemented in engine.py)
    loss = (prediction + (target * -1)) ** 2
    loss.backward()

    # After backward, all grads should be non-zero
    assert all_grads_are_nonzero(model)

    model.zero_grad()
    assert all_grads_are_zero(model)


# ----------------------------------------------------------------------
# parameters recursion and uniqueness
# ----------------------------------------------------------------------


def test_parameters_recursive_no_duplicates():
    model = MLP(input_size=2, layer_sizes=[2, 2])
    params = model.parameters()
    param_ids = [id(p) for p in params]
    assert len(param_ids) == len(set(param_ids)), "Duplicate parameters found"

    total_expected = sum(len(layer.parameters()) for layer in model.layers)
    assert len(params) == total_expected


# ----------------------------------------------------------------------
# SGD update test
# ----------------------------------------------------------------------


def test_sgd_update():
    model = MLP(input_size=1, layer_sizes=[1])  # Single neuron, linear
    inputs = [Value(2.0)]
    target = Value(4.0)
    learning_rate = 0.01

    initial_data = [get_param_value(p) for p in model.parameters()]

    prediction = model(inputs)
    # Use addition with -1 to avoid __sub__
    loss = (prediction + (target * -1)) ** 2
    model.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.value -= learning_rate * param.gradient

    after_data = [get_param_value(p) for p in model.parameters()]
    assert after_data != initial_data


# ----------------------------------------------------------------------
# Overfitting convergence tests (tanh and relu) – uses relative loss drop
# ----------------------------------------------------------------------


def test_mlp_overfitting_convergence_tanh():
    """Validates that MLP with tanh can overfit a tiny XOR-like dataset."""
    _run_overfitting_test(activation="tanh")


def test_mlp_overfitting_convergence_relu():
    """Validates that MLP with ReLU can overfit a tiny XOR-like dataset."""
    _run_overfitting_test(activation="relu")


def _run_overfitting_test(activation: str):
    """Helper to run overfitting test with a given activation."""
    model = MLP(input_size=2, layer_sizes=[16, 1], activation=activation)

    raw_inputs = [
        [2.0, 3.0],
        [3.0, -1.0],
        [-1.0, -2.0],
        [1.0, 1.0],
    ]
    targets = [1.0, -1.0, -1.0, 1.0]

    learning_rate = 0.05
    initial_loss = None
    final_loss = None

    for epoch in range(50):
        inputs_values = [[Value(v) for v in x] for x in raw_inputs]
        predictions = [model(x) for x in inputs_values]

        # Use addition with -1 to avoid __sub__
        loss = sum(
            (pred + (Value(target) * -1)) ** 2
            for pred, target in zip(predictions, targets)
        )

        if epoch == 0:
            initial_loss = get_param_value(loss)

        model.zero_grad()
        loss.backward()

        for param in model.parameters():
            param.value -= learning_rate * param.gradient

        final_loss = get_param_value(loss)

    # Check that loss decreased substantially (at least 80% reduction)
    assert final_loss < initial_loss
    assert (
        final_loss < initial_loss * 0.2
    ), f"Loss did not drop enough with {activation}: initial={initial_loss:.4f}, final={final_loss:.4f}"


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_single_neuron_layer():
    layer = Layer(input_size=2, output_size=1, activation="tanh")
    inputs = [Value(0.1), Value(0.2)]
    output = layer(inputs)
    assert isinstance(output, Value)
    assert len(layer.parameters()) == 3


def test_empty_mlp_raises():
    with pytest.raises(ValueError, match="Layer sizes list cannot be empty"):
        MLP(input_size=2, layer_sizes=[])


def test_negative_input_dimension_raises():
    with pytest.raises(ValueError, match="must be positive"):
        Neuron(num_inputs=-1)
    with pytest.raises(ValueError, match="must be positive"):
        Layer(input_size=-1, output_size=2)
    with pytest.raises(ValueError, match="must be positive"):
        MLP(input_size=0, layer_sizes=[1])


# ----------------------------------------------------------------------
# Run pytest if executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v", __file__])