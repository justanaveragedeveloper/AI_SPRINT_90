# -- test_classifier.py --
# Comprehensive pytest suite for MultiClassClassifier

import os
import random
import sys

import pytest

# Add parent directory to path so we can import classifier and engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classifier import MultiClassClassifier, _get_grad, _get_value
from engine import Value

# -------------------------------------------------------------------
# Basic Functionality Tests
# -------------------------------------------------------------------


def test_forward_pass_dimensions():
    """Forward pass should return a list of Value objects with length = num_classes,
    and the probabilities should sum to 1."""
    clf = MultiClassClassifier(input_dim=4, hidden_dims=[8, 8], num_classes=3)
    sample = [0.5, -1.2, 3.1, 0.0]
    probs = clf.forward(sample)

    assert len(probs) == 3
    # Sum of probabilities should be approximately 1
    assert abs(sum(_get_value(p) for p in probs) - 1.0) < 1e-5


def test_prediction():
    """Prediction should return a valid class index (0..num_classes-1)."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    pred = clf.predict([1.0, 2.0])
    assert pred in [0, 1, 2]


def test_probability_normalization():
    """Even with extreme logits, softmax should produce probabilities in [0,1]
    that sum to 1."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=5)
    # Force the first output neuron to have huge weights → extreme logit
    clf.model.layers[-1].neurons[0].w = [Value(100.0), Value(100.0)]

    probs = clf.forward([0.0, 0.0])
    assert abs(sum(_get_value(p) for p in probs) - 1.0) < 1e-5
    assert all(0.0 <= _get_value(p) <= 1.0 for p in probs)


# -------------------------------------------------------------------
# Training & Convergence
# -------------------------------------------------------------------


def test_mini_batch_training_convergence():
    """
    Train on a simple 3‑class dataset and verify that the classifier
    learns to classify all points correctly.
    """
    random.seed(42)
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[8], num_classes=3)

    # Synthetic dataset: three clusters in 2D space
    dataset_x = [
        [5.0, 5.0],
        [5.2, 4.8],  # class 0
        [-5.0, -5.0],
        [-4.9, -5.1],  # class 1
        [5.0, -5.0],
        [4.8, -5.2],  # class 2
    ]
    dataset_y = [0, 0, 1, 1, 2, 2]

    # Compute initial loss (with lr=0 so we get the loss without updates)
    initial_loss = clf.train_step(dataset_x, dataset_y, learning_rate=0.0)

    # Train for 60 epochs
    for _ in range(60):
        current_loss = clf.train_step(dataset_x, dataset_y, learning_rate=0.05)

    # After training, predictions should match the labels
    predictions = [clf.predict(x) for x in dataset_x]
    assert current_loss < initial_loss  # loss decreased
    assert predictions == dataset_y, f"Expected {dataset_y}, got {predictions}"


def test_batch_size_1():
    """The classifier should handle a batch of size 1."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    x = [[1.0, 2.0]]
    y = [1]
    loss = clf.train_step(x, y, learning_rate=0.1)
    assert isinstance(loss, float)


# -------------------------------------------------------------------
# Gradient & Parameter Update Tests
# -------------------------------------------------------------------


def test_gradient_propagation():
    """After a backward pass, all parameters should have non‑zero gradients."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[0.5, -0.5]]
    sample_y = [0]

    # Train step with lr=0 → no update, but gradients are computed
    clf.train_step(sample_x, sample_y, learning_rate=0.0)

    # Verify every parameter has a gradient (should be non‑zero)
    for param in clf.model.parameters():
        assert _get_grad(param) != 0.0, "Gradient not propagated"


def test_parameter_update():
    """Parameters should change after a training step with learning_rate > 0."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[1.0, 2.0]]
    sample_y = [1]

    before = [_get_value(p) for p in clf.model.parameters()]
    clf.train_step(sample_x, sample_y, learning_rate=0.1)
    after = [_get_value(p) for p in clf.model.parameters()]

    # At least one parameter must have changed
    assert any(a != b for a, b in zip(before, after)), "Parameters did not update"


def test_learning_rate_zero_no_change():
    """With learning_rate = 0, parameters should remain unchanged."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[1.0, 2.0]]
    sample_y = [1]

    before = [_get_value(p) for p in clf.model.parameters()]
    clf.train_step(sample_x, sample_y, learning_rate=0.0)
    after = [_get_value(p) for p in clf.model.parameters()]

    assert all(a == b for a, b in zip(before, after)), "Parameters changed with lr=0"


def test_prediction_does_not_modify_graph():
    """
    The predict() method should not set any gradients on parameters.
    After calling predict(), all gradients should remain zero (if zeroed first).
    """
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample = [1.0, 2.0]

    # Ensure all gradients start at zero
    clf.model.zero_grad()
    # Call predict – no backward is performed
    _ = clf.predict(sample)

    # Gradients should still be zero
    for param in clf.model.parameters():
        assert _get_grad(param) == 0.0, "Prediction should not modify gradients"


# -------------------------------------------------------------------
# Input Validation (Defensive Programming)
# -------------------------------------------------------------------


def test_invalid_labels():
    """Labels outside [0, num_classes-1] should raise a ValueError."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Invalid label"):
        clf.train_step([[1.0, 2.0]], [3], learning_rate=0.1)


def test_mismatched_batch_sizes():
    """Batch_x and batch_y must have the same length."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Mismatched batch sizes"):
        clf.train_step([[1.0, 2.0], [3.0, 4.0]], [0], learning_rate=0.1)


def test_empty_batches():
    """Empty batches should raise a ValueError."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Batch cannot be empty"):
        clf.train_step([], [], learning_rate=0.1)


def test_invalid_dimensions():
    """Input samples must have the correct number of features."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Input dimension mismatch"):
        clf.train_step([[1.0, 2.0, 3.0]], [0], learning_rate=0.1)


def test_forward_invalid_dimension():
    """forward() should also validate input dimension."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Expected input dimension"):
        clf.forward([1.0, 2.0, 3.0])


# -------------------------------------------------------------------
# Edge Cases
# -------------------------------------------------------------------


def test_single_sample_prediction():
    """Prediction works even with a different input dimension."""
    clf = MultiClassClassifier(input_dim=3, hidden_dims=[5], num_classes=2)
    pred = clf.predict([0.0, 1.0, -1.0])
    assert pred in [0, 1]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
