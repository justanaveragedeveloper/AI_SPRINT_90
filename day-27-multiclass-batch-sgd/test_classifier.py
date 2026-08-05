# -- test_classifier.py --

import os
import random
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classifier import MultiClassClassifier, _get_grad, _get_value
from engine import Value

# ---------- Basic Functionality ----------


def test_forward_pass_dimensions():
    clf = MultiClassClassifier(input_dim=4, hidden_dims=[8, 8], num_classes=3)
    sample = [0.5, -1.2, 3.1, 0.0]
    probs = clf.forward(sample)
    assert len(probs) == 3
    assert abs(sum(_get_value(p) for p in probs) - 1.0) < 1e-5


def test_prediction():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    pred = clf.predict([1.0, 2.0])
    assert pred in [0, 1, 2]


def test_probability_normalization():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=5)
    # Force extreme logits
    clf.model.layers[-1].neurons[0].w = [Value(100.0), Value(100.0)]
    probs = clf.forward([0.0, 0.0])
    assert abs(sum(_get_value(p) for p in probs) - 1.0) < 1e-5
    assert all(0.0 <= _get_value(p) <= 1.0 for p in probs)


# ---------- Training & Convergence ----------


def test_mini_batch_training_convergence():
    random.seed(42)
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[8], num_classes=3)

    dataset_x = [
        [5.0, 5.0],
        [5.2, 4.8],  # class 0
        [-5.0, -5.0],
        [-4.9, -5.1],  # class 1
        [5.0, -5.0],
        [4.8, -5.2],  # class 2
    ]
    dataset_y = [0, 0, 1, 1, 2, 2]

    initial_loss = clf.train_step(dataset_x, dataset_y, learning_rate=0.0)
    for _ in range(60):
        current_loss = clf.train_step(dataset_x, dataset_y, learning_rate=0.05)

    predictions = [clf.predict(x) for x in dataset_x]
    assert current_loss < initial_loss
    assert predictions == dataset_y, f"Expected {dataset_y}, got {predictions}"


def test_batch_size_1():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    x = [[1.0, 2.0]]
    y = [1]
    loss = clf.train_step(x, y, learning_rate=0.1)
    assert isinstance(loss, float)


# ---------- Gradient & Parameter Update ----------


def test_gradient_propagation():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[0.5, -0.5]]
    sample_y = [0]

    clf.train_step(sample_x, sample_y, learning_rate=0.0)
    for p in clf.model.parameters():
        assert _get_grad(p) != 0.0, "Gradient not propagated"


def test_parameter_update():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[1.0, 2.0]]
    sample_y = [1]

    before = [_get_value(p) for p in clf.model.parameters()]
    clf.train_step(sample_x, sample_y, learning_rate=0.1)
    after = [_get_value(p) for p in clf.model.parameters()]
    assert any(a != b for a, b in zip(before, after)), "Parameters did not update"


def test_learning_rate_zero_no_change():
    """Ensure parameters remain unchanged when learning_rate=0."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample_x = [[1.0, 2.0]]
    sample_y = [1]

    before = [_get_value(p) for p in clf.model.parameters()]
    clf.train_step(sample_x, sample_y, learning_rate=0.0)
    after = [_get_value(p) for p in clf.model.parameters()]
    assert all(a == b for a, b in zip(before, after)), "Parameters changed with lr=0"


def test_prediction_does_not_modify_graph():
    """Verify that predict() does not set any gradients (graph remains clean)."""
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    sample = [1.0, 2.0]
    clf.model.zero_grad()
    _ = clf.predict(sample)
    for p in clf.model.parameters():
        assert _get_grad(p) == 0.0, "Prediction should not modify gradients"


# ---------- Input Validation ----------


def test_invalid_labels():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Invalid label"):
        clf.train_step([[1.0, 2.0]], [3], learning_rate=0.1)


def test_mismatched_batch_sizes():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Mismatched batch sizes"):
        clf.train_step([[1.0, 2.0], [3.0, 4.0]], [0], learning_rate=0.1)


def test_empty_batches():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Batch cannot be empty"):
        clf.train_step([], [], learning_rate=0.1)


def test_invalid_dimensions():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Input dimension mismatch"):
        clf.train_step([[1.0, 2.0, 3.0]], [0], learning_rate=0.1)


def test_forward_invalid_dimension():
    clf = MultiClassClassifier(input_dim=2, hidden_dims=[4], num_classes=3)
    with pytest.raises(ValueError, match="Expected input dimension"):
        clf.forward([1.0, 2.0, 3.0])


# ---------- Edge Cases ----------


def test_single_sample_prediction():
    clf = MultiClassClassifier(input_dim=3, hidden_dims=[5], num_classes=2)
    pred = clf.predict([0.0, 1.0, -1.0])
    assert pred in [0, 1]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
