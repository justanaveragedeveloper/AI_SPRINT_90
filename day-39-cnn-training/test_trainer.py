"""
Test suite for Day 39 trainer and optimizer.

We test:
- Invalid hyperparameters → ValueError.
- Basic SGD update (with and without momentum).
- Persistence of velocity state.
- Updates for bias (1D), Dense weight (2D), Conv weight (4D).
- Shape mismatch rejection.
- Empty batch handling.
- Single batch behavior.
- Gradient averaging via `scale_gradients`.
- Actual gradient accumulation (two backward calls add up).
- Optimizer uses averaged gradient (not raw sum).
- State persistence across steps.
- Non‑finite (NaN/Inf) protection for gradients, velocities, parameters.
- Batch length validation.
- End‑to‑end training with real ConvNet (if available).

All tests pass with the current implementation.
"""

import math
import random

import pytest
from trainer import SGDMomentum, scale_gradients, train_epoch

# ------------------------------------------------------------------
# Mocks for Day 36–38 modules.
# These mimic the real ConvNet, DataLoader, and Loss without requiring
# the actual modules to be installed. They are sufficient to test the
# optimizer and training loop mechanics.
# ------------------------------------------------------------------


class DummyConv2D:
    """A simple Conv2D layer with weight, bias, and their gradients (nested lists)."""
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = [
            [
                [[0.0 for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias = [0.0 for _ in range(out_channels)]
        self.weight_grad = [
            [
                [[0.0 for _ in range(kernel_size)] for _ in range(kernel_size)]
                for _ in range(in_channels)
            ]
            for _ in range(out_channels)
        ]
        self.bias_grad = [0.0 for _ in range(out_channels)]


class DummyDense:
    """A simple Dense layer with weight, bias, and gradients."""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = [[0.0 for _ in range(in_features)] for _ in range(out_features)]
        self.bias = [0.0 for _ in range(out_features)]
        self.weight_grad = [[0.0 for _ in range(in_features)] for _ in range(out_features)]
        self.bias_grad = [0.0 for _ in range(out_features)]


class DummyModel:
    """
    A minimal model with `conv1` and `fc1` layers.
    It does not actually implement forward/backward logic; it's used only
    to test the optimizer's ability to discover and update parameters.
    """
    def __init__(self):
        self.conv1 = DummyConv2D(1, 2, 3)
        self.fc1 = DummyDense(8, 2)

    def zero_grad(self):
        pass   # Not needed for these tests.

    def forward(self, img):
        return [0.0, 1.0], None   # Dummy logits and closures.

    def backward(self, dlogits, closures):
        pass   # Not needed for optimizer tests.


class TrainableDummyLayer:
    """
    A minimal trainable layer that actually accumulates gradients.
    Used to test gradient accumulation and average behaviour.
    """
    def __init__(self, initial_weight=0.0):
        self.weight = [initial_weight]
        self.weight_grad = [0.0]

    def zero_grad(self):
        self.weight_grad[0] = 0.0

    def backward(self, grad):
        self.weight_grad[0] += grad


class TrainableDummyModel:
    """
    A model with a single trainable weight that implements backward.
    We expose it as a property `fc1` so the optimizer discovers it.
    """
    def __init__(self):
        self.layer = TrainableDummyLayer(initial_weight=0.0)

    def zero_grad(self):
        self.layer.zero_grad()

    def backward(self, grad):
        self.layer.backward(grad)

    @property
    def fc1(self):
        return self.layer


class DummyDataset:
    """Simple dataset that stores images and labels."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DummyDataLoader:
    """A simple DataLoader that yields batches from a dataset."""
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_imgs = [self.dataset[j][0] for j in batch_indices]
            batch_labels = [self.dataset[j][1] for j in batch_indices]
            yield batch_imgs, batch_labels


class DummyCrossEntropyLoss:
    """
    A dummy cross‑entropy loss that returns a loss value and a closure
    that returns p - one_hot(label) (the gradient of CCE w.r.t. logits).
    """
    def forward(self, logits, label):
        def loss_backward():
            # Compute softmax probabilities
            p = [math.exp(l) for l in logits]
            s = sum(p)
            p = [x / s for x in p]
            # Subtract one from the target index
            p[label] -= 1.0
            return p

        # Loss = -log(softmax(logits)[label]) with a small epsilon for stability.
        softmax_val = math.exp(logits[label]) / sum(math.exp(l) for l in logits)
        loss = -math.log(max(1e-12, softmax_val))
        return loss, loss_backward


# ------------------------------------------------------------------
# Helper function: synthetic spatial dataset
# ------------------------------------------------------------------

def generate_synthetic_spatial_dataset(num_samples=20):
    """
    Generate a toy dataset of 1×4×4 images with two classes:
      - Class 0: horizontal stripe on top row.
      - Class 1: vertical stripe on left column.
    This is used in the end‑to‑end integration test.
    """
    images, labels = [], []
    for i in range(num_samples):
        if i % 2 == 0:
            # Horizontal stripe
            img = [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
            labels.append(0)
        else:
            # Vertical stripe
            img = [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ]
            labels.append(1)
        images.append(img)
    return images, labels


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_invalid_learning_rate():
    """Ensure that a non‑positive learning rate raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        SGDMomentum(lr=-0.1)


def test_invalid_momentum():
    """Ensure momentum outside [0,1) raises ValueError."""
    with pytest.raises(ValueError, match="range"):
        SGDMomentum(momentum=1.5)
    with pytest.raises(ValueError, match="range"):
        SGDMomentum(momentum=-0.1)


def test_sgd_basic_without_momentum():
    """Vanilla SGD update: θ_new = θ - lr * grad."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.5, momentum=0.0)
    model.fc1.weight[0][0] = 2.0
    model.fc1.weight_grad[0][0] = 1.0
    opt.step(model)
    assert model.fc1.weight[0][0] == 1.5   # 2.0 - 0.5*1.0


def test_momentum_velocity_persistence():
    """
    Test that momentum accumulates correctly over two steps.
    v1 = μ*v0 + lr*g = 0.9*0 + 0.5*1 = 0.5  → weight = 1.5
    v2 = 0.9*0.5 + 0.5*1 = 0.95 → weight = 1.5 - 0.95 = 0.55
    """
    model = DummyModel()
    opt = SGDMomentum(lr=0.5, momentum=0.9)
    model.fc1.weight[0][0] = 2.0
    model.fc1.weight_grad[0][0] = 1.0
    opt.step(model)
    assert model.fc1.weight[0][0] == 1.5
    opt.step(model)
    assert model.fc1.weight[0][0] == 0.55


def test_1d_bias_update():
    """Bias is 1‑dimensional; test update on a single element."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.3, momentum=0.0)
    model.fc1.bias[1] = 5.0
    model.fc1.bias_grad[1] = 2.0
    opt.step(model)
    assert model.fc1.bias[1] == 5.0 - 0.3 * 2.0


def test_2d_dense_weight_update():
    """Dense weight is 2‑dimensional; update one element."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.1, momentum=0.0)
    model.fc1.weight[1][2] = 3.0
    model.fc1.weight_grad[1][2] = 4.0
    opt.step(model)
    assert model.fc1.weight[1][2] == 3.0 - 0.1 * 4.0


def test_4d_conv_weight_update():
    """Conv weight is 4‑dimensional; update one element."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.2, momentum=0.0)
    model.conv1.weight[0][0][1][1] = 7.0
    model.conv1.weight_grad[0][0][1][1] = 2.5
    opt.step(model)
    assert model.conv1.weight[0][0][1][1] == 7.0 - 0.2 * 2.5


def test_shape_mismatch_rejection():
    """The optimizer should reject a velocity with incorrect shape."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.1, momentum=0.0)
    opt.velocities["fc1.weight"] = [[0.0]]   # Wrong shape (should be 8x2)
    with pytest.raises(ValueError, match="shape mismatch"):
        opt.step(model)


def test_gradient_shape_validation():
    """The optimizer should reject a gradient with incorrect shape."""
    model = DummyModel()
    opt = SGDMomentum(lr=0.1)
    model.fc1.weight_grad = [[0.0]]   # Wrong shape
    with pytest.raises(ValueError, match="Gradient shape mismatch"):
        opt.step(model)


def test_empty_batch_handling():
    """An empty batch should be skipped; loss and accuracy become 0."""
    model = DummyModel()
    criterion = DummyCrossEntropyLoss()
    opt = SGDMomentum(lr=0.1)
    dataset = DummyDataset([], [])
    loader = DummyDataLoader(dataset, batch_size=4)
    avg_loss, acc = train_epoch(model, loader, criterion, opt)
    assert avg_loss == 0.0
    assert acc == 0.0


def test_single_batch_behavior():
    """
    A batch with 4 samples should run without errors and produce
    accuracy within [0,1].
    """
    images, labels = generate_synthetic_spatial_dataset(4)
    dataset = DummyDataset(images, labels)
    loader = DummyDataLoader(dataset, batch_size=4)
    model = DummyModel()
    criterion = DummyCrossEntropyLoss()
    opt = SGDMomentum(lr=0.01)
    _, acc = train_epoch(model, loader, criterion, opt)
    assert 0.0 <= acc <= 1.0


def test_correct_batch_gradient_averaging():
    """Verify that `scale_gradients` correctly divides gradients by batch size."""
    model = DummyModel()
    model.conv1.weight_grad[0][0][0][0] = 4.0
    model.conv1.bias_grad[0] = 2.0
    scale_gradients(model, 1.0 / 2)
    assert model.conv1.weight_grad[0][0][0][0] == 2.0
    assert model.conv1.bias_grad[0] == 1.0


def test_accuracy_bounded():
    """Accuracy should never exceed 1.0 or go below 0.0."""
    images, labels = generate_synthetic_spatial_dataset(10)
    dataset = DummyDataset(images, labels)
    loader = DummyDataLoader(dataset, batch_size=2)
    model = DummyModel()
    criterion = DummyCrossEntropyLoss()
    opt = SGDMomentum(lr=0.01)
    _, acc = train_epoch(model, loader, criterion, opt)
    assert 0.0 <= acc <= 1.0


def test_actual_gradient_accumulation():
    """
    Test that two backward calls with gradients 1.0 and 2.0 produce
    an accumulated gradient of 3.0.
    """
    model = TrainableDummyModel()
    model.zero_grad()
    model.backward(1.0)
    model.backward(2.0)
    assert model.layer.weight_grad[0] == 3.0


def test_optimizer_uses_batch_average():
    """
    After accumulating 4.0 and 6.0 (sum=10.0) and scaling by 1/2,
    the averaged gradient is 5.0. The optimizer should update using 5.0.
    """
    model = TrainableDummyModel()
    model.zero_grad()
    model.backward(4.0)
    model.backward(6.0)
    scale_gradients(model, 1.0 / 2)   # Average: (4+6)/2 = 5.0
    assert model.layer.weight_grad[0] == 5.0

    opt = SGDMomentum(lr=1.0, momentum=0.0)
    opt.step(model)   # weight = 0 - 1.0*5.0 = -5.0
    assert model.layer.weight[0] == -5.0


def test_optimizer_state_persistence():
    """Velocity should persist across steps and increase with repeated gradients."""
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    model = DummyModel()
    model.fc1.weight_grad[0][0] = 1.0
    opt.step(model)
    assert "fc1.weight" in opt.velocities
    v_before = opt.velocities["fc1.weight"][0][0]
    opt.step(model)
    v_after = opt.velocities["fc1.weight"][0][0]
    assert v_after > v_before   # Momentum should accumulate


def test_non_finite_gradient_protection():
    """A NaN gradient should be caught before update."""
    opt = SGDMomentum(lr=0.1, momentum=0.0)
    model = DummyModel()
    model.fc1.weight_grad[0][0] = float("nan")
    with pytest.raises(ValueError, match="contains non‑finite values"):
        opt.step(model)


def test_non_finite_velocity_protection():
    """A NaN velocity (with correct shape) should be caught."""
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    model = DummyModel()
    # Build a velocity tensor of the correct shape (8x2) with a NaN at (0,0)
    n_rows = len(model.fc1.weight)
    n_cols = len(model.fc1.weight[0])
    velocity = [
        [float("nan") if i == 0 and j == 0 else 0.0 for j in range(n_cols)]
        for i in range(n_rows)
    ]
    opt.velocities["fc1.weight"] = velocity
    model.fc1.weight_grad[0][0] = 1.0
    with pytest.raises(ValueError, match="contains non‑finite values"):
        opt.step(model)


def test_non_finite_parameter_protection():
    """A NaN parameter should be caught before update."""
    opt = SGDMomentum(lr=0.1, momentum=0.9)
    model = DummyModel()
    model.fc1.weight[0][0] = float("nan")
    model.fc1.weight_grad[0][0] = 1.0
    with pytest.raises(ValueError, match="contains non‑finite values"):
        opt.step(model)


def test_batch_length_mismatch_validation():
    """train_epoch should raise ValueError if batch images and labels have different lengths."""
    model = DummyModel()
    criterion = DummyCrossEntropyLoss()
    opt = SGDMomentum(lr=0.1)

    class BadDataLoader:
        def __iter__(self):
            yield [1, 2, 3], [0, 1]   # 3 images, 2 labels

    loader = BadDataLoader()
    with pytest.raises(ValueError, match="mismatched lengths"):
        train_epoch(model, loader, criterion, opt)


# ------------------------------------------------------------------
# Integration test with real modules (if available)
# ------------------------------------------------------------------

# Try to import the actual Day 36–38 modules.
# If they are present, we run a full training test on the synthetic dataset.
try:
    from cnn_arch import ConvNet
    from dataloader import DataLoader, SimpleImageDataset
    from losses import CrossEntropyLoss
    REAL_MODULES_AVAILABLE = True
except ImportError:
    REAL_MODULES_AVAILABLE = False


def evaluate(model, dataloader, criterion):
    """
    Helper to evaluate the model on a dataloader without updating weights.
    Returns (avg_loss, accuracy).
    """
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for batch_imgs, batch_labels in dataloader:
        batch_size = len(batch_imgs)
        if batch_size == 0:
            continue
        if len(batch_labels) != batch_size:
            raise ValueError("Mismatch")
        for img, label in zip(batch_imgs, batch_labels):
            logits, _ = model.forward(img)
            loss, _ = criterion.forward(logits, label)
            total_loss += loss
            pred_class = max(range(len(logits)), key=lambda i: logits[i])
            if pred_class == label:
                correct += 1
            total_samples += 1
    return (total_loss / total_samples if total_samples > 0 else 0.0,
            correct / total_samples if total_samples > 0 else 0.0)


@pytest.mark.skipif(not REAL_MODULES_AVAILABLE,
                    reason="Requires actual Day36-38 modules to be present")
def test_integration_with_real_modules():
    """
    End‑to‑end training test using the real ConvNet and synthetic dataset.
    Asserts:
      - Loss decreases significantly.
      - Accuracy improves.
      - All trainable parameters change.
    This is the strongest verification that the entire training pipeline works.
    """
    # Set a deterministic seed for reproducibility.
    random.seed(42)

    # Generate a larger dataset to ensure learning.
    images, labels = generate_synthetic_spatial_dataset(num_samples=200)
    dataset = SimpleImageDataset(images, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConvNet()
    criterion = CrossEntropyLoss()
    opt = SGDMomentum(lr=0.05, momentum=0.9)

    # Save a copy of initial parameters (nested lists are mutable, so we copy).
    initial_params = {
        "conv1.weight": [row[:] for row in model.conv1.weight],
        "conv1.bias": model.conv1.bias[:],
        "fc1.weight": [row[:] for row in model.fc1.weight],
        "fc1.bias": model.fc1.bias[:],
    }

    # Evaluate before training.
    initial_loss, initial_acc = evaluate(model, loader, criterion)

    # Train for 10 epochs.
    for _ in range(10):
        loss, acc = train_epoch(model, loader, criterion, opt)  # noqa: RUF059
        # We ignore loss/acc here; they are printed only for debugging if needed.

    # Evaluate after training.
    final_loss, final_acc = evaluate(model, loader, criterion)

    # Assert learning happened.
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    assert final_acc > initial_acc, f"Accuracy did not improve: {initial_acc:.4f} → {final_acc:.4f}"

    # Assert that every trainable parameter changed.
    assert model.conv1.weight != initial_params["conv1.weight"], "Conv weight unchanged"
    assert model.conv1.bias != initial_params["conv1.bias"], "Conv bias unchanged"
    assert model.fc1.weight != initial_params["fc1.weight"], "FC weight unchanged"
    assert model.fc1.bias != initial_params["fc1.bias"], "FC bias unchanged"