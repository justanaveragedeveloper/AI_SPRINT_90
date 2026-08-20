"""
Unit tests for the Day 33 Training Pipeline.

These tests cover:
    - Dataset construction, validation, and batching
    - Training pipeline initialization and configuration
    - Full training loop execution
    - History tracking and content
    - Validation metrics
    - Learning rate scheduler integration
    - Checkpoint creation and round‑trip
    - Edge cases (single sample, batch size > dataset, etc.)
    - Integration invariants (parameter updates, gradient sync, loss weighting)

Run with: pytest test_pipeline.py -v
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ------------------------------------------------------------------------------
# Path Setup – same as pipeline.py
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
for day_dir in [
    "day-24-autograd",
    "day-25-nn-from-scratch",
    "day-26-softmax-cross-entropy",
    "day-29-optimizers",
    "day-30-regularization",
    "day-31-lr-schedulers",
    "day-32-checkpoint-and-metrics",
]:
    path = PROJECT_ROOT / day_dir
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from checkpoint_metrics import MetricsCalculator, ModelCheckpoint
from engine import Value
from nn import MLP
from optimizers import SGD
from pipeline import Dataset, TrainingPipeline
from schedulers import StepLR

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data():
    """A small synthetic dataset (8 samples, 3 features, 3 classes)."""
    X = [
        [0.5, -0.3, 0.8],
        [0.1, 0.9, -0.2],
        [-0.4, 0.3, 0.6],
        [0.7, -0.1, -0.5],
        [0.2, 0.5, 0.1],
        [-0.8, 0.4, 0.3],
        [0.3, -0.7, 0.9],
        [0.6, 0.2, -0.4],
    ]
    y = [0, 1, 2, 0, 1, 2, 0, 1]
    return X, y


@pytest.fixture
def sample_dataset(sample_data):
    X, y = sample_data
    return Dataset(X, y)


@pytest.fixture
def simple_model():
    """A small MLP with 3 input features, 4 hidden, 3 output classes."""
    return MLP(input_size=3, layer_sizes=[4, 3])


@pytest.fixture
def metrics_calculator():
    return MetricsCalculator(num_classes=3)


@pytest.fixture
def optimizer(simple_model):
    return SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints (cleaned up after test)."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ==============================================================================
# Dataset Tests
# ==============================================================================


class TestDataset:
    """Validate the Dataset class's construction and batching behavior."""

    def test_construction_valid(self, sample_data):
        """A valid dataset should store X, y, and metadata correctly."""
        X, y = sample_data
        dataset = Dataset(X, y)
        assert len(dataset) == len(X)
        assert dataset.num_samples == len(X)
        assert dataset.input_dim == len(X[0])

    def test_construction_empty_raises(self):
        """Empty dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            Dataset([], [])

    def test_construction_mismatched_lengths_raises(self):
        """X and y must have the same length."""
        with pytest.raises(ValueError, match="X and y must have the same length"):
            Dataset([[0.5]], [0, 1])

    def test_construction_inconsistent_dimensions_raises(self):
        """All samples must have the same feature dimension."""
        with pytest.raises(ValueError, match="Inconsistent feature dimensions"):
            Dataset([[0.5, 0.3], [0.1]], [0, 1])

    def test_construction_invalid_label_raises(self):
        """Labels must be non‑negative integers."""
        with pytest.raises(ValueError, match="must be non-negative"):
            Dataset([[0.5]], [-1])

    def test_len(self, sample_dataset):
        """__len__ returns the number of samples."""
        assert len(sample_dataset) == 8

    def test_getitem(self, sample_dataset):
        """__getitem__ returns a (features, label) tuple."""
        X, y = sample_dataset[0]
        assert isinstance(X, list)
        assert isinstance(y, int)

    def test_getitem_out_of_bounds_raises(self, sample_dataset):
        """Index out of range should raise IndexError."""
        with pytest.raises(IndexError):
            _ = sample_dataset[100]

    def test_batches_boundaries(self, sample_dataset):
        """Batches should respect batch_size; the last batch may be smaller."""
        batch_size = 3
        batches = list(sample_dataset.batches(batch_size, shuffle=False))
        assert len(batches) == 3  # 8 samples → 3+3+2
        assert len(batches[0][0]) == 3
        assert len(batches[1][0]) == 3
        assert len(batches[2][0]) == 2

    def test_batches_preserves_correspondence(self, sample_dataset):
        """Each batch must keep X and y aligned (no cross‑sample mixing)."""
        for batch_X, batch_y in sample_dataset.batches(batch_size=4, shuffle=False):
            for x, y in zip(batch_X, batch_y):
                found = False
                for orig_x, orig_y in zip(sample_dataset.X, sample_dataset.y):
                    if x == orig_x and y == orig_y:
                        found = True
                        break
                assert found

    def test_batches_shuffling_deterministic(self, sample_dataset):
        """With a fixed seed, shuffling produces the same order."""
        batches1 = list(sample_dataset.batches(batch_size=3, shuffle=True, seed=42))
        batches2 = list(sample_dataset.batches(batch_size=3, shuffle=True, seed=42))
        assert batches1[0][0] == batches2[0][0]
        assert batches1[0][1] == batches2[0][1]

    def test_batches_shuffling_changes_order(self, sample_dataset):
        """Shuffling should reorder samples compared to no shuffle."""
        unshuffled = list(sample_dataset.batches(batch_size=3, shuffle=False))
        shuffled = list(sample_dataset.batches(batch_size=3, shuffle=True, seed=123))
        assert unshuffled[0][0] != shuffled[0][0]

    def test_batches_invalid_batch_size_raises(self, sample_dataset):
        """batch_size must be >0 and ≤ dataset size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(sample_dataset.batches(0))
        with pytest.raises(ValueError, match="batch_size .* cannot exceed"):
            list(sample_dataset.batches(100))


# ==============================================================================
# Training Pipeline Tests
# ==============================================================================


class TestTrainingPipeline:
    """Test the TrainingPipeline orchestrator."""

    def test_construction(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Pipeline should store all components and initialise state correctly."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        assert pipeline.model is simple_model
        assert pipeline.optimizer is optimizer
        assert pipeline.dataset is sample_dataset
        assert pipeline.history == []
        assert pipeline.best_val_loss == float("inf")
        assert pipeline.best_val_macro_f1 == -1.0
        assert pipeline.checkpoint_dir is None

    def test_construction_with_scheduler(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Pipeline accepts a scheduler."""
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            scheduler=scheduler,
        )
        assert pipeline.scheduler is scheduler

    def test_construction_with_checkpoint_dir(
        self,
        sample_dataset,
        simple_model,
        optimizer,
        metrics_calculator,
        temp_checkpoint_dir,
    ):
        """Pipeline accepts a checkpoint directory and creates it."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            checkpoint_dir=temp_checkpoint_dir,
        )
        assert pipeline.checkpoint_dir is not None
        assert pipeline.checkpoint_dir.exists()

    def test_construction_with_clip_value(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Pipeline accepts a positive clip_value."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            clip_value=1.0,
        )
        assert pipeline.clip_value == 1.0

    def test_construction_invalid_clip_value_raises(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """clip_value must be positive."""
        with pytest.raises(ValueError, match="clip_value must be positive"):
            TrainingPipeline(
                model=simple_model,
                optimizer=optimizer,
                dataset=sample_dataset,
                metrics_calculator=metrics_calculator,
                clip_value=0.0,
            )

    def test_train_loop_execution(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Basic training run should produce history entries."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        history = pipeline.train(epochs=3, batch_size=4, validation_split=0.25)
        assert len(history) == 3
        assert len(pipeline.history) == 3

    def test_history_length(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """History length equals number of epochs."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        epochs = 5
        history = pipeline.train(epochs=epochs, batch_size=4, validation_split=0.0)
        assert len(history) == epochs

    def test_history_contains_expected_fields(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Each history record should have the expected keys."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        history = pipeline.train(epochs=2, batch_size=4, validation_split=0.25)

        expected_fields = {
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "learning_rate",
        }
        for record in history:
            assert set(record.keys()) == expected_fields

    def test_training_loss_decreases(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Training loss should generally decrease over epochs (average check)."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        history = pipeline.train(epochs=10, batch_size=4, validation_split=0.0)

        mid = len(history) // 2
        first_half_avg = sum(r["train_loss"] for r in history[:mid]) / mid
        second_half_avg = sum(r["train_loss"] for r in history[mid:]) / (
            len(history) - mid
        )
        assert (
            second_half_avg < first_half_avg
        ), f"Training loss did not decrease: {first_half_avg:.4f} → {second_half_avg:.4f}"

    def test_validation_metrics_recorded(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """With validation_split>0, validation metrics should be recorded (not NaN)."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        history = pipeline.train(epochs=3, batch_size=4, validation_split=0.25)

        for record in history:
            assert not math.isnan(record["val_loss"])
            assert not math.isnan(record["val_accuracy"])
            assert not math.isnan(record["val_macro_f1"])

    def test_validation_metrics_nan_when_no_split(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """With validation_split=0, validation metrics should be NaN."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        history = pipeline.train(epochs=3, batch_size=4, validation_split=0.0)

        for record in history:
            assert math.isnan(record["val_loss"])
            assert math.isnan(record["val_accuracy"])
            assert math.isnan(record["val_macro_f1"])

    def test_learning_rate_scheduler_exact_sequence(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """StepLR should produce the exact expected learning rate sequence."""
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            scheduler=scheduler,
        )

        initial_lr = optimizer.lr
        history = pipeline.train(epochs=5, batch_size=4, validation_split=0.0)

        lrs = [r["learning_rate"] for r in history]
        expected = [
            initial_lr,
            initial_lr,
            initial_lr * 0.5,
            initial_lr * 0.5,
            initial_lr * 0.25,
        ]
        assert lrs == pytest.approx(expected, rel=1e-6)

    def test_best_checkpoint_creation(
        self,
        sample_dataset,
        simple_model,
        optimizer,
        metrics_calculator,
        temp_checkpoint_dir,
    ):
        """Checkpoints should be created (at least two files: epoch and best)."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            checkpoint_dir=temp_checkpoint_dir,
        )

        pipeline.train(epochs=5, batch_size=4, validation_split=0.25)

        checkpoint_files = list(Path(temp_checkpoint_dir).glob("*.json"))
        assert len(checkpoint_files) >= 2

    def test_checkpoint_round_trip(
        self,
        sample_dataset,
        simple_model,
        optimizer,
        metrics_calculator,
        temp_checkpoint_dir,
    ):
        """
        A saved checkpoint should contain the expected keys and match the best epoch.
        (scheduler_state is optional; we don't require it when no scheduler.)
        """
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            checkpoint_dir=temp_checkpoint_dir,
        )

        _ = pipeline.train(epochs=3, batch_size=4, validation_split=0.25)

        best_path = Path(temp_checkpoint_dir) / "best_checkpoint.json"
        assert best_path.exists()

        loaded = ModelCheckpoint.load_checkpoint(str(best_path))

        assert "epoch" in loaded
        assert "model_state" in loaded
        assert "optimizer_state" in loaded
        assert "metrics" in loaded
        # scheduler_state is optional; we don't require it when no scheduler

    def test_invalid_epochs_raises(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """epochs must be >0."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        with pytest.raises(ValueError, match="epochs must be positive"):
            pipeline.train(epochs=0, batch_size=4)
        with pytest.raises(ValueError, match="epochs must be positive"):
            pipeline.train(epochs=-1, batch_size=4)

    def test_single_sample_dataset(self):
        """Pipeline should handle a dataset with only one sample."""
        X = [[0.5, 0.3, 0.8]]
        y = [0]
        dataset = Dataset(X, y)
        model = MLP(input_size=3, layer_sizes=[4, 3])
        optimizer = SGD(model.parameters(), lr=0.01)
        metrics = MetricsCalculator(num_classes=3)

        pipeline = TrainingPipeline(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            metrics_calculator=metrics,
        )

        history = pipeline.train(epochs=3, batch_size=4, validation_split=0.0)
        assert len(history) == 3

    def test_single_batch_handling(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Pipeline should work when batch_size == dataset size."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )

        history = pipeline.train(epochs=3, batch_size=8, validation_split=0.0)
        assert len(history) == 3
        assert history[0]["train_loss"] > 0

    # ==========================================================================
    # Integration Invariants
    # ==========================================================================

    def test_parameters_change_after_training(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """
        Critical invariant: after training, at least one model parameter must change.
        This ensures gradients and optimizer steps actually happened.
        """
        before = [p.value for p in simple_model.parameters()]

        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )
        pipeline.train(epochs=3, batch_size=4, validation_split=0.0)

        after = [p.value for p in simple_model.parameters()]

        assert any(
            b != a for b, a in zip(before, after)
        ), "Parameters did not change after training"

    def test_gradient_sync(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """
        Verify that after backward, 'gradient' is populated and sync copies it to 'grad'.
        This ensures the optimizer can see the gradients.
        """
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )

        batch_X, batch_y = next(sample_dataset.batches(batch_size=4, shuffle=False))

        pipeline._zero_gradients()
        for p in simple_model.parameters():
            assert p.grad == 0.0
            assert p.gradient == 0.0

        logits = pipeline._forward_batch(batch_X)
        loss = pipeline._compute_batch_loss(logits, batch_y)
        loss.backward()

        # After backward, 'gradient' should be non‑zero (for most parameters)
        for p in simple_model.parameters():
            assert p.gradient != 0.0

        # Sync and verify
        pipeline._sync_gradients()
        for p in simple_model.parameters():
            assert p.grad == p.gradient

    def test_epoch_loss_weighted_by_batch_size(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """
        The epoch loss should be a sample‑weighted average, not a batch‑average.
        We mock _compute_batch_loss to return a value equal to the batch size,
        so the weighted average becomes (sum batch_size²) / total_samples.
        """
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )

        original_compute = pipeline._compute_batch_loss

        def mock_compute(logits, y):
            return Value(float(len(y)))

        pipeline._compute_batch_loss = mock_compute  # type: ignore

        # With batch_size=3, batches are 3,3,2 samples.
        train_loss = pipeline._train_epoch(batch_size=3, shuffle=False)

        expected = (3 * 3 + 3 * 3 + 2 * 2) / 8.0  # = 2.75
        assert train_loss == pytest.approx(expected, rel=1e-6)

        # Restore original method
        pipeline._compute_batch_loss = original_compute  # type: ignore

    def test_clipping_actually_clips(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """
        If gradient norm exceeds clip_value, clipping should reduce it to ≤ clip_value.
        """
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
            clip_value=0.5,
        )

        batch_X, batch_y = next(sample_dataset.batches(batch_size=4, shuffle=False))
        pipeline._zero_gradients()
        logits = pipeline._forward_batch(batch_X)
        loss = pipeline._compute_batch_loss(logits, batch_y)
        loss.backward()
        pipeline._sync_gradients()

        grads = [p.grad for p in simple_model.parameters()]
        original_norm = math.sqrt(sum(g * g for g in grads))

        pipeline._apply_gradient_clipping()

        grads_after = [p.grad for p in simple_model.parameters()]
        new_norm = math.sqrt(sum(g * g for g in grads_after))

        if original_norm > 0.5:
            assert new_norm <= 0.5
        else:
            # If already under threshold, it should remain unchanged
            assert new_norm == original_norm

    def test_train_validation_disjoint(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """Training and validation sets should be disjoint (no sample overlap)."""
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )

        train_dataset, val_dataset = pipeline._split_dataset(
            validation_split=0.2, seed=42
        )

        total = len(sample_dataset)
        assert len(train_dataset) == int(total * 0.8)
        assert len(val_dataset) == total - len(train_dataset)  # and they are disjoint

    def test_empty_validation_raises_when_requested(
        self, sample_dataset, simple_model, optimizer, metrics_calculator
    ):
        """
        If validation_split > 0 but the dataset is too small to split,
        we should raise a clear ValueError (not silently return empty).
        """
        pipeline = TrainingPipeline(
            model=simple_model,
            optimizer=optimizer,
            dataset=sample_dataset,
            metrics_calculator=metrics_calculator,
        )

        small_dataset = Dataset([[0.5]], [0])
        pipeline.dataset = small_dataset
        with pytest.raises(
            ValueError, match="would produce empty training or validation set"
        ):
            pipeline._split_dataset(validation_split=0.5, seed=42)
