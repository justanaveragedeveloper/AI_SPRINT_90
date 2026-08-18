"""
Comprehensive test suite for Day 32 implementation.

Every expected value has been independently computed by hand or by
mathematical derivation, so we are confident that the tests catch
real regressions, not just repeat the implementation.

We test:
  - Confusion matrix construction and basic properties.
  - Accuracy, precision, recall, F1, macro‑F1, micro‑F1.
  - Many edge cases: empty, mismatched lengths, invalid labels,
    zero‑support classes, no predictions, etc.
  - Checkpoint save/load, atomicity, error handling, and type safety.
  - Mathematical invariants: row sums = support, micro‑F1 = accuracy.
"""

import json
import os
import tempfile
from collections import Counter
from unittest.mock import patch

import pytest
from checkpoint_metrics import MetricsCalculator, ModelCheckpoint

# ----------------------------------------------------------------------
# Fixtures – reusable test data
# ----------------------------------------------------------------------


@pytest.fixture
def calc_2():
    """Return a calculator for 2 classes."""
    return MetricsCalculator(num_classes=2)


@pytest.fixture
def calc_3():
    """Return a calculator for 3 classes."""
    return MetricsCalculator(num_classes=3)


@pytest.fixture
def sample_data_3():
    """
    A classic 8‑sample dataset with known confusion matrix.

    y_true = [0, 1, 2, 0, 1, 2, 0, 0]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1]

    Expected confusion matrix (rows = true, columns = pred):
        [[3, 1, 0],
         [0, 1, 1],
         [0, 1, 1]]
    """
    y_true = [0, 1, 2, 0, 1, 2, 0, 0]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
    return y_true, y_pred


@pytest.fixture
def temp_checkpoint_path():
    """Provide a temporary file path for checkpoint tests, and clean up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        path = tmp.name
    yield path
    # Clean up both the main file and any leftover .tmp file
    if os.path.exists(path):
        os.remove(path)
    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# ----------------------------------------------------------------------
# 1. Confusion Matrix Tests
# ----------------------------------------------------------------------


class TestConfusionMatrix:
    def test_basic(self, calc_3, sample_data_3):
        """Verify the known confusion matrix for sample_data_3."""
        y_true, y_pred = sample_data_3
        cm = calc_3.confusion_matrix(y_true, y_pred)
        expected = [
            [3, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
        assert cm == expected

    def test_empty_input_raises(self, calc_3):
        """Confusion matrix must reject empty inputs."""
        with pytest.raises(ValueError, match="must not be empty"):
            calc_3.confusion_matrix([], [])

    def test_mismatched_length_raises(self, calc_3):
        """Input lists must have same length."""
        with pytest.raises(ValueError, match="same length"):
            calc_3.confusion_matrix([0, 1], [0])

    def test_invalid_label_raises(self, calc_2):
        """Labels outside [0, num_classes-1] are rejected."""
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            calc_2.confusion_matrix([0, 2], [0, 1])
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            calc_2.confusion_matrix([0, 1], [0, -1])

    def test_non_integer_label_raises(self, calc_2):
        """Only integers are allowed; floats are rejected."""
        with pytest.raises(ValueError, match="integers"):
            calc_2.confusion_matrix([0, 1.5], [0, 1])

    def test_boolean_labels_rejected(self, calc_2):
        """Booleans are considered invalid class labels."""
        with pytest.raises(ValueError, match="integers"):
            calc_2.confusion_matrix([True], [1])

    def test_numpy_integers_supported(self, calc_3):
        """
        NumPy integer types are accepted (common in ML code).
        We require numpy for this test.
        """
        np = pytest.importorskip("numpy")
        y_true = [np.int64(0), np.int64(1), np.int64(2)]
        y_pred = [np.int64(0), np.int64(1), np.int64(2)]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_all_predictions_correct(self, calc_3):
        """When all predictions match truth, diagonal is full."""
        y_true = [0, 1, 2, 1, 0]
        y_pred = y_true.copy()
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]

    def test_all_predictions_wrong(self, calc_3):
        """When all predictions are wrong, diagonal is zero."""
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]

    def test_single_sample(self, calc_3):
        """Confusion matrix works with a single sample."""
        y_true = [1]
        y_pred = [1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]

    def test_only_one_class_true(self, calc_3):
        """Only one class appears in ground truth."""
        y_true = [0, 0, 0]
        y_pred = [0, 1, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [2, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

    def test_missing_predicted_class(self, calc_3):
        """Some class is never predicted."""
        y_true = [0, 1, 2]
        y_pred = [0, 0, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]


# ----------------------------------------------------------------------
# 2. Accuracy Tests
# ----------------------------------------------------------------------


class TestAccuracy:
    def test_basic(self, calc_3, sample_data_3):
        """Accuracy for sample_data_3 is 5/8 = 0.625."""
        y_true, y_pred = sample_data_3
        assert calc_3.accuracy(y_true, y_pred) == 5 / 8

    def test_empty_raises(self, calc_3):
        """Accuracy must reject empty inputs (metrics undefined)."""
        with pytest.raises(ValueError):
            calc_3.accuracy([], [])

    def test_perfect(self, calc_3):
        """Perfect predictions yield accuracy 1.0."""
        y_true = [0, 1, 2, 0]
        y_pred = y_true.copy()
        assert calc_3.accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self, calc_3):
        """All wrong predictions yield accuracy 0.0."""
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        assert calc_3.accuracy(y_true, y_pred) == 0.0


# ----------------------------------------------------------------------
# 3. Per‑Class Precision/Recall/F1
# ----------------------------------------------------------------------


class TestPerClassMetrics:
    def test_precision_recall_f1(self, calc_2):
        """Classic binary example: balanced performance."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        result = calc_2.precision_recall_f1_per_class(y_true, y_pred)
        # Both classes have TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        assert result[0]["precision"] == 0.5
        assert result[0]["recall"] == 0.5
        assert result[0]["f1"] == 0.5
        assert result[1] == result[0]

    def test_zero_denominator_precision(self, calc_2):
        """
        Class 0 has no predictions → precision = 0.
        Class 1 has TP=2, FP=2, FN=0 → precision=0.5, recall=1, F1=2/3.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        result = calc_2.precision_recall_f1_per_class(y_true, y_pred)
        assert result[0]["precision"] == 0.0
        assert result[0]["recall"] == 0.0
        assert result[0]["f1"] == 0.0
        assert result[1]["precision"] == 0.5
        assert result[1]["recall"] == 1.0
        assert result[1]["f1"] == pytest.approx(2 / 3)

    def test_zero_denominator_recall(self, calc_2):
        """
        Class 0 has TP=2, FP=0, FN=1 → precision=1, recall=2/3, F1=4/5.
        Class 1 has TP=0, FP=1, FN=0 → all metrics = 0.
        """
        y_true = [0, 0, 0]
        y_pred = [0, 0, 1]
        result = calc_2.precision_recall_f1_per_class(y_true, y_pred)
        assert result[0]["precision"] == 1.0
        assert result[0]["recall"] == pytest.approx(2 / 3)
        assert result[0]["f1"] == pytest.approx(4 / 5)
        assert result[1]["precision"] == 0.0
        assert result[1]["recall"] == 0.0
        assert result[1]["f1"] == 0.0


# ----------------------------------------------------------------------
# 4. Macro‑F1 Tests
# ----------------------------------------------------------------------


class TestMacroF1:
    def test_basic(self, calc_2):
        """Both classes F1=0.5 → macro = 0.5."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        assert calc_2.macro_f1(y_true, y_pred) == 0.5

    def test_perfect(self, calc_3):
        """All classes perfect → macro = 1."""
        y_true = [0, 1, 2, 0, 1]
        y_pred = y_true.copy()
        assert calc_3.macro_f1(y_true, y_pred) == 1.0

    def test_class_with_zero_support(self, calc_3):
        """
        Class 2 has zero true examples → F1=0. Classes 0 & 1 have F1=0.5.
        Macro = (0.5+0.5+0)/3 = 1/3.
        """
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        macro = calc_3.macro_f1(y_true, y_pred)
        assert macro == (0.5 + 0.5 + 0.0) / 3

    def test_class_with_no_predictions(self, calc_3):
        """
        Only class 0 has predictions (all samples predicted as 0).
        Class 0: TP=2, FP=3, FN=0 → P=0.4, R=1, F1=4/7.
        Classes 1 & 2: F1=0.
        Macro = (4/7)/3 = 4/21 ≈0.190476.
        """
        y_true = [0, 1, 2, 1, 0]
        y_pred = [0, 0, 0, 0, 0]
        macro = calc_3.macro_f1(y_true, y_pred)
        assert macro == pytest.approx(4 / 21)


# ----------------------------------------------------------------------
# 5. Micro‑F1 Tests
# ----------------------------------------------------------------------


class TestMicroF1:
    def test_basic(self, calc_2):
        """Binary example: micro = accuracy = 0.5."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        assert calc_2.micro_f1(y_true, y_pred) == 0.5

    def test_equals_accuracy(self, calc_3, sample_data_3):
        """For single‑label multiclass, micro‑F1 must equal accuracy."""
        y_true, y_pred = sample_data_3
        acc = calc_3.accuracy(y_true, y_pred)
        micro = calc_3.micro_f1(y_true, y_pred)
        assert micro == pytest.approx(acc)

    def test_perfect(self, calc_3):
        """All correct → micro = 1."""
        y_true = [0, 1, 2, 0, 1]
        y_pred = y_true.copy()
        assert calc_3.micro_f1(y_true, y_pred) == 1.0

    def test_all_wrong(self, calc_3):
        """All wrong → micro = 0."""
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        assert calc_3.micro_f1(y_true, y_pred) == 0.0


# ----------------------------------------------------------------------
# 6. Mathematical Invariants
# ----------------------------------------------------------------------


class TestMathInvariants:
    def test_confusion_matrix_row_sums_equal_support(self, calc_3):
        """Row sums must equal the true class counts."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 0]
        y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        support = Counter(y_true)
        for i in range(3):
            assert sum(cm[i]) == support[i]

    def test_total_count_equals_dataset_size(self, calc_3):
        """Sum of all matrix entries must equal number of samples."""
        y_true = [0, 1, 2, 0, 1, 2, 0, 0]
        y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        total = sum(sum(row) for row in cm)
        assert total == len(y_true)

    def test_accuracy_equals_diagonal_over_total(self, calc_3, sample_data_3):
        """Accuracy = diagonal sum / total sum."""
        y_true, y_pred = sample_data_3
        cm = calc_3.confusion_matrix(y_true, y_pred)
        diag = sum(cm[i][i] for i in range(3))
        total = sum(sum(row) for row in cm)
        acc_manual = diag / total
        assert calc_3.accuracy(y_true, y_pred) == acc_manual

    def test_micro_f1_equals_accuracy_random(self, calc_3):
        """
        Randomly generate datasets and verify micro‑F1 = accuracy.
        This is a strong statistical check.
        """
        import random

        random.seed(42)
        for _ in range(10):
            n = 20
            y_true = [random.randint(0, 2) for _ in range(n)]
            y_pred = [random.randint(0, 2) for _ in range(n)]
            acc = calc_3.accuracy(y_true, y_pred)
            micro = calc_3.micro_f1(y_true, y_pred)
            assert micro == pytest.approx(acc, abs=1e-12)


# ----------------------------------------------------------------------
# 7. Checkpoint Tests
# ----------------------------------------------------------------------


class TestCheckpointSaveLoad:
    def test_basic_save_load(self, temp_checkpoint_path):
        """Save a simple checkpoint and verify it loads correctly."""
        model_state = {"weights": [1.0, 2.0, 3.0], "bias": 0.1}
        opt_state = {"lr": 0.01, "momentum": 0.9, "step": 100}
        metrics = {"loss": 0.234, "acc": 0.95}
        epoch = 5

        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, model_state, opt_state, epoch, metrics
        )
        assert os.path.exists(temp_checkpoint_path)

        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["epoch"] == epoch
        assert loaded["model_state"] == model_state
        assert loaded["optimizer_state"] == opt_state
        assert loaded["metrics"] == metrics

    def test_nested_state(self, temp_checkpoint_path):
        """Checkpoints can contain nested dictionaries and lists."""
        model_state = {"layer1": {"w": [1, 2], "b": 0}, "layer2": {"w": [3, 4], "b": 1}}
        opt_state = {"param_groups": [{"lr": 0.01, "params": [0, 1]}]}
        metrics = {"train_loss": [0.5, 0.4]}
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, model_state, opt_state, 10, metrics
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["model_state"] == model_state
        assert loaded["optimizer_state"] == opt_state

    def test_epoch_and_metrics_persistence(self, temp_checkpoint_path):
        """Epoch and metrics are stored correctly."""
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {}, {}, 7, {"val_acc": 0.88}
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["epoch"] == 7
        assert loaded["metrics"]["val_acc"] == 0.88

    def test_replace_existing(self, temp_checkpoint_path):
        """Saving a second checkpoint overwrites the first atomically."""
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {"v1": 1}, {}, 1, {"a": 1.0}
        )
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {"v2": 2}, {}, 2, {"b": 2.0}
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["epoch"] == 2
        assert loaded["model_state"] == {"v2": 2}

    def test_optional_scheduler_state(self, temp_checkpoint_path):
        """Scheduler state is optionally saved and loaded."""
        scheduler_state = {"last_epoch": 10, "best": 0.95, "cooldown": 0}
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {}, {}, 5, {"acc": 0.9}, scheduler_state
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["scheduler_state"] == scheduler_state

    def test_missing_checkpoint_raises(self):
        """Loading a non‑existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ModelCheckpoint.load_checkpoint("/nonexistent/file.json")

    def test_malformed_json_raises(self, temp_checkpoint_path):
        """Invalid JSON raises ValueError."""
        with open(temp_checkpoint_path, "w") as f:
            f.write("this is not json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_empty_checkpoint_file_raises(self, temp_checkpoint_path):
        """An empty file is not a valid checkpoint."""
        with open(temp_checkpoint_path, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="Malformed JSON"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_missing_keys_raises(self, temp_checkpoint_path):
        """Missing required keys raise KeyError."""
        with open(temp_checkpoint_path, "w") as f:
            json.dump({"epoch": 1, "model_state": {}}, f)
        with pytest.raises(KeyError, match="missing required keys"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_non_json_serializable_state_raises(self, temp_checkpoint_path):
        """Attempting to save non‑JSON‑serializable data raises TypeError."""
        class Dummy:
            pass

        with pytest.raises(TypeError, match="non‑JSON‑serializable"):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path,
                model_state={"bad": Dummy()},
                optimizer_state={},
                epoch=0,
                metrics={},
            )

    def test_checkpoint_root_not_dict_raises(self, temp_checkpoint_path):
        """Root of checkpoint must be a JSON object (dict)."""
        with open(temp_checkpoint_path, "w") as f:
            json.dump([], f)
        with pytest.raises(ValueError, match="root must be a JSON object"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_wrong_state_type_raises(self, temp_checkpoint_path):
        """model_state must be a dict, not a list."""
        checkpoint = {
            "epoch": 1,
            "model_state": [],  # invalid: should be dict
            "optimizer_state": {},
            "metrics": {},
        }
        with open(temp_checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        with pytest.raises(TypeError, match="model_state must be dict"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_reject_nan_in_checkpoint(self, temp_checkpoint_path):
        """NaN values are rejected (allow_nan=False)."""
        with pytest.raises(ValueError, match="Out of range float"):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path,
                model_state={"weight": float("nan")},
                optimizer_state={},
                epoch=1,
                metrics={},
            )

    def test_realistic_adam_state_round_trip(self, temp_checkpoint_path):
        """
        Simulate a realistic Adam optimizer state with moments.
        Ensures that nested structures survive the round trip.
        """
        model_state = {"layer1": {"weights": [0.1, -0.2], "bias": [0.0]}}
        optimizer_state = {
            "lr": 0.001,
            "step": 42,
            "m": {"layer1.weights": [0.01, -0.03]},
            "v": {"layer1.weights": [0.001, 0.002]},
        }
        scheduler_state = {"last_epoch": 42}
        metrics = {"train_loss": 0.12, "val_loss": 0.15, "val_accuracy": 0.94}

        ModelCheckpoint.save_checkpoint(
            filepath=temp_checkpoint_path,
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=42,
            metrics=metrics,
            scheduler_state=scheduler_state,
        )

        restored = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert restored["epoch"] == 42
        assert restored["model_state"] == model_state
        assert restored["optimizer_state"] == optimizer_state
        assert restored["scheduler_state"] == scheduler_state
        assert restored["metrics"] == metrics

    def test_directory_creation(self, tmp_path):
        """Parent directories are created automatically."""
        deep_path = tmp_path / "sub" / "dir" / "checkpoint.json"
        ModelCheckpoint.save_checkpoint(str(deep_path), {}, {}, 0, {})
        assert os.path.exists(deep_path)
        assert os.path.isdir(deep_path.parent)

    def test_atomicity_on_serialization_failure(self, temp_checkpoint_path):
        """
        If serialization fails (e.g., non‑serializable object), the original
        checkpoint remains intact and the temporary file is cleaned up.
        """
        # Save a valid checkpoint first
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {"original": 42}, {}, 0, {}
        )
        original_content: str
        with open(temp_checkpoint_path) as f:
            original_content = f.read()

        # Attempt to save a bad object – should fail
        class Bad:
            pass

        with pytest.raises(TypeError):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path, {"bad": Bad()}, {}, 1, {}
            )

        # Original should still be there
        assert os.path.exists(temp_checkpoint_path)
        with open(temp_checkpoint_path) as f:
            assert f.read() == original_content
        # Temporary file must be removed
        tmp_path = temp_checkpoint_path + ".tmp"
        assert not os.path.exists(tmp_path)

    def test_atomicity_on_os_replace_failure(self, temp_checkpoint_path):
        """
        If os.replace() fails (simulated by a permission error), the original
        checkpoint remains untouched and the temporary file is cleaned.
        """
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {"original": 42}, {}, 0, {}
        )
        original_content: str
        with open(temp_checkpoint_path) as f:
            original_content = f.read()

        with (
            patch("os.replace", side_effect=OSError("permission denied")),
            pytest.raises(OSError),
        ):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path, {"new": 99}, {}, 1, {}
            )

        assert os.path.exists(temp_checkpoint_path)
        with open(temp_checkpoint_path) as f:
            assert f.read() == original_content
        tmp_path = temp_checkpoint_path + ".tmp"
        assert not os.path.exists(tmp_path)