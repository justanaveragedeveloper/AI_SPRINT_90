"""
Comprehensive pytest suite for Day 32 implementation.
All expected values have been independently verified.
"""

import json
import os
import tempfile
from collections import Counter
from unittest.mock import patch

import pytest
from checkpoint_metrics import MetricsCalculator, ModelCheckpoint

# ----------------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------------


@pytest.fixture
def calc_2():
    return MetricsCalculator(num_classes=2)


@pytest.fixture
def calc_3():
    return MetricsCalculator(num_classes=3)


@pytest.fixture
def sample_data_3():
    y_true = [0, 1, 2, 0, 1, 2, 0, 0]
    y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
    return y_true, y_pred


@pytest.fixture
def temp_checkpoint_path():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)
    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# ----------------------------------------------------------------------
# METRICS TESTS
# ----------------------------------------------------------------------


class TestConfusionMatrix:
    def test_basic(self, calc_3, sample_data_3):
        y_true, y_pred = sample_data_3
        cm = calc_3.confusion_matrix(y_true, y_pred)
        expected = [
            [3, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
        assert cm == expected

    def test_empty_input_raises(self, calc_3):
        with pytest.raises(ValueError, match="must not be empty"):
            calc_3.confusion_matrix([], [])

    def test_mismatched_length_raises(self, calc_3):
        with pytest.raises(ValueError, match="same length"):
            calc_3.confusion_matrix([0, 1], [0])

    def test_invalid_label_raises(self, calc_2):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            calc_2.confusion_matrix([0, 2], [0, 1])
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            calc_2.confusion_matrix([0, 1], [0, -1])

    def test_non_integer_label_raises(self, calc_2):
        with pytest.raises(ValueError, match="integers"):
            calc_2.confusion_matrix([0, 1.5], [0, 1])

    def test_boolean_labels_rejected(self, calc_2):
        with pytest.raises(ValueError, match="integers"):
            calc_2.confusion_matrix([True], [1])

    def test_numpy_integers_supported(self, calc_3):
        np = pytest.importorskip("numpy")
        y_true = [np.int64(0), np.int64(1), np.int64(2)]
        y_pred = [np.int64(0), np.int64(1), np.int64(2)]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_all_predictions_correct(self, calc_3):
        y_true = [0, 1, 2, 1, 0]
        y_pred = y_true.copy()
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 1],
        ]

    def test_all_predictions_wrong(self, calc_3):
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]

    def test_single_sample(self, calc_3):
        y_true = [1]
        y_pred = [1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]

    def test_only_one_class_true(self, calc_3):
        y_true = [0, 0, 0]
        y_pred = [0, 1, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [2, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

    def test_missing_predicted_class(self, calc_3):
        y_true = [0, 1, 2]
        y_pred = [0, 0, 0]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        assert cm == [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]


class TestAccuracy:
    def test_basic(self, calc_3, sample_data_3):
        y_true, y_pred = sample_data_3
        assert calc_3.accuracy(y_true, y_pred) == 5 / 8

    def test_empty_raises(self, calc_3):
        with pytest.raises(ValueError):
            calc_3.accuracy([], [])

    def test_perfect(self, calc_3):
        y_true = [0, 1, 2, 0]
        y_pred = y_true.copy()
        assert calc_3.accuracy(y_true, y_pred) == 1.0

    def test_all_wrong(self, calc_3):
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        assert calc_3.accuracy(y_true, y_pred) == 0.0


class TestPerClassMetrics:
    def test_precision_recall_f1(self, calc_2):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        result = calc_2.precision_recall_f1_per_class(y_true, y_pred)
        assert result[0]["precision"] == 0.5
        assert result[0]["recall"] == 0.5
        assert result[0]["f1"] == 0.5
        assert result[1] == result[0]

    def test_zero_denominator_precision(self, calc_2):
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
        y_true = [0, 0, 0]
        y_pred = [0, 0, 1]
        result = calc_2.precision_recall_f1_per_class(y_true, y_pred)
        assert result[0]["precision"] == 1.0
        assert result[0]["recall"] == pytest.approx(2 / 3)
        assert result[0]["f1"] == pytest.approx(4 / 5)
        assert result[1]["precision"] == 0.0
        assert result[1]["recall"] == 0.0
        assert result[1]["f1"] == 0.0


class TestMacroF1:
    def test_basic(self, calc_2):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        assert calc_2.macro_f1(y_true, y_pred) == 0.5

    def test_perfect(self, calc_3):
        y_true = [0, 1, 2, 0, 1]
        y_pred = y_true.copy()
        assert calc_3.macro_f1(y_true, y_pred) == 1.0

    def test_class_with_zero_support(self, calc_3):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]
        macro = calc_3.macro_f1(y_true, y_pred)
        assert macro == (0.5 + 0.5 + 0.0) / 3

    def test_class_with_no_predictions(self, calc_3):
        y_true = [0, 1, 2, 1, 0]
        y_pred = [0, 0, 0, 0, 0]
        macro = calc_3.macro_f1(y_true, y_pred)
        assert macro == pytest.approx(4 / 21)


class TestMicroF1:
    def test_basic(self, calc_2):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        assert calc_2.micro_f1(y_true, y_pred) == 0.5

    def test_equals_accuracy(self, calc_3, sample_data_3):
        y_true, y_pred = sample_data_3
        acc = calc_3.accuracy(y_true, y_pred)
        micro = calc_3.micro_f1(y_true, y_pred)
        assert micro == pytest.approx(acc)

    def test_perfect(self, calc_3):
        y_true = [0, 1, 2, 0, 1]
        y_pred = y_true.copy()
        assert calc_3.micro_f1(y_true, y_pred) == 1.0

    def test_all_wrong(self, calc_3):
        y_true = [0, 1, 2]
        y_pred = [1, 2, 0]
        assert calc_3.micro_f1(y_true, y_pred) == 0.0


class TestMathInvariants:
    def test_confusion_matrix_row_sums_equal_support(self, calc_3):
        y_true = [0, 1, 2, 0, 1, 2, 0, 0]
        y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        support = Counter(y_true)
        for i in range(3):
            assert sum(cm[i]) == support[i]

    def test_total_count_equals_dataset_size(self, calc_3):
        y_true = [0, 1, 2, 0, 1, 2, 0, 0]
        y_pred = [0, 1, 1, 0, 2, 2, 0, 1]
        cm = calc_3.confusion_matrix(y_true, y_pred)
        total = sum(sum(row) for row in cm)
        assert total == len(y_true)

    def test_accuracy_equals_diagonal_over_total(self, calc_3, sample_data_3):
        y_true, y_pred = sample_data_3
        cm = calc_3.confusion_matrix(y_true, y_pred)
        diag = sum(cm[i][i] for i in range(3))
        total = sum(sum(row) for row in cm)
        acc_manual = diag / total
        assert calc_3.accuracy(y_true, y_pred) == acc_manual

    def test_micro_f1_equals_accuracy_random(self, calc_3):
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
# CHECKPOINT TESTS
# ----------------------------------------------------------------------


class TestCheckpointSaveLoad:
    def test_basic_save_load(self, temp_checkpoint_path):
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
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {}, {}, 7, {"val_acc": 0.88}
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["epoch"] == 7
        assert loaded["metrics"]["val_acc"] == 0.88

    def test_replace_existing(self, temp_checkpoint_path):
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
        scheduler_state = {"last_epoch": 10, "best": 0.95, "cooldown": 0}
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {}, {}, 5, {"acc": 0.9}, scheduler_state
        )
        loaded = ModelCheckpoint.load_checkpoint(temp_checkpoint_path)
        assert loaded["scheduler_state"] == scheduler_state

    def test_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelCheckpoint.load_checkpoint("/nonexistent/file.json")

    def test_malformed_json_raises(self, temp_checkpoint_path):
        with open(temp_checkpoint_path, "w") as f:
            f.write("this is not json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_empty_checkpoint_file_raises(self, temp_checkpoint_path):
        with open(temp_checkpoint_path, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="Malformed JSON"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_missing_keys_raises(self, temp_checkpoint_path):
        with open(temp_checkpoint_path, "w") as f:
            json.dump({"epoch": 1, "model_state": {}}, f)
        with pytest.raises(KeyError, match="missing required keys"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_non_json_serializable_state_raises(self, temp_checkpoint_path):
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
        with open(temp_checkpoint_path, "w") as f:
            json.dump([], f)
        with pytest.raises(ValueError, match="root must be a JSON object"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_wrong_state_type_raises(self, temp_checkpoint_path):
        checkpoint = {
            "epoch": 1,
            "model_state": [],
            "optimizer_state": {},
            "metrics": {},
        }
        with open(temp_checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        with pytest.raises(TypeError, match="model_state must be dict"):
            ModelCheckpoint.load_checkpoint(temp_checkpoint_path)

    def test_reject_nan_in_checkpoint(self, temp_checkpoint_path):
        with pytest.raises(ValueError, match="Out of range float"):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path,
                model_state={"weight": float("nan")},
                optimizer_state={},
                epoch=1,
                metrics={},
            )

    def test_realistic_adam_state_round_trip(self, temp_checkpoint_path):
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
        deep_path = tmp_path / "sub" / "dir" / "checkpoint.json"
        ModelCheckpoint.save_checkpoint(str(deep_path), {}, {}, 0, {})
        assert os.path.exists(deep_path)
        assert os.path.isdir(deep_path.parent)

    def test_atomicity_on_serialization_failure(self, temp_checkpoint_path):
        ModelCheckpoint.save_checkpoint(
            temp_checkpoint_path, {"original": 42}, {}, 0, {}
        )
        original_content: str
        with open(temp_checkpoint_path) as f:
            original_content = f.read()

        class Bad:
            pass

        with pytest.raises(TypeError):
            ModelCheckpoint.save_checkpoint(
                temp_checkpoint_path, {"bad": Bad()}, {}, 1, {}
            )

        assert os.path.exists(temp_checkpoint_path)
        with open(temp_checkpoint_path) as f:
            assert f.read() == original_content
        tmp_path = temp_checkpoint_path + ".tmp"
        assert not os.path.exists(tmp_path)

    def test_atomicity_on_os_replace_failure(self, temp_checkpoint_path):
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