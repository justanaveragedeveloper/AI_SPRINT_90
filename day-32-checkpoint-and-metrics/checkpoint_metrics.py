"""
Day 32: Model Persistence, Checkpointing, and Evaluation Metrics
First‑principles implementation of metric calculations and safe model serialization.
"""

import json
import os
from numbers import Integral
from typing import Any


class MetricsCalculator:
    """
    Calculates multiclass classification metrics from ground truth and predictions.

    All inputs must be finite sequences of integer class labels in [0, num_classes-1].
    Empty inputs raise ValueError – metrics are undefined for zero samples.
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes (>0).
        Raises:
            TypeError: If num_classes is not a positive integer (bool is rejected).
        """
        if (
            not isinstance(num_classes, Integral)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise TypeError("num_classes must be a positive integer.")
        self.num_classes = int(num_classes)

    def _validate_labels(self, y_true: list[int], y_pred: list[int]) -> None:
        """Check lengths, non‑empty, and label bounds."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if not y_true:
            raise ValueError("Input lists must not be empty.")

        for label in y_true + y_pred:
            if (
                isinstance(label, bool)
                or not isinstance(label, Integral)
                or not (0 <= int(label) < self.num_classes)
            ):
                raise ValueError(
                    f"All labels must be integers in [0, {self.num_classes - 1}]. "
                    f"Found {label}."
                )

    def confusion_matrix(self, y_true: list[int], y_pred: list[int]) -> list[list[int]]:
        """Build confusion matrix C where C[i][j] = count of (true=i, pred=j)."""
        self._validate_labels(y_true, y_pred)
        matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        for t, p in zip(y_true, y_pred):
            matrix[int(t)][int(p)] += 1
        return matrix

    def accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
        """Accuracy = sum(diag(C)) / sum(all C)."""
        cm = self.confusion_matrix(y_true, y_pred)
        correct = sum(cm[i][i] for i in range(self.num_classes))
        total = sum(sum(row) for row in cm)
        # total is guaranteed > 0 because confusion_matrix raises for empty
        return correct / total

    def precision_recall_f1_per_class(
        self, y_true: list[int], y_pred: list[int]
    ) -> dict[int, dict[str, float]]:
        """
        Per‑class precision, recall, F1.

        For zero denominator:
          - precision = 0.0  (no predicted positives)
          - recall    = 0.0  (no actual positives)
          - F1        = 0.0  (undefined, set to 0)
        """
        cm = self.confusion_matrix(y_true, y_pred)
        result = {}

        for k in range(self.num_classes):
            tp = cm[k][k]
            fp = sum(cm[i][k] for i in range(self.num_classes) if i != k)
            fn = sum(cm[k][j] for j in range(self.num_classes) if j != k)

            denom_prec = tp + fp
            denom_rec = tp + fn

            precision = tp / denom_prec if denom_prec > 0 else 0.0
            recall = tp / denom_rec if denom_rec > 0 else 0.0

            denom_f1 = precision + recall
            f1 = 2 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0

            result[k] = {"precision": precision, "recall": recall, "f1": f1}

        return result

    def macro_f1(self, y_true: list[int], y_pred: list[int]) -> float:
        """Unweighted average of per‑class F1 scores."""
        per_class = self.precision_recall_f1_per_class(y_true, y_pred)
        return sum(metrics["f1"] for metrics in per_class.values()) / self.num_classes

    def micro_f1(self, y_true: list[int], y_pred: list[int]) -> float:
        """
        Global F1 computed from total TP, FP, FN.

        For ordinary single‑label multiclass, micro‑F1 == accuracy.
        """
        cm = self.confusion_matrix(y_true, y_pred)
        total_tp = sum(cm[k][k] for k in range(self.num_classes))
        total_fp = sum(
            sum(cm[i][k] for i in range(self.num_classes) if i != k)
            for k in range(self.num_classes)
        )
        total_fn = sum(
            sum(cm[k][j] for j in range(self.num_classes) if j != k)
            for k in range(self.num_classes)
        )

        denom = 2 * total_tp + total_fp + total_fn
        # denom > 0 because total samples > 0
        return (2 * total_tp) / denom


class ModelCheckpoint:
    """
    Atomically save / load training state (epoch, model, optimizer, metrics).

    Serialization uses JSON with allow_nan=False, so all state components must be
    JSON‑serializable and contain no NaN/Infinity.

    You may optionally include scheduler_state for learning rate schedulers.
    Exact resume requires that optimizer_state and scheduler_state contain all
    internal state (e.g., Adam moments, ReduceLROnPlateau counters).
    """

    @staticmethod
    def save_checkpoint(
        filepath: str,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        epoch: int,
        metrics: dict[str, Any],
        scheduler_state: dict[str, Any] | None = None,
    ) -> None:
        """
        Atomically save checkpoint.

        Args:
            filepath: Target file path.
            model_state: JSON‑serializable model parameters.
            optimizer_state: JSON‑serializable optimizer state (must include full internal state).
            epoch: Current epoch number.
            metrics: Dictionary of metric names to values (must be JSON‑serializable).
            scheduler_state: (Optional) JSON‑serializable scheduler state.

        Raises:
            TypeError: If any argument has incorrect type or contains non‑JSON‑serializable data.
            ValueError: If serialisation produces NaN/Infinity (allow_nan=False).
            OSError: On filesystem errors.
        """
        # Type checks – raise TypeError for type mismatches
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be int")
        if not isinstance(model_state, dict):
            raise TypeError("model_state must be dict")
        if not isinstance(optimizer_state, dict):
            raise TypeError("optimizer_state must be dict")
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be dict")
        if scheduler_state is not None and not isinstance(scheduler_state, dict):
            raise TypeError("scheduler_state must be dict")

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": metrics,
        }
        if scheduler_state is not None:
            checkpoint["scheduler_state"] = scheduler_state

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        tmp_path = f"{filepath}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    checkpoint,
                    f,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,  # reject NaN/Infinity
                )
            os.replace(tmp_path, filepath)
        except TypeError as e:
            # JSON serialisation error (non‑serialisable object)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise TypeError(
                "Checkpoint contains non‑JSON‑serializable objects or invalid numeric "
                "values (NaN/Infinity). Convert tensors/arrays to lists and remove NaNs."
            ) from e
        except Exception:
            # Clean up temporary file on any other failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def load_checkpoint(filepath: str) -> dict[str, Any]:
        """
        Load checkpoint with rigorous validation.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is malformed or root is not a dict.
            KeyError: If required keys are missing.
            TypeError: If epoch, model_state, optimizer_state, metrics, or scheduler_state
                       have the wrong type.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in checkpoint: {e}") from e

        # Root must be a dict
        if not isinstance(data, dict):
            raise ValueError("Checkpoint root must be a JSON object.") # noqa: TRY004

        # Required keys
        required_keys = {"epoch", "model_state", "optimizer_state", "metrics"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")

        # Type checks
        if not isinstance(data["epoch"], int) or isinstance(data["epoch"], bool):
            raise TypeError("epoch must be int")
        if not isinstance(data["model_state"], dict):
            raise TypeError("model_state must be dict")
        if not isinstance(data["optimizer_state"], dict):
            raise TypeError("optimizer_state must be dict")
        if not isinstance(data["metrics"], dict):
            raise TypeError("metrics must be dict")

        if "scheduler_state" in data and not isinstance(data["scheduler_state"], dict):
            raise TypeError("scheduler_state must be dict")

        return data
