"""
Day 32: Model Persistence, Checkpointing, and Evaluation Metrics

This module provides two core capabilities:
1. **MetricsCalculator** – computes classification metrics from scratch,
   including confusion matrix, accuracy, precision, recall, F1, macro‑F1,
   and micro‑F1. It handles edge cases like zero denominators gracefully.
2. **ModelCheckpoint** – safely saves and loads training state (model weights,
   optimizer state, epoch, metrics, and optional scheduler state) using atomic
   file operations. All state is serialised to JSON (with NaN/Infinity rejected).

These building blocks are essential for production‑ready training pipelines.
"""

import json
import os
from numbers import Integral
from typing import Any


class MetricsCalculator:
    """
    Calculate multiclass classification metrics from ground truth and predictions.

    All inputs must be lists of integer class labels in [0, num_classes-1].
    Empty inputs raise ValueError – metrics are undefined for zero samples.

    Example:
        >>> calc = MetricsCalculator(num_classes=3)
        >>> y_true = [0, 1, 2, 0]
        >>> y_pred = [0, 1, 1, 0]
        >>> calc.accuracy(y_true, y_pred)  # 3/4 = 0.75
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes (>0). Must be a positive integer.
        Raises:
            TypeError: If num_classes is not a positive integer (bool is rejected).
        """
        # Reject booleans (they are subclasses of int but semantically wrong)
        if (
            not isinstance(num_classes, Integral)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise TypeError("num_classes must be a positive integer.")
        self.num_classes = int(num_classes)

    def _validate_labels(self, y_true: list[int], y_pred: list[int]) -> None:
        """
        Internal helper: check lengths, non‑emptiness, and label bounds.
        Raises ValueError if anything is wrong.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if not y_true:
            raise ValueError("Input lists must not be empty.")

        # Every label must be a valid integer in the allowed range
        for label in y_true + y_pred:
            if (
                isinstance(label, bool)  # bool is a subclass of int, but we reject it
                or not isinstance(label, Integral)
                or not (0 <= int(label) < self.num_classes)
            ):
                raise ValueError(
                    f"All labels must be integers in [0, {self.num_classes - 1}]. "
                    f"Found {label}."
                )

    def confusion_matrix(self, y_true: list[int], y_pred: list[int]) -> list[list[int]]:
        """
        Build the confusion matrix.

        Convention: C[i][j] = number of samples whose true class is i
        and predicted class is j.  Rows = true, columns = predicted.

        The matrix has shape (num_classes, num_classes).
        """
        self._validate_labels(y_true, y_pred)

        # Initialise a square matrix of zeros
        matrix = [[0] * self.num_classes for _ in range(self.num_classes)]

        # Count each (true, predicted) pair
        for t, p in zip(y_true, y_pred):
            matrix[int(t)][int(p)] += 1

        return matrix

    def accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
        """
        Compute accuracy = (correct predictions) / (total predictions).

        This is equivalent to sum(diag(C)) / sum(all C).
        """
        cm = self.confusion_matrix(y_true, y_pred)
        correct = sum(cm[i][i] for i in range(self.num_classes))
        total = sum(sum(row) for row in cm)
        # total > 0 because we forbid empty inputs
        return correct / total

    def precision_recall_f1_per_class(
        self, y_true: list[int], y_pred: list[int]
    ) -> dict[int, dict[str, float]]:
        """
        Compute precision, recall, and F1 for each class.

        For each class k:
          - TP_k = C[k][k]
          - FP_k = sum(C[i][k] for i != k)
          - FN_k = sum(C[k][j] for j != k)

        Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2*P*R/(P+R).

        If the denominator is zero (no predictions or no true positives),
        we return 0.0 for that metric. This is a common practical convention.
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
        """
        Compute macro‑averaged F1: the unweighted mean of per‑class F1 scores.

        Every class contributes equally, regardless of its support.
        """
        per_class = self.precision_recall_f1_per_class(y_true, y_pred)
        return sum(metrics["f1"] for metrics in per_class.values()) / self.num_classes

    def micro_f1(self, y_true: list[int], y_pred: list[int]) -> float:
        """
        Compute micro‑averaged F1.

        This aggregates total TP, FP, FN across all classes, then computes
        F1 = 2*TP / (2*TP + FP + FN).

        For ordinary single‑label multiclass classification, micro‑F1 equals
        accuracy – a useful sanity check.
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
        # denom > 0 because we have at least one sample
        return (2 * total_tp) / denom


class ModelCheckpoint:
    """
    Safely save and load training checkpoints using atomic file operations.

    The checkpoint is a JSON file containing:
      - epoch (int)
      - model_state (dict)
      - optimizer_state (dict)
      - metrics (dict)
      - scheduler_state (dict, optional)

    We use a temporary file + os.replace() to ensure that the target file
    is never left in a partially written state – readers always see a
    complete checkpoint or the previous one.

    All state must be JSON‑serializable; we reject NaN/Infinity to keep
    the data clean and portable.
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
        Atomically save a checkpoint to `filepath`.

        Steps:
          1. Validate input types.
          2. Build the checkpoint dictionary.
          3. Create parent directories if needed.
          4. Write the JSON to a temporary file (filepath.tmp).
          5. Atomically rename the temporary file to the target.

        If any step fails, the temporary file is removed, and the
        previous checkpoint (if any) remains untouched.

        Args:
            filepath: Where to save the checkpoint.
            model_state: Model parameters (e.g., weights and biases).
            optimizer_state: Optimizer internal state (must be complete).
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
            scheduler_state: (Optional) Learning rate scheduler state.
        """
        # --- Type checks (clear, early failures) ---
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

        # --- Build checkpoint structure ---
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": metrics,
        }
        if scheduler_state is not None:
            checkpoint["scheduler_state"] = scheduler_state

        # --- Ensure directory exists ---
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # --- Atomic write via temporary file ---
        tmp_path = f"{filepath}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    checkpoint,
                    f,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,  # Reject NaN/Infinity – they are not valid JSON numbers.
                )
            # Atomic rename (replaces destination only if rename succeeds)
            os.replace(tmp_path, filepath)
        except TypeError as e:
            # JSON serialisation error (non‑serialisable object or invalid float)
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
        Load a checkpoint from disk with thorough validation.

        Checks:
          - File exists and is readable.
          - JSON is well‑formed.
          - The root is a JSON object.
          - All required keys are present.
          - The values have the expected types (epoch int, state dicts, etc.).

        Returns:
            The checkpoint dictionary.

        Raises appropriate exceptions if any check fails.
        """
        # --- File existence ---
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # --- Read and parse JSON ---
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in checkpoint: {e}") from e

        # --- Ensure the root is a dictionary ---
        if not isinstance(data, dict):
            raise ValueError("Checkpoint root must be a JSON object.")

        # --- Required keys ---
        required_keys = {"epoch", "model_state", "optimizer_state", "metrics"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")

        # --- Type validation ---
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
