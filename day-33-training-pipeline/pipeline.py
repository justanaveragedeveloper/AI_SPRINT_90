"""
Day 33: End-to-End Training Pipeline Orchestrator

This module integrates all components from Days 24–32 into a single,
testable training pipeline. It acts as an orchestrator, not an implementer.

The pipeline manages:
    - Dataset batching (with deterministic shuffling)
    - Training loop (forward, loss, backward, gradient clipping, optimizer step)
    - Learning rate scheduling
    - Validation (with eval mode, metrics, no gradient updates)
    - Checkpointing (atomic save on best validation Macro-F1)
    - Training history (loss, metrics, learning rate per epoch)

It does NOT reimplement autograd, neural layers, optimizers, schedulers, or metrics.
All heavy lifting is delegated to the existing Day 24–32 modules.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------------------
# Path Setup – Make previous day modules importable
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
# Imports from previous days
# ------------------------------------------------------------------------------
from checkpoint_metrics import MetricsCalculator, ModelCheckpoint
from engine import Value
from loss_and_ops import categorical_cross_entropy, softmax
from nn import MLP
from optimizers import SGD, Adam, Optimizer, RMSprop  # <-- fixed import
from schedulers import BaseLRScheduler

logger = logging.getLogger(__name__)


class Dataset:
    """
    Simple in‑memory dataset abstraction.

    Responsibilities:
        - Store features (X) and labels (y)
        - Validate shape, dimensions, and label ranges
        - Generate mini‑batches (with optional deterministic shuffling)
        - Preserve X/y correspondence in every batch

    Attributes:
        X (list[list[float]]): Feature matrix, shape (N, D)
        y (list[int]): Class labels, shape (N,)
        num_samples (int): Total number of samples
        input_dim (int): Number of features per sample
    """

    def __init__(self, X: list[list[float]], y: list[int]) -> None:
        """
        Create a Dataset.

        Args:
            X: Feature matrix – list of lists, each inner list is one sample.
            y: Class labels – list of integers, same length as X.

        Raises:
            TypeError: If X or y are not lists, or if elements have wrong types.
            ValueError: If lengths mismatch, dataset empty, feature dimensions
                       inconsistent, or labels are negative.
        """
        # Basic type checks
        if not isinstance(X, list):
            raise TypeError(f"X must be a list, got {type(X).__name__}")
        if not isinstance(y, list):
            raise TypeError(f"y must be a list, got {type(y).__name__}")

        # Length consistency
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. "
                f"Got len(X)={len(X)}, len(y)={len(y)}"
            )

        # Non‑empty dataset
        if len(X) == 0:
            raise ValueError("Dataset cannot be empty")

        # Ensure each sample has at least one feature
        if len(X[0]) == 0:
            raise ValueError("Each sample must have at least one feature")

        # Verify all samples have the same number of features
        first_dim = len(X[0])
        for i, sample in enumerate(X):
            if not isinstance(sample, list):
                raise TypeError(f"X[{i}] must be a list, got {type(sample).__name__}")
            if len(sample) != first_dim:
                raise ValueError(
                    f"Inconsistent feature dimensions: "
                    f"X[0] has {first_dim} features, X[{i}] has {len(sample)}"
                )

        # Labels must be non‑negative integers
        for i, label in enumerate(y):
            if not isinstance(label, int):
                raise TypeError(f"y[{i}] must be an int, got {type(label).__name__}")
            if label < 0:
                raise ValueError(f"y[{i}] must be non-negative, got {label}")

        # Store data
        self.X = X
        self.y = y
        self.num_samples = len(X)
        self.input_dim = first_dim if X else 0

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[list[float], int]:
        """Get a single sample by index."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        return self.X[idx], self.y[idx]

    def batches(
        self, batch_size: int, shuffle: bool = True, seed: int | None = None
    ) -> Generator[tuple[list[list[float]], list[int]], None, None]:
        """
        Generate mini‑batches.

        Args:
            batch_size: Number of samples per batch (must be >0 and <= total samples).
            shuffle: If True, shuffle the dataset before batching.
            seed: Random seed for deterministic shuffling (used only if shuffle=True).

        Yields:
            Tuple (batch_X, batch_y) where:
                - batch_X is a list of feature vectors (shape: batch_size × D)
                - batch_y is a list of labels (length: batch_size)

        Raises:
            ValueError: If batch_size is invalid (<=0 or > num_samples).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > self.num_samples:
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed "
                f"dataset size ({self.num_samples})"
            )

        # Build index list and optionally shuffle
        indices = list(range(self.num_samples))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        # Yield batches in slices
        for start in range(0, self.num_samples, batch_size):
            end = min(start + batch_size, self.num_samples)
            batch_indices = indices[start:end]

            batch_X = [self.X[i] for i in batch_indices]
            batch_y = [self.y[i] for i in batch_indices]
            yield batch_X, batch_y


class TrainingPipeline:
    """
    End‑to‑end training orchestrator.

    It coordinates all components and maintains training state. It does NOT
    implement autograd, layer math, optimizer formulas, scheduler logic, or
    metric formulas – those are delegated to existing Day 24–32 modules.

    The pipeline is production‑aware:
        - Defensive validation of all inputs
        - Deterministic shuffling and splitting (reproducible)
        - Numerical stability checks (NaN/Inf)
        - Atomic checkpointing (via Day 32)
        - Sample‑weighted epoch loss averaging
        - Proper train/eval mode switching

    Attributes:
        model (MLP): Neural network (Day 25)
        optimizer (Optimizer): Parameter optimizer (Day 29)
        dataset (Dataset): Training data
        metrics_calculator (MetricsCalculator): Metrics for validation (Day 32)
        scheduler (BaseLRScheduler | None): Learning rate scheduler (Day 31)
        checkpoint_dir (Path | None): Directory to save checkpoints (if any)
        clip_value (float | None): Gradient clipping threshold (if any)
        history (list[dict]): Per‑epoch training records
        best_val_loss (float): Best validation loss seen so far
        best_val_macro_f1 (float): Best validation Macro‑F1 seen so far
    """

    def __init__(
        self,
        model: MLP,
        optimizer: Optimizer,
        dataset: Dataset,
        metrics_calculator: MetricsCalculator,
        scheduler: BaseLRScheduler | None = None,
        checkpoint_dir: str | None = None,
        clip_value: float | None = None,
    ) -> None:
        """
        Initialize the pipeline with all necessary components.

        Args:
            model: MLP instance (must implement __call__ and parameters()).
            optimizer: Optimizer instance (must have step() and lr).
            dataset: Dataset instance (training data).
            metrics_calculator: MetricsCalculator instance (for validation).
            scheduler: Optional learning rate scheduler.
            checkpoint_dir: Optional directory for saving checkpoints.
            clip_value: Optional gradient clipping threshold (positive float).

        Raises:
            TypeError: If arguments have incorrect types.
            ValueError: If clip_value is non‑positive or non‑finite.
        """
        # Validate model
        if not isinstance(model, MLP):
            raise TypeError(f"model must be an MLP, got {type(model).__name__}")
        self.model = model

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"optimizer must be an Optimizer, got {type(optimizer).__name__}"
            )
        self.optimizer = optimizer

        # Validate dataset
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset, got {type(dataset).__name__}")
        self.dataset = dataset

        # Validate metrics calculator
        if not isinstance(metrics_calculator, MetricsCalculator):
            raise TypeError(
                f"metrics_calculator must be a MetricsCalculator, "
                f"got {type(metrics_calculator).__name__}"
            )
        self.metrics_calculator = metrics_calculator

        # Validate scheduler (optional)
        if scheduler is not None and not isinstance(scheduler, BaseLRScheduler):
            raise TypeError(
                f"scheduler must be a BaseLRScheduler, "
                f"got {type(scheduler).__name__}"
            )
        self.scheduler = scheduler

        # Validate checkpoint directory (optional)
        if checkpoint_dir is not None:
            if not isinstance(checkpoint_dir, str):
                raise TypeError(
                    f"checkpoint_dir must be a str, got {type(checkpoint_dir).__name__}"
                )
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Validate gradient clip value (optional)
        if clip_value is not None:
            if not isinstance(clip_value, (int, float)):
                raise TypeError(
                    f"clip_value must be a number, got {type(clip_value).__name__}"
                )
            if clip_value <= 0:
                raise ValueError(f"clip_value must be positive, got {clip_value}")
            if not math.isfinite(clip_value):
                raise ValueError(f"clip_value must be finite, got {clip_value}")
        self.clip_value = clip_value

        # Internal state
        self.history: list[dict[str, float]] = []  # stores one dict per epoch
        self.best_val_loss: float = float("inf")
        self.best_val_macro_f1: float = -1.0
        self._epoch: int = 0  # current epoch number (1‑based during training)

        logger.info(
            f"TrainingPipeline initialized with "
            f"dataset_size={dataset.num_samples}, "
            f"input_dim={dataset.input_dim}, "
            f"num_classes={metrics_calculator.num_classes}, "
            f"optimizer={optimizer.__class__.__name__}, "
            f"scheduler={scheduler.__class__.__name__ if scheduler else 'None'}, "
            f"clip_value={clip_value}"
        )

    # --------------------------------------------------------------------------
    # Forward, Loss, Gradients, Clipping – Internal Helpers
    # --------------------------------------------------------------------------

    def _forward_batch(self, batch_X: list[list[float]]) -> list[list[Value]]:
        """
        Run forward pass for a batch through the model.

        Input:  batch_X – list of sample feature vectors.
        Output: batch_logits – list of logit Value lists (one per sample).

        The model is callable (MLP.__call__) – this method handles that.
        It also validates the output shape and type to catch errors early.
        """
        batch_logits: list[list[Value]] = []
        for x in batch_X:
            # Convert features to Value objects if they aren't already
            x_values = [
                Value(float(feat)) if not isinstance(feat, Value) else feat
                for feat in x
            ]
            # Forward pass – MLP uses __call__, not .forward()
            logits = self.model(x_values)

            # Defensive checks
            if not isinstance(logits, list):
                raise TypeError(
                    f"Model forward returned {type(logits).__name__}, expected list"
                )
            if len(logits) != self.metrics_calculator.num_classes:
                raise TypeError(
                    f"Model output has {len(logits)} logits, "
                    f"but num_classes={self.metrics_calculator.num_classes}"
                )
            for v in logits:
                if not isinstance(v, Value):
                    raise TypeError(
                        f"Model output contains {type(v).__name__}, expected Value"
                    )

            batch_logits.append(logits)
        return batch_logits

    def _compute_batch_loss(
        self, batch_logits: list[list[Value]], batch_y: list[int]
    ) -> Value:
        """
        Compute the average cross‑entropy loss for a batch.

        Steps:
            1. For each sample: softmax(logits) → probabilities.
            2. categorical_cross_entropy(probs, target) → scalar loss Value.
            3. Average losses across the batch (returns a single Value).

        The returned Value is a scalar that represents the batch‑average loss.
        It is attached to the computation graph, so calling .backward() on it
        will backpropagate gradients to all model parameters.
        """
        if not batch_logits:
            raise ValueError("Cannot compute loss for empty batch")
        if len(batch_logits) != len(batch_y):
            raise ValueError(
                f"batch_logits length ({len(batch_logits)}) "
                f"does not match batch_y length ({len(batch_y)})"
            )

        losses: list[Value] = []
        for logits, target in zip(batch_logits, batch_y):
            # Validate target
            num_classes = self.metrics_calculator.num_classes
            if target < 0 or target >= num_classes:
                raise ValueError(
                    f"Invalid target {target} for num_classes={num_classes}"
                )

            # Softmax + cross‑entropy
            probs = softmax(logits)
            loss = categorical_cross_entropy(probs, target)
            losses.append(loss)

        # If only one sample, return its loss directly
        if len(losses) == 1:
            return losses[0]

        # Average: sum(losses) / batch_size
        total_loss = losses[0]
        for l in losses[1:]:
            total_loss = total_loss + l
        batch_size_value = Value(float(len(losses)))
        avg_loss = total_loss / batch_size_value
        return avg_loss

    def _zero_gradients(self) -> None:
        """
        Reset gradients for all model parameters.

        We reset both 'grad' (used by Day 29 optimizers) and 'gradient'
        (used by Day 24 autograd) to avoid accumulation.
        """
        for p in self.model.parameters():
            p.grad = 0.0
            p.gradient = 0.0

    def _sync_gradients(self) -> None:
        """
        Copy gradients from 'gradient' (Day 24) to 'grad' (Day 29).

        After loss.backward(), the autograd engine updates p.gradient.
        The optimizers from Day 29 expect p.grad. This method bridges them.
        """
        for p in self.model.parameters():
            p.grad = p.gradient

    def _apply_gradient_clipping(self) -> None:
        """
        Apply global L2 norm clipping to all gradients.

        If the L2 norm of all gradients exceeds clip_value, scale them down
        so that the norm becomes exactly clip_value. This is the classic
        gradient clipping technique to prevent exploding gradients.
        """
        if self.clip_value is None:
            return

        params = self.model.parameters()
        grads = [p.grad for p in params]
        norm_sq = sum(g * g for g in grads)
        norm = math.sqrt(norm_sq)

        if norm > self.clip_value:
            scale = self.clip_value / norm
            for p in params:
                p.grad *= scale

    # --------------------------------------------------------------------------
    # Training and Validation Epochs
    # --------------------------------------------------------------------------

    def _train_epoch(self, batch_size: int, shuffle: bool = True) -> float:
        """
        Run one full training epoch over the dataset.

        Steps per batch:
            1. Zero gradients.
            2. Forward pass → logits.
            3. Compute batch‑average loss.
            4. Backward pass (computes gradients).
            5. Sync gradients (gradient → grad).
            6. Apply gradient clipping (if enabled).
            7. Optimizer step (update parameters).

        Returns:
            Sample‑weighted average loss for the epoch.
            (Each batch's loss is weighted by its actual number of samples,
             so the epoch loss is a true average over samples, not batches.)
        """
        # Ensure model is in training mode (if it has the flag)
        if hasattr(self.model, "train"):
            self.model.train()

        total_loss = 0.0
        total_samples = 0

        # Iterate over mini‑batches
        for batch_X, batch_y in self.dataset.batches(
            batch_size=batch_size, shuffle=shuffle, seed=self._epoch
        ):
            self._zero_gradients()

            batch_logits = self._forward_batch(batch_X)
            loss_value = self._compute_batch_loss(batch_logits, batch_y)

            # Backward pass (autograd)
            loss_value.backward()

            # Bridge to optimizer API
            self._sync_gradients()

            # Clip if needed
            self._apply_gradient_clipping()

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss weighted by actual batch size
            batch_size_actual = len(batch_y)
            total_loss += loss_value.value * batch_size_actual
            total_samples += batch_size_actual

        if total_samples == 0:
            raise RuntimeError("Training epoch produced no samples")

        return total_loss / total_samples

    def _validate(self, val_dataset: Dataset) -> tuple[float, float, float]:
        """
        Evaluate the model on a validation dataset.

        Note: This still builds the autograd Value graph (since we use the same
        forward/loss functions), but we intentionally do NOT call .backward().
        Thus parameters are not updated – this is validation-only.

        Steps:
            1. Switch to eval mode (disables dropout, if any).
            2. Run forward passes without backward/update.
            3. Compute validation loss (sample‑weighted average).
            4. Collect predictions (argmax of probabilities).
            5. Compute accuracy and Macro‑F1 via MetricsCalculator.

        Returns:
            Tuple (val_loss, accuracy, macro_f1).

        Raises:
            ValueError: If val_dataset is empty.
        """
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset cannot be empty")

        # Remember if we were in training mode to restore later
        was_training = False
        if hasattr(self.model, "training"):
            was_training = self.model.training

        try:
            # Switch to eval mode
            if hasattr(self.model, "eval"):
                self.model.eval()

            all_preds: list[int] = []
            all_targets: list[int] = []
            total_loss = 0.0
            total_samples = 0

            # We use one big batch (the whole validation set) for simplicity
            for batch_X, batch_y in val_dataset.batches(
                batch_size=len(val_dataset), shuffle=False
            ):
                # Forward pass (no backward/update)
                batch_logits = self._forward_batch(batch_X)
                loss_value = self._compute_batch_loss(batch_logits, batch_y)

                # Accumulate loss weighted by batch size
                total_loss += loss_value.value * len(batch_y)
                total_samples += len(batch_y)

                # Get predictions: argmax of softmax probabilities
                for logits in batch_logits:
                    probs = softmax(logits)
                    pred_class = max(range(len(probs)), key=lambda i: probs[i].value)
                    all_preds.append(pred_class)

                all_targets.extend(batch_y)

            # Compute metrics
            accuracy = self.metrics_calculator.accuracy(all_targets, all_preds)
            macro_f1 = self.metrics_calculator.macro_f1(all_targets, all_preds)

            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            return avg_loss, accuracy, macro_f1

        finally:
            # Restore training mode if it was previously on
            if was_training and hasattr(self.model, "train"):
                self.model.train()

    # --------------------------------------------------------------------------
    # Dataset Splitting
    # --------------------------------------------------------------------------

    def _split_dataset(
        self, validation_split: float, seed: int | None = None
    ) -> tuple[Dataset, Dataset | None]:
        """
        Split the dataset into training and validation sets.

        Args:
            validation_split: Fraction of data to reserve for validation (0.0 to <1.0).
            seed: Random seed for deterministic shuffle.

        Returns:
            (train_dataset, val_dataset) where val_dataset is None if split is 0.0.

        Raises:
            ValueError: If validation_split is out of range, or if the split
                       would result in an empty training or validation set.
        """
        if not 0.0 <= validation_split < 1.0:
            raise ValueError(
                f"validation_split must be in [0.0, 1.0), got {validation_split}"
            )

        # No validation set requested
        if validation_split == 0.0:
            return self.dataset, None

        indices = list(range(self.dataset.num_samples))
        rng = random.Random(seed)
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1.0 - validation_split))

        # Ensure both sets have at least one sample
        if split_idx == 0 or split_idx >= len(indices):
            raise ValueError(
                f"validation_split={validation_split} would produce empty training or validation set. "
                f"Dataset has {len(indices)} samples."
            )

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_X = [self.dataset.X[i] for i in train_indices]
        train_y = [self.dataset.y[i] for i in train_indices]
        train_dataset = Dataset(train_X, train_y)

        val_X = [self.dataset.X[i] for i in val_indices]
        val_y = [self.dataset.y[i] for i in val_indices]
        val_dataset = Dataset(val_X, val_y)

        return train_dataset, val_dataset

    # --------------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------------

    def _serialize_optimizer_state(self) -> dict[str, Any]:
        """
        Extract serializable optimizer state for checkpointing.

        This avoids dumping the optimizer's __dict__ which may contain
        non‑JSON‑serializable objects (e.g., Value references). Instead,
        we explicitly copy the numeric states (lr, momentum, velocities, etc.)

        Returns a dictionary that can be JSON‑encoded.
        """
        opt = self.optimizer
        state: dict[str, Any] = {
            "class": opt.__class__.__name__,
            "lr": opt.lr,
        }

        # Handle different optimizer types
        if isinstance(opt, SGD):
            # Handles both plain SGD and momentum SGD (same class)
            state["momentum"] = getattr(opt, "momentum", 0.0)
            velocity = getattr(opt, "v", None)
            if velocity is not None:
                state["velocity"] = list(velocity)

        elif isinstance(opt, Adam):
            state["beta1"] = getattr(opt, "beta1", 0.9)
            state["beta2"] = getattr(opt, "beta2", 0.999)
            state["eps"] = getattr(opt, "eps", 1e-8)
            m = getattr(opt, "m", None)
            if m is not None:
                state["m"] = list(m)
            v = getattr(opt, "v", None)
            if v is not None:
                state["v"] = list(v)
            state["t"] = getattr(opt, "t", 0)

        elif isinstance(opt, RMSprop):
            state["alpha"] = getattr(opt, "alpha", 0.99)
            state["eps"] = getattr(opt, "eps", 1e-8)
            s = getattr(opt, "s", None)
            if s is not None:
                state["s"] = list(s)

        # If we encounter an unknown optimizer, we still have class and lr
        return state

    def _save_checkpoint(
        self, epoch: int, val_loss: float, val_macro_f1: float
    ) -> None:
        """
        Save a checkpoint using Day 32's atomic save function.

        The checkpoint includes:
            - Model parameters (values and gradients)
            - Optimizer state (explicitly serialized)
            - Scheduler state (if present)
            - Epoch number
            - Validation metrics

        The best checkpoint (based on Macro‑F1) is also saved separately.
        """
        if self.checkpoint_dir is None:
            return

        # Model state: list of dicts with value and gradient (JSON‑friendly)
        model_state_list = [
            {"value": p.value, "gradient": p.gradient}
            for p in self.model.parameters()
        ]
        # Day 32 expects a dict with a 'parameters' key
        model_state = {"parameters": model_state_list}

        # Optimizer state (explicitly serialized)
        optimizer_state = self._serialize_optimizer_state()

        # Scheduler state (if available)
        scheduler_state = None
        if self.scheduler is not None:
            scheduler_state = {
                "class": self.scheduler.__class__.__name__,
                "last_epoch": getattr(self.scheduler, "last_epoch", 0),
                "base_lr": getattr(self.scheduler, "base_lr", 0.0),
                "config": {
                    k: v
                    for k, v in self.scheduler.__dict__.items()
                    if not callable(v) and not isinstance(v, (list, dict, set))
                },
            }

        metrics = {
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
        }

        # File paths
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        best_path = self.checkpoint_dir / "best_checkpoint.json"

        # Save using Day 32's static method (atomic)
        ModelCheckpoint.save_checkpoint(
            filepath=str(checkpoint_path),
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=epoch,
            metrics=metrics,
            scheduler_state=scheduler_state,
        )

        # Also save as best if this is the best so far
        if val_macro_f1 == self.best_val_macro_f1:
            ModelCheckpoint.save_checkpoint(
                filepath=str(best_path),
                model_state=model_state,
                optimizer_state=optimizer_state,
                epoch=epoch,
                metrics=metrics,
                scheduler_state=scheduler_state,
            )
            logger.info(f"Saved best checkpoint to {best_path}")

    # --------------------------------------------------------------------------
    # Public Training API
    # --------------------------------------------------------------------------

    def train(
        self,
        epochs: int,
        batch_size: int,
        validation_split: float = 0.2,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> list[dict[str, float]]:
        """
        Run the complete training loop for a given number of epochs.

        For each epoch:
            1. (Optionally) step the learning rate scheduler.
            2. Train for one epoch (updates model).
            3. Validate (if validation set exists).
            4. Update best metrics and checkpoint if improved.
            5. Record history.

        Args:
            epochs: Number of epochs to train (must be >0).
            batch_size: Number of samples per batch (must be >0).
            validation_split: Fraction of data to hold out for validation (0.0 to <1.0).
            shuffle: Whether to shuffle the dataset each epoch.
            seed: Random seed for dataset split and shuffling (reproducibility).

        Returns:
            List of history dictionaries, one per epoch. Each dict contains:
                - epoch: int
                - train_loss: float
                - val_loss: float (or NaN if no validation)
                - val_accuracy: float (or NaN)
                - val_macro_f1: float (or NaN)
                - learning_rate: float

        Raises:
            ValueError: If epochs <=0, batch_size <=0, or validation_split invalid.
        """
        # Input validation
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # If batch_size > dataset size, clamp it to dataset size (with a warning)
        if batch_size > self.dataset.num_samples:
            logger.warning(
                f"batch_size ({batch_size}) > dataset size ({self.dataset.num_samples}). "
                f"Using batch_size = {self.dataset.num_samples}"
            )
            batch_size = self.dataset.num_samples

        # Split dataset
        train_dataset, val_dataset = self._split_dataset(validation_split, seed)

        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty after split")

        # Temporarily replace the dataset with the training subset
        original_dataset = self.dataset
        self.dataset = train_dataset

        try:
            for epoch in range(1, epochs + 1):
                self._epoch = epoch

                # --- Learning Rate Scheduling (step before training) ---
                # For epoch 1 we use the initial LR; for subsequent epochs we step.
                if self.scheduler is not None and epoch > 1:
                    self.scheduler.step()
                current_lr = getattr(self.optimizer, "lr", 0.0)

                # --- Training ---
                train_loss = self._train_epoch(batch_size, shuffle=shuffle)

                # --- Validation ---
                if val_dataset is not None and len(val_dataset) > 0:
                    val_loss, val_accuracy, val_macro_f1 = self._validate(val_dataset)
                else:
                    val_loss = float("nan")
                    val_accuracy = float("nan")
                    val_macro_f1 = float("nan")

                # --- Update best metrics ---
                if (
                    not math.isnan(val_macro_f1)
                    and val_macro_f1 > self.best_val_macro_f1
                ):
                    self.best_val_macro_f1 = val_macro_f1
                    self.best_val_loss = val_loss

                # --- Checkpoint (if validation metrics available) ---
                if self.checkpoint_dir is not None and not math.isnan(val_macro_f1):
                    self._save_checkpoint(epoch, val_loss, val_macro_f1)

                # --- Record history ---
                epoch_record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "learning_rate": current_lr,
                }
                self.history.append(epoch_record)

                # --- Log progress ---
                log_msg = (
                    f"Epoch {epoch}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"lr={current_lr:.6f}"
                )
                if not math.isnan(val_loss):
                    log_msg += (
                        f" | val_loss={val_loss:.4f} | "
                        f"val_acc={val_accuracy:.4f} | "
                        f"val_macro_f1={val_macro_f1:.4f}"
                    )
                logger.info(log_msg)

        finally:
            # Restore original dataset (important for future calls)
            self.dataset = original_dataset

        return self.history

    def get_best_epoch(self) -> int | None:
        """
        Return the epoch number that produced the best validation Macro‑F1.

        Returns None if no checkpoint was ever saved.
        """
        if self.best_val_macro_f1 < 0:
            return None
        for record in self.history:
            if (
                not math.isnan(record["val_macro_f1"])
                and record["val_macro_f1"] == self.best_val_macro_f1
            ):
                return record["epoch"]
        return None

    def get_history(self) -> list[dict[str, float]]:
        """Return a copy of the training history (list of epoch records)."""
        return self.history.copy()