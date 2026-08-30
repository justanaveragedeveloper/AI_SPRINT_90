"""
Day 37: Dataset and DataLoader (First‑Principles)

This module provides:
- `Dataset`: abstract interface for retrieving a single sample (image, label).
- `SimpleImageDataset`: in‑memory dataset with optional shape validation.
- `DataLoader`: organises samples into mini‑batches, with shuffling and drop_last.

The pipeline bridges raw images to a CNN:
    (C,H,W)  ->  Dataset  ->  DataLoader  ->  (B,C,H,W)  ->  Day 36 CNN

Shape invariance:
    Every image in a batch must have exactly the same shape (C, H, W).
    The DataLoader validates this before stacking.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class Dataset:
    """
    Abstract base class for datasets.

    Subclasses must implement:
        - __len__(): number of samples
        - __getitem__(idx): returns (image, label) for the given index
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        raise NotImplementedError


class SimpleImageDataset(Dataset):
    """
    In‑memory image dataset with optional shape validation.

    Args:
        images:         List of 3D tensors (C, H, W) or objects convertible to tensor.
        labels:         List of integer labels (same length as images).
        transform:      Optional transform to apply on retrieval.
        validate_shapes: If True, enforce that all images have identical (C,H,W).

    Why validate shapes?
    ────────────────────
    Convolutional networks expect a fixed input shape. Early validation prevents
    cryptic errors later when trying to stack batches.
    """

    def __init__(
        self,
        images: Sequence[Tensor | Any],
        labels: Sequence[int],
        transform: Callable | None = None,
        validate_shapes: bool = True,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("Number of images and labels must match.")

        self.images: list[Tensor] = []
        self.labels = list(labels)
        self.transform = transform

        # Reference shape: (C, H, W) of the first image (if any).
        ref_shape: tuple[int, int, int] | None = None

        for i, img in enumerate(images):
            # Convert non‑tensor inputs (e.g., NumPy arrays) to torch tensors.
            if not isinstance(img, Tensor):
                img = torch.tensor(img, dtype=torch.float32)

            # A valid image must be 3D.
            if img.ndim != 3:
                raise ValueError(
                    f"Image at index {i} must have shape (C,H,W), got {tuple(img.shape)}"
                )

            shape = tuple(img.shape)  # (C, H, W)

            # If validation is enabled, ensure all shapes match exactly.
            if validate_shapes:
                if ref_shape is None:
                    ref_shape = shape
                elif shape != ref_shape:
                    raise ValueError(
                        f"Image {i} has shape {shape}, expected {ref_shape}. "
                        "All images must have the same dimensions (C, H, W)."
                    )

            self.images.append(img)

        logger.debug("Dataset initialized with %d images.", len(self.images))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DataLoader:
    """
    Organises a Dataset into mini‑batches.

    Key responsibilities:
    - Shuffling indices (not the data itself) to preserve image‑label alignment.
    - Grouping samples into batches of fixed size.
    - Optionally dropping the last incomplete batch (`drop_last`).

    Why shuffle indices, not images/labels?
    ──────────────────────────────────────
    If we shuffled images and labels separately, we would break the alignment.
    Shuffling indices ensures that (image[i], label[i]) always stay together.

    Why batch_size?
    ───────────────
    Mini‑batch SGD computes gradients over a small subset, trading off
    variance (noisy gradients) against computational efficiency (GPU utilisation).

    Why drop_last?
    ──────────────
    drop_last=True ensures every yielded batch has exactly B samples.
    This can be useful when a training setup expects fixed batch sizes.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # Use a dedicated RNG for shuffling to avoid interfering with other randomness.
        self.rng = rng if rng is not None else random.Random()

        logger.debug(
            "DataLoader: batch_size=%d, shuffle=%s, drop_last=%s",
            batch_size, shuffle, drop_last
        )

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size          # floor(N / B)
        return (n + self.batch_size - 1) // self.batch_size  # ceil(N / B)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield mini‑batches as (batch_images, batch_labels)."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(indices)   # shuffles in‑place

        batch_images: list[Tensor] = []
        batch_labels: list[int] = []

        for idx in indices:
            img, label = self.dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

            # When we have collected exactly batch_size samples, pack and yield.
            if len(batch_images) == self.batch_size:
                yield self._make_batch(batch_images, batch_labels)
                batch_images.clear()
                batch_labels.clear()

        # After the loop, if there are leftover samples and drop_last is False,
        # yield the final (partial) batch.
        if batch_images and not self.drop_last:
            yield self._make_batch(batch_images, batch_labels)

    @staticmethod
    def _make_batch(images: list[Tensor], labels: list[int]) -> tuple[Tensor, Tensor]:
        """
        Stack a list of image tensors and labels into a single batch.

        Raises ValueError if images have inconsistent shapes.

        Why check shapes here?
        ──────────────────────
        Even if the Dataset validated shapes, the batch could still contain
        inconsistent shapes if the Dataset was constructed with validate_shapes=False.
        This method provides a final safety net.
        """
        if not images:
            raise ValueError("Cannot make batch from empty list.")

        ref_shape = images[0].shape  # (C, H, W)
        for i, img in enumerate(images[1:], start=1):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Inconsistent shapes in batch: image 0 has {ref_shape}, "
                    f"image {i} has {img.shape}. All images must have identical shape."
                )

        # Stack along the batch dimension (dimension 0).
        # Result: (B, C, H, W)
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)