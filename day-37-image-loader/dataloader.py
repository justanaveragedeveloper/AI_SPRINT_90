"""
Day 37: PyTorch-style Dataset and DataLoader implementations.
Supports dataset indexing, augmentation, shuffling, and mini-batch creation.
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
    """Abstract base dataset interface."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        raise NotImplementedError


class SimpleImageDataset(Dataset):
    """
    In-memory image dataset with optional shape validation.

    Args:
        images: List of 3D tensors (C, H, W) or objects convertible to tensor.
        labels: List of integer labels (same length as images).
        transform: Optional callable applied to each image on retrieval.
        validate_shapes: If True (default), enforce that all images have
                         the same (C, H, W) dimensions.

    Raises:
        ValueError: If lengths of images and labels differ,
                    if any image is not 3D after conversion,
                    or if validate_shapes=True and shapes differ.
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

        ref_shape: tuple[int, int, int] | None = None

        for i, img in enumerate(images):
            if not isinstance(img, Tensor):
                img = torch.tensor(img, dtype=torch.float32)

            if img.ndim != 3:
                raise ValueError(
                    f"Image at index {i} must have shape (C,H,W), got {tuple(img.shape)}"
                )

            shape = tuple(img.shape)  # (C, H, W)

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
    Mini-batch DataLoader with shuffling and drop_last.

    Args:
        dataset:    Dataset instance.
        batch_size: Number of samples per batch (positive int).
        shuffle:    Whether to shuffle indices at each epoch.
        drop_last:  If True, drop the last incomplete batch.
        rng:        Optional random.Random instance for reproducible shuffling.
                    If None, a new random.Random() is created.

    Raises:
        ValueError: If batch_size <= 0.
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
        self.rng = rng if rng is not None else random.Random()

        logger.debug(
            "DataLoader: batch_size=%d, shuffle=%s, drop_last=%s",
            batch_size, shuffle, drop_last
        )

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(indices)

        batch_images: list[Tensor] = []
        batch_labels: list[int] = []

        for idx in indices:
            img, label = self.dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

            if len(batch_images) == self.batch_size:
                yield self._make_batch(batch_images, batch_labels)
                batch_images.clear()
                batch_labels.clear()

        if batch_images and not self.drop_last:
            yield self._make_batch(batch_images, batch_labels)

    @staticmethod
    def _make_batch(images: list[Tensor], labels: list[int]) -> tuple[Tensor, Tensor]:
        """
        Stack a list of image tensors and labels into a single batch.

        Raises:
            ValueError: If images have inconsistent shapes.
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

        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)