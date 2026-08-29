"""
Day 37: Image Augmentations & Preprocessing Pipelines from First Principles.
Provides spatial and intensity transformations for multi-channel 3D image tensors (C, H, W).
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class Normalize:
    """
    Channel-wise Z-score normalization: (X - mean) / (std + eps).

    Args:
        mean: Sequence of length C containing per-channel means.
        std:  Sequence of length C containing per-channel standard deviations.
        eps:  Small constant for numerical stability (must be >= 0, default 1e-8).

    Raises:
        ValueError: If mean/std lengths mismatch, are empty, if eps < 0, or if any std < 0.
    """

    def __init__(self, mean: list[float], std: list[float], eps: float = 1e-8) -> None:
        if not mean or len(mean) != len(std):
            raise ValueError("mean and std must be non-empty and have equal length.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        if any(v < 0 for v in std):
            raise ValueError("std values must be non-negative.")

        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.eps = eps
        self._n_channels = len(mean)
        logger.debug("Normalize: %d channels, eps=%f", self._n_channels, eps)

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W), got shape {img.shape}")

        # Convert integer images to float to avoid truncating normalization stats.
        if not img.is_floating_point():
            img = img.float()

        C = img.size(0)
        if C != self._n_channels:
            raise ValueError(
                f"Channel mismatch: img has {C} channels, mean/std have {self._n_channels}."
            )

        # Move statistics to the same device as the image, but keep them as float.
        mean = self.mean.to(device=img.device)
        std = self.std.to(device=img.device)

        return (img - mean) / (std + self.eps)


class RandomHorizontalFlip:
    """
    Horizontally flips a 3D image tensor with probability p.

    Args:
        p:   Probability of applying the flip (default 0.5).
        rng: Optional random.Random instance for reproducible randomness.
             If None, creates a new random.Random() instance.
    """

    def __init__(self, p: float = 0.5, rng: random.Random | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {img.shape}")
        if self.rng.random() < self.p:
            logger.debug("Horizontal flip applied.")
            return torch.flip(img, dims=[2])
        return img


class RandomCrop:
    """
    Pads image with zeros then extracts a random crop of given size.

    Args:
        crop_size: Desired output (height, width). If int, square crop.
        padding:   Number of zeros added to each side (default 0).
        rng:       Optional random.Random instance for reproducible randomness.

    Raises:
        ValueError: If crop_size is not an int or a tuple of two ints,
                    if crop_size components are not positive integers,
                    if padding is negative,
                    or if crop size exceeds padded image dimensions.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int],
        padding: int = 0,
        rng: random.Random | None = None,
    ) -> None:
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        elif isinstance(crop_size, tuple) and len(crop_size) == 2:
            if not all(isinstance(v, int) for v in crop_size):
                raise ValueError(
                    "crop_size tuple must contain two integers."
                )
            self.crop_h, self.crop_w = crop_size
        else:
            raise ValueError("crop_size must be an int or a tuple of two ints.")

        if self.crop_h <= 0 or self.crop_w <= 0:
            raise ValueError(f"crop_size components must be positive, got ({self.crop_h}, {self.crop_w})")
        if padding < 0:
            raise ValueError("padding must be non-negative.")

        self.padding = padding
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {img.shape}")

        _C, H, W = img.shape

        if self.padding > 0:
            img = F.pad(
                img,
                (self.padding, self.padding, self.padding, self.padding),
                mode='constant',
                value=0
            )
            H += 2 * self.padding
            W += 2 * self.padding

        if self.crop_h > H or self.crop_w > W:
            raise ValueError(
                f"Crop size ({self.crop_h}, {self.crop_w}) exceeds "
                f"image dimensions ({H}, {W}) after padding."
            )

        start_h = self.rng.randint(0, H - self.crop_h)
        start_w = self.rng.randint(0, W - self.crop_w)
        logger.debug("Random crop: start=(%d,%d), size=(%d,%d)", start_h, start_w, self.crop_h, self.crop_w)
        return img[:, start_h:start_h + self.crop_h, start_w:start_w + self.crop_w]


class Compose:
    """Chains multiple transforms sequentially."""

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img