"""
Day 37: Image Preprocessing & Augmentation (First Principles)

This module implements channel‑wise normalization and common spatial augmentations
for 3D image tensors in (C, H, W) format.

All transforms are:
- Deterministic or stochastic (with injectable RNG for reproducibility)
- Vectorised (using PyTorch operations)
- Defensively validated (input shape, parameter bounds)

They are designed to be composed via `Compose`.
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
    Channel‑wise Z‑score normalisation: (X - mean) / (std + eps).

    Why channel‑wise?
    ────────────────
    Each colour channel (R, G, B) has a different intensity distribution.
    Using the same mean/std for all channels would distort colour information.
    Instead, we compute mean and std separately for each channel.

    Why eps?
    ────────
    Prevents division by zero when a channel has zero variance (e.g., all pixels black).
    Even if std=0, eps=1e‑8 ensures the output is finite.
    """

    def __init__(self, mean: list[float], std: list[float], eps: float = 1e-8) -> None:
        # Defensive checks: statistics must be valid
        if not mean or len(mean) != len(std):
            raise ValueError("mean and std must be non‑empty and have equal length.")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        if any(v < 0 for v in std):
            raise ValueError("std values must be non-negative.")

        # Store statistics as (C,1,1) tensors for broadcasting against (C,H,W) images.
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.eps = eps
        self._n_channels = len(mean)
        logger.debug("Normalize: %d channels, eps=%f", self._n_channels, eps)

    def __call__(self, img: Tensor) -> Tensor:
        # Input must be a 3D tensor (C, H, W)
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C,H,W), got shape {img.shape}")

        # Normalization is a floating‑point operation. If the image is integer,
        # convert it to float to avoid truncating the mean/subtraction.
        if not img.is_floating_point():
            img = img.float()

        # Check that the number of channels matches the statistics.
        C = img.size(0)
        if C != self._n_channels:
            raise ValueError(
                f"Channel mismatch: img has {C} channels, mean/std have {self._n_channels}."
            )

        # Move statistics to the same device as the image (CPU/GPU) but keep them as float.
        mean = self.mean.to(device=img.device)
        std = self.std.to(device=img.device)

        # Vectorised channel‑wise normalisation:
        #   (img - mean) / (std + eps)
        # Broadcasting applies the correct mean/std to each channel.
        return (img - mean) / (std + self.eps)


class RandomHorizontalFlip:
    """
    Randomly flips an image horizontally with probability p.

    Mathematical mapping:
        (c, i, j)  ->  (c, i, W - 1 - j)
    where W is the image width (dimension 2).

    Why horizontal flipping?
    ────────────────────────
    Many objects are left‑right symmetric (cats, cars, etc.).
    This augmentation teaches the model that mirror images are semantically
    the same, improving generalisation.

    RNG injection:
    ──────────────
    Each instance can have its own random.Random object, making experiments
    reproducible without affecting the global `random` state.
    """

    def __init__(self, p: float = 0.5, rng: random.Random | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        # Use a dedicated RNG if provided; otherwise create a new one.
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {img.shape}")

        # Flip the width dimension (index 2) with probability p.
        if self.rng.random() < self.p:
            logger.debug("Horizontal flip applied.")
            return torch.flip(img, dims=[2])   # vectorised: reverses the width axis
        return img


class RandomCrop:
    """
    Randomly crops an image after optional zero‑padding.

    How it works:
    1. If padding > 0, add zeros to all four sides:
           H' = H + 2*padding,  W' = W + 2*padding
    2. Choose a random top‑left corner (start_h, start_w) such that:
           0 ≤ start_h ≤ H' - crop_h
           0 ≤ start_w ≤ W' - crop_w
    3. Extract the sub‑window of size (crop_h, crop_w).

    Why padding?
    ────────────
    When cropping, padding allows the crop to extend beyond the original
    image boundaries, effectively filling with zeros. This is common in
    classification (e.g., Inception) to avoid losing border information.

    Why random?
    ────────────
    Random cropping forces the model to recognise objects regardless of their
    position in the frame, improving translation invariance.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int],
        padding: int = 0,
        rng: random.Random | None = None,
    ) -> None:
        # Validate crop_size: can be an int (square) or a tuple of two ints.
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        elif isinstance(crop_size, tuple) and len(crop_size) == 2:
            # Ensure both elements are integers (not floats, bools, etc.)
            if not all(isinstance(v, int) for v in crop_size):
                raise ValueError("crop_size tuple must contain two integers.")
            self.crop_h, self.crop_w = crop_size
        else:
            raise ValueError("crop_size must be an int or a tuple of two ints.")

        # Crop dimensions must be positive.
        if self.crop_h <= 0 or self.crop_w <= 0:
            raise ValueError(f"crop_size components must be positive, got ({self.crop_h}, {self.crop_w})")
        if padding < 0:
            raise ValueError("padding must be non‑negative.")

        self.padding = padding
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, img: Tensor) -> Tensor:
        if img.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {img.shape}")

        _C, H, W = img.shape   # _C is unused but kept for clarity

        # Apply zero‑padding if requested.
        if self.padding > 0:
            # F.pad expects padding order: (left, right, top, bottom)
            img = F.pad(
                img,
                (self.padding, self.padding, self.padding, self.padding),
                mode='constant',
                value=0
            )
            H += 2 * self.padding
            W += 2 * self.padding

        # After padding, the crop must fit inside the padded image.
        if self.crop_h > H or self.crop_w > W:
            raise ValueError(
                f"Crop size ({self.crop_h}, {self.crop_w}) exceeds "
                f"image dimensions ({H}, {W}) after padding."
            )

        # Randomly choose the top‑left corner.
        start_h = self.rng.randint(0, H - self.crop_h)
        start_w = self.rng.randint(0, W - self.crop_w)
        logger.debug("Random crop: start=(%d,%d), size=(%d,%d)", start_h, start_w, self.crop_h, self.crop_w)

        # Extract the crop using vectorised slicing.
        # Result shape: (C, crop_h, crop_w)
        return img[:, start_h:start_h + self.crop_h, start_w:start_w + self.crop_w]


class Compose:
    """
    Chains multiple transforms together.

    Example:
        pipeline = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomHorizontalFlip(p=0.5),
            RandomCrop(crop_size=224, padding=8)
        ])
        augmented_image = pipeline(original_image)

    Transform order matters! E.g., crop after flip gives different results
    than flip after crop. This class guarantees sequential application.
    """

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img