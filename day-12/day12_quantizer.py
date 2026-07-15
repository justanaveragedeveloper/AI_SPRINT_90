"""
Simple SQ8 Quantizer implementation.

Converts float32 vectors into uint8 representations
to reduce memory usage in vector databases and
retrieval systems.
"""

from __future__ import annotations

import struct
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


class QuantizationError(Exception):
    """Raised when quantization input is invalid."""


class QuantizedVectorBroker:
    """
    Performs simple uint8 vector quantization.

    Header format:
        scale      -> float32
        zero_point -> float32
        dimension  -> uint32
    """

    HEADER_FORMAT = "<ffI"

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")

        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self.header_size = struct.calcsize(self.HEADER_FORMAT)

    def compress(
        self,
        vector: Iterable[float] | NDArray[np.float32],
    ) -> bytes:
        """
        Compress float vector into quantized bytes.
        """

        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1:
            raise QuantizationError("Input vector must be one-dimensional.")

        if vector.size != self.dimension:
            raise QuantizationError(
                f"Expected dimension {self.dimension}, " f"got {vector.size}."
            )

        if not np.isfinite(vector).all():
            raise QuantizationError("Vector contains NaN or infinite values.")

        v_min = float(vector.min())
        v_max = float(vector.max())

        if np.isclose(v_min, v_max):
            scale = 1.0
            zero_point = 0.0

            quantized = np.clip(
                np.round(vector),
                0,
                255,
            ).astype(np.uint8)

        else:
            scale = (v_max - v_min) / 255.0

            zero_point = float(
                np.clip(
                    round(-v_min / scale),
                    0,
                    255,
                )
            )

            quantized = np.clip(
                np.round(vector / scale + zero_point),
                0,
                255,
            ).astype(np.uint8)

        header = struct.pack(
            self.HEADER_FORMAT,
            scale,
            zero_point,
            self.dimension,
        )

        return header + quantized.tobytes()

    def decompress(
        self,
        payload: bytes,
    ) -> NDArray[np.float32]:
        """
        Recover vector from compressed bytes.
        """

        if len(payload) < self.header_size:
            raise QuantizationError("Payload too small.")

        scale, zero_point, dimension = struct.unpack(
            self.HEADER_FORMAT,
            payload[: self.header_size],
        )

        if dimension != self.dimension:
            raise QuantizationError("Dimension mismatch.")

        quantized_bytes = payload[self.header_size :]

        if len(quantized_bytes) != self.dimension:
            raise QuantizationError("Corrupted payload.")

        quantized = np.frombuffer(
            quantized_bytes,
            dtype=np.uint8,
        )

        reconstructed = (quantized.astype(np.float32) - zero_point) * scale

        return reconstructed
