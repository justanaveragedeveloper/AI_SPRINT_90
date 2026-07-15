import numpy as np
import pytest

from day12_quantizer import (
    QuantizationError,
    QuantizedVectorBroker,
)


def test_compress_returns_bytes():
    broker = QuantizedVectorBroker(128)

    vector = np.random.uniform(
        -1,
        1,
        size=128,
    ).astype(np.float32)

    compressed = broker.compress(vector)

    assert isinstance(compressed, bytes)


def test_reconstruction_error_is_small():
    broker = QuantizedVectorBroker(128)

    vector = np.random.uniform(
        -2,
        3,
        size=128,
    ).astype(np.float32)

    compressed = broker.compress(vector)
    recovered = broker.decompress(compressed)

    mae = np.mean(np.abs(vector - recovered))

    assert mae < 0.05


def test_dimension_mismatch():
    broker = QuantizedVectorBroker(64)

    vector = np.zeros(32, dtype=np.float32)

    with pytest.raises(QuantizationError):
        broker.compress(vector)


def test_nan_rejected():
    broker = QuantizedVectorBroker(16)

    vector = np.zeros(16, dtype=np.float32)
    vector[0] = np.nan

    with pytest.raises(QuantizationError):
        broker.compress(vector)


def test_inf_rejected():
    broker = QuantizedVectorBroker(16)

    vector = np.zeros(16, dtype=np.float32)
    vector[0] = np.inf

    with pytest.raises(QuantizationError):
        broker.compress(vector)


def test_corrupted_payload():
    broker = QuantizedVectorBroker(8)

    vector = np.random.rand(8).astype(np.float32)

    compressed = bytearray(broker.compress(vector))

    compressed.pop()

    with pytest.raises(QuantizationError):
        broker.decompress(bytes(compressed))


def test_uniform_vector():
    broker = QuantizedVectorBroker(32)

    vector = np.full(
        32,
        5.0,
        dtype=np.float32,
    )

    compressed = broker.compress(vector)
    recovered = broker.decompress(compressed)

    assert recovered.shape == vector.shape


def test_non_1d_input():
    broker = QuantizedVectorBroker(4)

    vector = np.zeros(
        (2, 2),
        dtype=np.float32,
    )

    with pytest.raises(QuantizationError):
        broker.compress(vector)
