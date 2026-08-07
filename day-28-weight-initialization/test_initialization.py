"""
Pytest suite for Day 28 – Weight Initialization Strategies.

Covers:
  - Xavier/He variance and mean correctness
  - Auto‑mode selection
  - Invalid inputs (modes, dimensions, activations)
  - Deep network stability (no NaN/Inf, no exploding activations)
  - Compatibility with the original MLP API
"""

import math
import os
import random
import statistics
import sys

import pytest

# Add paths to previous days' modules
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../day-24-autograd"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../day-25-nn-from-scratch")
    )
)

from engine import Value  # noqa: I001
from initializers import (
    initialize_weights,
    ScaledMLP,
    ScaledNeuron,
)
from nn import MLP


# -------------------------------------------------------------------
# Helpers for sampling
# -------------------------------------------------------------------


def sample_weights(mode, nin, nouts=None, n_samples=10000):
    """Return a list of n_samples weights from the given initialiser."""
    sampler = initialize_weights(mode, nin=nin, nouts=nouts)
    if sampler is None:
        # default mode: uniform[-1,1] from parent
        return [random.uniform(-1, 1) for _ in range(n_samples)]
    return [sampler() for _ in range(n_samples)]


# -------------------------------------------------------------------
# Tests: Variance & Mean
# -------------------------------------------------------------------


def test_xavier_uniform_variance():
    nin, nout = 1000, 500
    samples = sample_weights("xavier_uniform", nin, nout)
    expected_var = 2.0 / (nin + nout)  # variance of Uniform(-a,a) with a^2/3
    actual_var = statistics.variance(samples)
    assert abs(actual_var - expected_var) < 1e-3


def test_xavier_normal_variance():
    nin, nout = 1000, 500
    samples = sample_weights("xavier_normal", nin, nout)
    expected_var = 2.0 / (nin + nout)
    actual_var = statistics.variance(samples)
    assert abs(actual_var - expected_var) < 1e-3


def test_he_uniform_variance():
    nin = 1000
    samples = sample_weights("he_uniform", nin)
    expected_var = 2.0 / nin
    actual_var = statistics.variance(samples)
    assert abs(actual_var - expected_var) < 1e-3


def test_he_normal_variance():
    nin = 1000
    samples = sample_weights("he_normal", nin)
    expected_var = 2.0 / nin
    actual_var = statistics.variance(samples)
    assert abs(actual_var - expected_var) < 1e-3


def test_xavier_mean_near_zero():
    nin, nout = 1000, 500
    samples = sample_weights("xavier_normal", nin, nout)
    mean = statistics.mean(samples)
    assert abs(mean) < 1e-2


def test_he_mean_near_zero():
    nin = 1000
    samples = sample_weights("he_normal", nin)
    mean = statistics.mean(samples)
    assert abs(mean) < 1e-2


# -------------------------------------------------------------------
# Tests: Auto‑mode selection (implicitly via forward pass stability)
# -------------------------------------------------------------------


def test_auto_mode_relu_uses_he():
    # Create a deep ReLU network with auto mode; forward pass should not explode.
    mlp = ScaledMLP(
        nin=10, nouts=[10] * 10, init_mode="auto", activations=["relu"] * 10
    )
    x = [Value(1.0)] * 10
    out = mlp(x)
    if isinstance(out, list):
        out_vals = [v.value for v in out]
    else:
        out_vals = [out.value]
    assert all(not math.isnan(v) and not math.isinf(v) for v in out_vals)
    # Check that values are not exploding (e.g., < 1000). ReLU can grow, but with He it should be moderate.
    assert max(abs(v) for v in out_vals) < 100.0


def test_auto_mode_tanh_uses_xavier():
    # Similarly for tanh; values should stay in [-1,1] approximately.
    mlp = ScaledMLP(
        nin=10, nouts=[10] * 10, init_mode="auto", activations=["tanh"] * 10
    )
    x = [Value(1.0)] * 10
    out = mlp(x)
    if isinstance(out, list):
        out_vals = [v.value for v in out]
    else:
        out_vals = [out.value]
    assert all(not math.isnan(v) and not math.isinf(v) for v in out_vals)
    # Tanh saturates, so values should be within [-1,1] plus small noise.
    assert all(abs(v) <= 1.1 for v in out_vals)


# -------------------------------------------------------------------
# Tests: Invalid inputs
# -------------------------------------------------------------------


def test_invalid_init_mode():
    with pytest.raises(ValueError, match="Unknown initialisation mode"):
        initialize_weights("invalid_mode", nin=10)


def test_invalid_dimensions_nin_zero():
    with pytest.raises(ValueError, match="nin must be positive"):
        initialize_weights("xavier_uniform", nin=0)


def test_invalid_dimensions_nout_negative():
    with pytest.raises(ValueError, match="nouts must be positive"):
        initialize_weights("xavier_uniform", nin=10, nouts=-5)


def test_invalid_activation():
    with pytest.raises(ValueError, match="Invalid activation"):
        ScaledNeuron(10, activation="invalid_act")


def test_invalid_activation_in_mlp():
    with pytest.raises(ValueError, match="Invalid activation"):
        ScaledMLP(nin=10, nouts=[5], activations=["invalid_act"])


def test_mlp_empty_nouts():
    with pytest.raises(ValueError, match="nouts cannot be empty"):
        ScaledMLP(nin=10, nouts=[])


# -------------------------------------------------------------------
# Tests: Deep network forward stability (no NaN/Inf)
# -------------------------------------------------------------------


def test_deep_network_no_nan_inf():
    mlp = ScaledMLP(
        nin=10, nouts=[10] * 15, init_mode="auto", activations=["relu"] * 14 + ["tanh"]
    )
    x = [Value(1.0)] * 10
    out = mlp(x)
    if isinstance(out, list):
        out_vals = [v.value for v in out]
    else:
        out_vals = [out.value]
    assert all(not math.isnan(v) and not math.isinf(v) for v in out_vals)


# -------------------------------------------------------------------
# Tests: Compatibility with original MLP API
# -------------------------------------------------------------------


def test_compatible_with_mlp_api():
    # ScaledMLP should accept the same arguments as MLP (except init_mode)
    mlp_orig = MLP(10, [5, 3])
    mlp_scaled = ScaledMLP(10, [5, 3], init_mode="xavier_uniform")

    # Both have forward method (__call__)
    x = [Value(1.0)] * 10
    out_orig = mlp_orig(x)
    out_scaled = mlp_scaled(x)
    assert isinstance(out_orig, (Value, list))
    assert isinstance(out_scaled, (Value, list))

    # Both have parameters() method returning list of Value
    params_orig = mlp_orig.parameters()
    params_scaled = mlp_scaled.parameters()
    assert isinstance(params_orig, list)
    assert isinstance(params_scaled, list)
    # The number of parameters should match (same architecture)
    assert len(params_orig) == len(params_scaled)


# -------------------------------------------------------------------
# Additional: ensure no NaN/Inf from initialisation directly
# -------------------------------------------------------------------


def test_initialization_no_nan_inf():
    for mode in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal"]:
        samples = sample_weights(mode, nin=100, nouts=50, n_samples=1000)
        assert all(not math.isnan(s) and not math.isinf(s) for s in samples)


if __name__ == "__main__":
    pytest.main(["-v", __file__])