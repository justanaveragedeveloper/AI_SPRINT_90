"""test_gqa.py — Verification suite for day59_gqa.

What's in here
--------------
32 test functions covering:
  * mapping formula and boundaries
  * constructor validation
  * input / weight validation
  * forward output contract (shape, dtype)
  * attention correctness (hand-computed + reference)
  * numerical stability
  * determinism / RNG policy

How each test is categorised
----------------------------
Every test carries a tag at the top of its docstring:

  [ROBUST]   Catches a specific wrong implementation. The docstring
             names what would have to break for the test to fail.
  [CONTRACT] Verifies a public-interface property (shape, dtype).
  [THEATER]  Happy-path only. None in this file.

Validation tests (input guards, weight guards, seed guards) are
[ROBUST], not [THEATER]: each fails against a concrete wrong
implementation. For example, "constructor silently accepts H % G != 0"
is a specific mutation that test_constructor_rejects_invalid_configuration
rejects.

Balance at last count: 29 ROBUST / 2 CONTRACT / 0 THEATER (functions),
expanding to 51 ROBUST / 2 CONTRACT / 0 THEATER (collected cases via
parametrization).

Where the property tests went
-----------------------------
The PROPERTY block at the bottom of this file documents the Hypothesis
strategies that would run if Hypothesis were installed. It's a design
document, not executed coverage. Hypothesis is not a dependency of this
suite.
"""

import numpy as np
import pytest
from day59_gqa import GroupedQueryAttention, compute_kv_head_mapping


def _make_x(B, N, D, dtype=np.float32, seed=0):
    """Local, seeded random tensor for tests. Never touches global RNG."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, N, D)).astype(dtype)


# ======================================================================
# Mapping tests — "which KV head does query head h use?"
# ======================================================================


def test_kv_mapping_distribution_h8g2():
    """[ROBUST] Guards the KV-head mapping formula.

    Would fail if ``compute_kv_head_mapping`` returned anything other
    than ``floor(h * G / H)`` — e.g. ``h // G``, ``h % G``,
    ``h * H // G``, or a float version that rounds at very large H.
    """
    mapping = [compute_kv_head_mapping(h, 8, 2) for h in range(8)]
    assert mapping == [0, 0, 0, 0, 1, 1, 1, 1]


def test_kv_mapping_boundaries_and_coverage():
    """[ROBUST] Guards that every KV group gets used, and none is starved.

    Would fail if the last query head wrapped to an out-of-range KV
    index, or if one of the KV groups got zero heads (off-by-one in
    ``floor(h * G / H)``).
    """
    H, G = 8, 4
    mapping = [compute_kv_head_mapping(h, H, G) for h in range(H)]
    assert mapping[0] == 0
    assert mapping[H - 1] == G - 1
    counts = np.bincount(mapping, minlength=G)
    assert counts.tolist() == [H // G] * G


def test_mha_identity_mapping():
    """[ROBUST] Guards the MHA boundary (G == H).

    Would fail if the mapping always collapsed to 0 regardless of G —
    i.e. an implementation that secretly implements MQA even when
    MHA was asked for.
    """
    H = 4
    mapping = [compute_kv_head_mapping(h, H, H) for h in range(H)]
    assert mapping == list(range(H))


def test_mqa_all_zero_mapping():
    """[ROBUST] Guards the MQA boundary (G == 1).

    Would fail if the mapping let ``h`` leak into the KV index when
    there's only one KV head to choose from.
    """
    H = 8
    mapping = [compute_kv_head_mapping(h, H, 1) for h in range(H)]
    assert mapping == [0] * H


@pytest.mark.parametrize(
    "h,H,G",
    [
        (0, 0, 1),
        (0, 8, 0),
        (0, -1, 2),
        (0, 8, -1),
        (0, 4, 8),
        (0, 8, 3),
        (-1, 8, 2),
        (8, 8, 2),
        (100, 8, 2),
    ],
)
def test_mapping_rejects_invalid_arguments(h, H, G):
    """[ROBUST] Guards the input checks on the mapping helper.

    Would fail if any documented precondition were skipped —
    zero or negative counts, ``G > H``, ``H % G != 0``, or
    ``h`` outside ``[0, H)``. Each parametrised case is a
    concrete wrong input the validator must reject.
    """
    with pytest.raises((ValueError, TypeError)):
        compute_kv_head_mapping(h, H, G)


@pytest.mark.parametrize("bad", [True, False])
def test_mapping_rejects_bool_as_int(bad):
    """[ROBUST] Guards against ``bool`` sneaking through integer validation.

    Would fail if a bool were treated as an integer. Python's ``bool``
    is an ``int`` subclass, so ``np.random`` / integer arithmetic would
    silently accept it as 0 or 1.
    """
    with pytest.raises(TypeError):
        compute_kv_head_mapping(0, bad, 2)


# ======================================================================
# Constructor / weight / forward-input validation
# ======================================================================


@pytest.mark.parametrize(
    "d_model,H,G,exc",
    [
        (10, 4, 2, ValueError),
        (8, 2, 4, ValueError),
        (12, 6, 4, ValueError),
        (True, 2, 1, TypeError),
        (0, 2, 1, ValueError),
        (8, 2, 0, ValueError),
    ],
)
def test_constructor_rejects_invalid_configuration(d_model, H, G, exc):
    """[ROBUST] Guards against building a broken configuration.

    Would fail if the constructor accepted a shape that would break the
    equal-sized contiguous grouping or produce a non-integer head
    width. The ``(8, 2, 4)`` case (``G > H``) is rejected by the
    ``H % G != 0`` branch, not by a separate ``G > H`` check.
    """
    with pytest.raises(exc):
        GroupedQueryAttention(d_model=d_model, num_query_heads=H, num_kv_heads=G)


def test_constructor_rejects_bool_seed():
    """[ROBUST] Guards against ``seed=True`` being silently accepted.

    Would fail if a bool seed slipped through. ``bool`` is an ``int``
    subclass, so ``np.random.default_rng(True)`` silently treats it
    as 1 — making ``seed=True`` and ``seed=1`` indistinguishable. That
    is a contract violation given the ``int`` type annotation.
    """
    with pytest.raises(TypeError):
        GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=True)


def test_seed_and_rng_mutually_exclusive():
    """[ROBUST] Guards against ambiguous RNG setup.

    Would fail if both ``seed`` and ``rng`` were silently accepted.
    Passing both leaves ambiguous which source initialised the
    weights.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            d_model=8, num_query_heads=2, num_kv_heads=1, seed=1, rng=rng
        )


@pytest.mark.parametrize(
    "X",
    [
        np.zeros((4, 8), np.float32),
        np.zeros((1, 1, 4, 8), np.float32),
        np.zeros((1, 4, 4), np.float32),
        np.zeros((0, 4, 8), np.float32),
        np.zeros((2, 0, 8), np.float32),
        np.zeros((1, 4, 8), np.int32),
    ],
)
def test_forward_rejects_malformed_input(X):
    """[ROBUST] Guards the input-tensor validator in ``forward``.

    Would fail if any of the documented shape/dtype rejections were
    skipped. Each malformed input would silently propagate garbage
    into the attention math.
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
    with pytest.raises((ValueError, TypeError)):
        gqa.forward(X)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_forward_rejects_non_finite(bad_value):
    """[ROBUST] Guards against non-finite values reaching the public API.

    Would fail if a NaN or ±Inf slipped through the input validator.
    Non-finite values silently propagate through matmul and corrupt
    everything downstream.
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
    X = np.zeros((1, 4, 8), np.float32)
    X[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        gqa.forward(X)


def test_weight_nan_rejected():
    """[ROBUST] Guards against a NaN sneaking into a public weight.

    Would fail if a NaN injected post-construction propagated to the
    output without an exception. Before weight validation was added,
    this returned NaN output with no error — a fail-silent bug.
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
    gqa.W_q[0, 0] = np.nan
    X = _make_x(1, 4, 8)
    with pytest.raises(ValueError, match="W_q"):
        gqa.forward(X)


def test_weight_wrong_shape_rejected():
    """[ROBUST] Guards against a wrong-shaped weight being used silently.

    Would fail if a shape-mismatched weight produced an opaque numpy
    broadcast or matmul error instead of a clear ValueError naming the
    offending attribute.

    Config: d_model=8, H=2, G=2 -> head_dim=4 -> W_k expected shape
    (8, 8). We set W_k to (8, 4), which is the shape that would be
    correct if G were silently treated as 1. This catches the mutation
    "K/V projection sized from G=1 regardless of the configured G".
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=2)
    gqa.W_k = np.zeros((8, 4), dtype=np.float64)  # expected (8, 8)
    X = _make_x(1, 4, 8)
    with pytest.raises(ValueError, match="W_k shape"):
        gqa.forward(X)


def test_weight_non_float_dtype_rejected():
    """[ROBUST] Guards against non-float weights.

    Would fail if an integer weight silently truncated during the
    projection matmul (numpy would downcast without complaint).
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
    gqa.W_o = np.eye(8, dtype=np.int64)
    X = _make_x(1, 4, 8)
    with pytest.raises(TypeError, match="W_o dtype"):
        gqa.forward(X)


# ======================================================================
# Forward contract — shape and dtype
# ======================================================================


def test_forward_output_shape_matches_input():
    """[CONTRACT] Verifies the output shape matches the input shape.

    Would fail if the public forward pass returned a different shape
    than (B, N, d_model) — a break in the output contract.
    """
    B, N, D, H, G = 2, 16, 32, 4, 2
    X = _make_x(B, N, D)
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)
    assert gqa.forward(X).shape == (B, N, D)


def test_forward_output_dtype_matches_input():
    """[CONTRACT] Verifies output dtype matches input dtype.

    Would fail if a float32 input produced a float64 output, silently
    doubling the downstream layer's buffer footprint.
    """
    for dt in (np.float32, np.float64):
        X = _make_x(1, 4, 8, dtype=dt)
        gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
        assert gqa.forward(X).dtype == dt


# ======================================================================
# Forward robustness — attention correctness, stability, non-mutation
# ======================================================================


def test_forward_does_not_mutate_input():
    """[ROBUST] Guards against in-place modification of the caller's input.

    Would fail if ``forward`` overwrote ``X`` (e.g. via ``X -= mean``,
    ``X *= scale``, or ``np.add(X, ..., out=X)``). Caller-owned buffers
    must be preserved.
    """
    X = _make_x(2, 5, 8)
    X_copy = X.copy()
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1)
    gqa.forward(X)
    assert np.array_equal(X, X_copy)


def test_kv_projection_uses_g_heads_not_h():
    """[ROBUST] Guards the "real GQA, not cosmetic GQA" invariant.

    Would fail if W_k / W_v secretly had shape (d_model, H*d_k) — i.e.
    if the implementation allocated independent K/V heads for every
    query head and merely labelled them "shared". This is the central
    architectural invariant, and it must be observable in the parameter
    shapes, not just in a docstring.
    """
    D, H, G = 32, 8, 2
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)
    head_dim = D // H
    assert gqa.W_k.shape == (D, G * head_dim)
    assert gqa.W_v.shape == (D, G * head_dim)
    assert gqa.W_q.shape == (D, D)
    assert gqa.W_o.shape == (D, D)


def test_forward_matches_manual_gqa_reference():
    """[ROBUST] Compares forward() to a from-scratch reference.

    Would fail if ``forward`` diverged from a manually-written
    reference that explicitly applies the mapped KV head. This test
    shares ``compute_kv_head_mapping`` with the implementation, so
    there is a small shared-assumption risk — that risk is closed by
    ``test_hand_computed_gqa_h4g2_distinct_kv``, which uses hardcoded
    expected values and no shared helper.
    """
    D, H, G, B, N = 8, 4, 2, 1, 3
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G, seed=1)
    X = _make_x(B, N, D, seed=42)

    head_dim = D // H
    Q = (X @ gqa.W_q.astype(X.dtype)).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
    K = (X @ gqa.W_k.astype(X.dtype)).reshape(B, N, G, head_dim).transpose(0, 2, 1, 3)
    V = (X @ gqa.W_v.astype(X.dtype)).reshape(B, N, G, head_dim).transpose(0, 2, 1, 3)
    mapping = [compute_kv_head_mapping(h, H, G) for h in range(H)]
    scale = X.dtype.type(1.0 / np.sqrt(head_dim))

    per_head_outputs = []
    for h in range(H):
        kv = mapping[h]
        scores = (Q[:, h] @ K[:, kv].swapaxes(-1, -2)) * scale
        row_max = scores.max(axis=-1, keepdims=True)
        p = np.exp(scores - row_max)
        p = p / p.sum(axis=-1, keepdims=True)
        per_head_outputs.append(p @ V[:, kv])
    concat = np.stack(per_head_outputs, axis=1).transpose(0, 2, 1, 3).reshape(B, N, D)
    expected = concat @ gqa.W_o.astype(X.dtype)

    np.testing.assert_allclose(gqa.forward(X), expected, atol=1e-6, rtol=1e-6)


def test_hand_computed_attention_mqa_g1():
    """[ROBUST] Hand-computes MQA (G=1) attention, byte for byte.

    Would fail if the 1/sqrt(d_k) scale were missing, if K/V were read
    from the wrong place (in MQA both heads share KV 0), or if head
    concatenation order were swapped (here out_0 != out_1, so a swap is
    visible).

    Note: the score matrix in this test is symmetric, so it does NOT
    distinguish softmax axis. ``test_hand_computed_attention_gqa_asymmetric``
    covers that axis specifically.
    """
    D, H, G, B, N = 4, 2, 1, 1, 2  # noqa: F841
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)
    gqa.W_q = np.eye(4, dtype=np.float64)
    gqa.W_k = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
    gqa.W_v = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
    gqa.W_o = np.eye(4, dtype=np.float64)

    X = np.array([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=np.float64)
    t = 1.0 / np.sqrt(2.0)
    a = np.exp(t) / (np.exp(t) + 1.0)
    expected = np.array(
        [[[a, 1.0 - a, 0.5, 0.5], [1.0 - a, a, 0.5, 0.5]]], dtype=np.float64
    )
    np.testing.assert_allclose(gqa.forward(X), expected, atol=1e-6)


def test_hand_computed_attention_gqa_asymmetric():
    """[ROBUST] Hand-computes H=G=2 attention with asymmetric scores.

    Would fail if softmax ran over axis=-2 instead of -1 (row-softmax
    and column-softmax differ for asymmetric scores), if the
    1/sqrt(d_k) scale were missing, if the wrong V head were used, or
    if head concatenation order were swapped.

    Because H == G, every query head has its own KV head, and K/V for
    the two groups are distinct (W=I and X is asymmetric).
    """
    D, H, G, B, N = 4, 2, 2, 1, 2  # noqa: F841
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)
    gqa.W_q = np.eye(D, dtype=np.float64)
    gqa.W_k = np.eye(D, dtype=np.float64)
    gqa.W_v = np.eye(D, dtype=np.float64)
    gqa.W_o = np.eye(D, dtype=np.float64)

    X = np.array([[[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]]], dtype=np.float64)

    s = 1.0 / np.sqrt(2.0)

    def _softmax_row(v):
        v_max = v.max()
        v_exp = np.exp(v - v_max)
        return v_exp / v_exp.sum()

    Q0 = K0 = V0 = np.array([[1.0, 1.0], [1.0, 0.0]])
    Q1 = K1 = V1 = np.array([[1.0, 0.0], [1.0, 1.0]])

    sm0 = np.array([_softmax_row(r) for r in (Q0 @ K0.T) * s])
    out0 = sm0 @ V0
    sm1 = np.array([_softmax_row(r) for r in (Q1 @ K1.T) * s])
    out1 = sm1 @ V1

    expected = np.concatenate([out0, out1], axis=-1)[None, :, :]
    np.testing.assert_allclose(gqa.forward(X), expected, atol=1e-6)


def test_hand_computed_gqa_h4g2_distinct_kv():
    """[ROBUST] Hand-computes a GQA case where each group has distinct K/V.

    Would fail if any query head read from the wrong KV group. This
    test is fully independent of ``compute_kv_head_mapping`` — the
    expected output is hardcoded below, derived by hand. H=4, G=2, so
    two query heads share KV group 0 and two share KV group 1. Distinct
    K/V per group makes a wrong group selection detectable.

    Setup: d_model=8, H=4, G=2, head_dim=2. W_q=I_8, W_o=I_8.
    W_k and W_v build K[g=0] from X[0:2] and K[g=1] from X[4:6].
    X[0,0] = [1, 0, 0, 0, 0, 0, 0, 0]
    X[0,1] = [0, 0, 0, 0, 1, 0, 0, 0]
    """
    D, H, G = 8, 4, 2
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)

    W_k = np.zeros((D, G * (D // H)), dtype=np.float64)
    W_k[0:2, 0:2] = np.eye(2)  # X[0:2] feeds KV group 0
    W_k[4:6, 2:4] = np.eye(2)  # X[4:6] feeds KV group 1
    gqa.W_q = np.eye(D, dtype=np.float64)
    gqa.W_k = W_k
    gqa.W_v = W_k.copy()
    gqa.W_o = np.eye(D, dtype=np.float64)

    X = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    t = 1.0 / np.sqrt(2.0)
    a = np.exp(t) / (np.exp(t) + 1.0)

    # Hand-derived expected output:
    # Head 0 (group 0): Q_0=[[1,0],[0,0]], K_0=[[1,0],[0,0]]
    #   row 0 scores [t, 0] -> softmax [a, 1-a] -> out [a, 0]
    #   row 1 scores [0, 0] -> [0.5, 0.5] -> out [0.5, 0]
    # Head 1 (group 0): Q_1 = 0 -> uniform -> out [0.5, 0]
    # Head 2 (group 1): Q_2=[[0,0],[1,0]], K_1=[[0,0],[1,0]]
    #   row 0 scores [0, 0] -> [0.5, 0.5] -> out [0.5, 0]
    #   row 1 scores [0, t] -> [1-a, a] -> out [a, 0]
    # Head 3 (group 1): Q_3 = 0 -> uniform -> out [0.5, 0]
    expected = np.array(
        [
            [
                [a, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
                [0.5, 0.0, 0.5, 0.0, a, 0.0, 0.5, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(gqa.forward(X), expected, atol=1e-6)


def test_output_projection_w_o_is_applied():
    """[ROBUST] Guards that the final output projection is actually applied.

    Would fail if ``forward`` returned the concatenated heads without
    multiplying by W_o. Detected by scaling W_o by 3 and checking the
    output scales by 3 — if W_o were being ignored, the output would
    not change.
    """
    D, H, G, B, N = 8, 2, 2, 1, 3
    X = _make_x(B, N, D)
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G)

    gqa.W_o = np.eye(D, dtype=gqa.W_o.dtype)
    y_identity = gqa.forward(X)

    gqa.W_o = 3.0 * np.eye(D, dtype=gqa.W_o.dtype)
    y_tripled = gqa.forward(X)

    np.testing.assert_allclose(y_tripled, 3.0 * y_identity, atol=1e-6, rtol=1e-6)


def test_softmax_stable_under_large_logits():
    """[ROBUST] Guards softmax numerical stability.

    Would fail if softmax used the naive ``exp(x) / sum(exp(x))``
    without max subtraction. With 1e7-magnitude logits, float32 exp
    overflows (its limit is around exp(88.7)), producing Inf then NaN.
    """
    D, H, G = 8, 2, 1
    gqa = GroupedQueryAttention(d_model=D, num_query_heads=H, num_kv_heads=G, seed=0)
    gqa.W_q = gqa.W_q * 1e3
    gqa.W_k = gqa.W_k * 1e3
    X = np.ones((1, 4, D), dtype=np.float32)
    y = gqa.forward(X)
    assert np.all(np.isfinite(y))


def test_output_either_finite_or_raises_on_overflow():
    """[ROBUST] Guards the "finite or raise" numerical contract.

    Would fail if ``forward`` silently returned NaN/Inf for finite
    input (the pre-fix behaviour), OR raised a wrong exception type on
    overflow. Documented contract: finite output for moderate inputs;
    ``FloatingPointError`` for overflow; never silent NaN/Inf.
    """
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=5)

    # Moderate scales: finite output guaranteed.
    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        X = _make_x(2, 4, 8, seed=11) * np.float32(scale)
        y = gqa.forward(X)
        assert np.all(np.isfinite(y)), f"non-finite output at scale={scale}"

    # Extreme scale: raises FloatingPointError, does NOT return NaN.
    X_big = np.full((1, 2, 8), 1e20, dtype=np.float32)
    with pytest.raises(FloatingPointError):
        gqa.forward(X_big)


def test_output_projection_overflow_raises():
    """[ROBUST] Guards the final ``concat @ W_o`` overflow check.

    Would fail if the last matmul in ``forward`` were unguarded.
    Without the guard, finite inputs with finite but large weights
    could return Inf silently — a contract violation.

    This is the exact reviewer-supplied counterexample from the v2
    review: 2 * 1.7e308 sums past float64 max (~1.798e308), producing
    Inf. Inputs and weights are all finite, so the failure is solely
    due to the final matmul being unguarded.
    """
    gqa = GroupedQueryAttention(d_model=4, num_query_heads=1, num_kv_heads=1, seed=0)
    # W_q, W_k = 0 -> uniform attention over V. W_v = I -> concatenated
    # rows equal [1, 1, 0, 0] (sum of X rows / 2). W_o at 1.7e308 ->
    # the two nonzero contributions sum to 3.4e308 > float64 max.
    gqa.W_q = np.zeros((4, 4), dtype=np.float64)
    gqa.W_k = np.zeros((4, 4), dtype=np.float64)
    gqa.W_v = np.eye(4, dtype=np.float64)
    gqa.W_o = np.full((4, 4), 1.7e308, dtype=np.float64)
    X = np.array(
        [[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]],
        dtype=np.float64,
    )

    # Precondition: all operands are finite. If this assertion fails,
    # the real regression (unguarded matmul) is being masked by a
    # rejected-input error instead.
    assert np.all(np.isfinite(X))
    assert all(np.all(np.isfinite(w)) for w in (gqa.W_q, gqa.W_k, gqa.W_v, gqa.W_o))

    with pytest.raises(FloatingPointError, match="Output projection"):
        gqa.forward(X)


def test_single_head_single_token():
    """[ROBUST] Guards the smallest possible configuration.

    Would fail if H=1, G=1, B=1, N=1 broke (e.g. index error during
    reshape, or wrong concat when H*d_k == d_model trivially). Edge
    cases are where off-by-ones live.
    """
    gqa = GroupedQueryAttention(d_model=4, num_query_heads=1, num_kv_heads=1, seed=0)
    X = _make_x(1, 1, 4)
    y = gqa.forward(X)
    assert y.shape == (1, 1, 4)
    assert np.all(np.isfinite(y))


# ======================================================================
# Determinism / RNG policy
# ======================================================================


def test_same_seed_reproducible():
    """[ROBUST] Guards that a given seed always produces the same weights.

    Would fail if init used global ``np.random`` (non-deterministic
    across processes) or ignored the seed parameter.
    """
    a = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=123)
    b = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=123)
    for weight_name in ("W_q", "W_k", "W_v", "W_o"):
        assert np.array_equal(getattr(a, weight_name), getattr(b, weight_name)), (
            f"{weight_name} is not seed-reproducible"
        )


def test_different_seeds_differ():
    """[ROBUST] Guards that the seed parameter actually affects the weights.

    Would fail if ``seed`` were accepted but silently ignored — in
    which case all inits would be identical regardless of seed.
    """
    a = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=1)
    b = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=2)
    assert any(
        not np.array_equal(getattr(a, weight_name), getattr(b, weight_name))
        for weight_name in ("W_q", "W_k", "W_v", "W_o")
    )


def test_global_numpy_random_state_unchanged():
    """[ROBUST] Guards that construction/forward never touches global RNG.

    Would fail if either construction or ``forward`` advanced the
    process-global ``np.random`` state. A deterministic local generator
    must not mutate global state — otherwise unrelated tests or
    consumers see non-reproducible behaviour after touching this module.
    """
    state_before = np.random.get_state()
    gqa = GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, seed=42)
    gqa.forward(_make_x(1, 4, 8))
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_rng_generator_state_advances():
    """[ROBUST] Guards that a supplied Generator is actually consumed.

    Would fail if a caller's ``Generator`` were silently ignored. If
    its state is unchanged after construction, either the generator
    was ignored or a global one was used instead — both break
    reproducibility guarantees the caller is entitled to reason about.
    """
    rng = np.random.default_rng(seed=7)
    state_before = rng.bit_generator.state
    GroupedQueryAttention(d_model=8, num_query_heads=2, num_kv_heads=1, rng=rng)
    state_after = rng.bit_generator.state

    assert state_before != state_after


def test_equivalent_generator_states_produce_identical_weights():
    """[ROBUST] Guards that init reads the Generator's state directly.

    Would fail if init re-seeded internally or fell back to global RNG.
    Two independent generators in the same initial state must produce
    identical weights.
    """
    a = GroupedQueryAttention(
        d_model=8,
        num_query_heads=2,
        num_kv_heads=1,
        rng=np.random.default_rng(seed=99),
    )
    b = GroupedQueryAttention(
        d_model=8,
        num_query_heads=2,
        num_kv_heads=1,
        rng=np.random.default_rng(seed=99),
    )
    for weight_name in ("W_q", "W_k", "W_v", "W_o"):
        assert np.array_equal(getattr(a, weight_name), getattr(b, weight_name))


# ----------------------------------------------------------------------
# Not tested (and why)
# ----------------------------------------------------------------------
# - Causal masking: the Day 59 spec doesn't define mask semantics, so we
#   deliberately don't invent a mask API. Calling forward() on
#   autoregressive-style input will leak future tokens to the past.
# - float16 / bfloat16: outside the documented dtype contract (float32
#   and float64 only).
# - Persistent KV-cache reuse across decode steps: different invariants
#   (per-position append, eviction, block scheduling); out of scope.
# - Performance / throughput: no benchmark is run, so no claim is made.
# - Gradient correctness: no backward pass is implemented.
# - Numerical agreement with a trained model: weights are random, not
#   trained.
# - Concurrency / thread safety: not claimed.
# - Physical K/V repetition: not observable without a memory profiler.
#   The implementation slices K[:, kv] (a view, not a copy), so no
#   repetition occurs. The module docstring explicitly disclaims
#   runtime KV-cache memory reduction as a runtime feature.
# - Attention-value product overflow: the AV product is a convex
#   combination of finite V_kv rows, so it is bounded by max |V_kv| in
#   exact arithmetic. The guard is defensive; its reachability under
#   realistic float64/BLAS rounding is not established by any test, and
#   no constructible counterexample was found. The guard remains so the
#   "finite or raise" contract is arithmetic-independent, not because a
#   failing case exists in the suite.
# - BLAS errstate reliability: NumPy's errstate sets FP flags, but
#   whether BLAS vendors honour them for intermediate matmul operations
#   is not verified for large matrices. Documented as a limitation.
# - Hypothesis property tests: the PROPERTY block below describes
#   strategies that would run if Hypothesis were installed. They are not
#   executed in this suite; do not read the block as coverage.
#
# PROPERTY (Hypothesis strategies — document only, not executed)
# ---------------------------------------------------------------
#
#   Mapping property
#   ----------------
#   Strategy: integers(H, min=1, max=64);
#             divisors(G) with G | H;
#             integers(h, min=0, max=H-1).
#   Assert: 0 <= compute_kv_head_mapping(h, H, G) < G;
#           the sequence is non-decreasing in h;
#           every integer in [0, G) appears exactly H/G times.
#
#   Forward property
#   ----------------
#   Strategy: integers(H, min=1, max=8); integers(G) with G | H;
#             integers(B, min=1, max=4); integers(N, min=1, max=8);
#             integers(d_model) with d_model % H == 0, max d_model 64;
#             floats(shape=(B, N, d_model), allow_nan=False,
#                    allow_infinity=False, dtype=float32|float64,
#                    min_value=-1e3, max_value=1e3).
#   Assert: forward(X).shape == X.shape;
#           forward(X).dtype == X.dtype;
#           np.all(np.isfinite(forward(X))).
# ----------------------------------------------------------------------
