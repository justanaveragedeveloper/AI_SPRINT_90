"""day59_gqa.py — A small, readable NumPy implementation of Grouped-Query Attention (GQA).

What problem does this solve?
=============================
Standard Multi-Head Attention (MHA) gives every query head its own pair of
Key and Value projections. That works well for quality, but during
autoregressive decoding the KV cache becomes the dominant memory and
bandwidth cost — and it scales with the number of query heads.

Two alternatives shrink that cost:

* Multi-Query Attention (MQA) shares a single K/V head across all query
  heads. Cheapest KV cache, but accuracy can drop in practice.
* Grouped-Query Attention (GQA) splits the query heads into G groups and
  lets each group share one K/V head. It sits between MHA and MQA, and is
  what most modern LLMs use (LLaMA-2, Mistral, PaLM).

This file is a small, self-contained reference that implements GQA in
plain NumPy. It's meant to be read, tested, and modified — not deployed.

How is it organised?
====================
* ``compute_kv_head_mapping`` — a tiny helper that answers "which KV head
  does query head h use?". A pure function, easy to test in isolation.
* ``GroupedQueryAttention`` — a class holding the four projections
  (W_q, W_k, W_v, W_o), a ``forward`` method, and a stable softmax.
* ``_stable_softmax`` — a softmax that doesn't overflow on large logits.

How expensive is it?
====================
Notation: B = batch size, N = sequence length, H = query heads,
G = KV heads, d_k = d_model / H (per-head width).

* Attention arithmetic: O(B * H * N^2 * d_k). Same as MHA — GQA saves
  K/V memory, not query-head compute. (Per query head we form an N x N
  score matrix costing N^2 * d_k, then multiply by V for another N^2 *
  d_k; summing over H heads gives B * H * N^2 * d_k.)
* Plus projection work: O(B * N * d_model^2).
* Peak intermediate memory (asymptotic in B and N): O(B * N * d_model +
  B * N^2). The score matrix is materialised one head at a time, not
  all H at once, so peak storage stays at one (B, N, N) matrix — not
  (B, H, N, N).
* Theoretical KV-cache size: K and V each need N * G * d_k elements per
  layer and per batch, versus N * H * d_k for MHA. Reduction factor:
  H / G. Note this is arithmetic, not measured — this file does not
  implement a persistent KV cache.

What this file is NOT
=====================
To avoid confusion, this reference is specifically not any of these:

* Not a GPU kernel, not Triton, not CUDA, not FlashAttention.
* Not an inference engine — no persistent KV cache, no batching
  scheduler, no beam search.
* Not causally masked. The Day 59 spec does not define mask semantics,
  so we don't invent one. Calling forward() on autoregressive-style
  input will let tokens attend to future positions. That's a deliberate
  scope choice, not a bug.
* Not trainable — no gradients, no optimiser, no loss.
* Not benchmarked. Any throughput claim would require controlled
  measurement.

Numerical behaviour
===================
For finite X of dtype float32 or float64, ``forward`` returns a finite
result of the same shape and dtype — OR raises ``FloatingPointError``
with a message that names the stage where overflow happened. It never
silently returns NaN or Inf.

Every matrix multiply in ``forward`` is wrapped in
``np.errstate(over="raise", invalid="raise")``, so overflow becomes an
explicit exception instead of a silent NaN:

    * Q / K / V projections     -> "Projection overflow..."
    * attention-score product   -> "Attention-score overflow..."
    * attention-value product   -> "Attention-value product overflow..." (defensive)
    * final output projection   -> "Output projection overflow..."

Caveat: whether errstate reliably fires depends on the BLAS backend.
This has been exercised on small matrices; very large matmuls on some
BLAS builds may not honour the FP flags. Treat the contract as
"finite or explicit FloatingPointError under the tested configuration."

Reproducibility
===============
* Default seed is 0. Same seed -> bit-identical weights; different
  seeds -> different weights.
* You can pass either ``seed=<int>`` or ``rng=<numpy.random.Generator>``.
  If you pass a Generator, its state is consumed, so reproducibility
  becomes the caller's responsibility.
* ``forward`` is deterministic: it never consults global ``np.random``
  state.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Module-wide constants
# ---------------------------------------------------------------------------

# The dtypes this implementation accepts. Anything else (integer, float16,
# bfloat16, object) is rejected by the input validator.
_FLOAT_DTYPES = (np.float32, np.float64)

# Small floor for the softmax denominator. Strictly speaking, after the
# max-subtraction trick the softmax denominator is always >= 1, so this
# clamp never actually fires on finite input. It exists to satisfy the
# "always clamp divisions" project rule and to defend against a future
# refactor that removes the max subtraction.
_EPSILON = 1e-8


def _require_positive_int(value: object, name: str) -> int:
    """Validate that ``value`` is a plain positive integer (not a bool).

    Why not just call ``int(value)``? Because ``bool`` is an ``int``
    subclass in Python, so ``np.random.default_rng(True)`` silently
    becomes ``seed=1``. We want that surprise caught early.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} should be a plain integer; received {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(
            f"{name} should be a positive integer (>= 1); received {value}"
        )
    return int(value)


def compute_kv_head_mapping(
    query_head_index: int,
    num_query_heads: int,
    num_kv_heads: int,
) -> int:
    """Return which KV head ``query_head_index`` uses.

    The formula
    -----------
    ``kv(h) = floor(h * G / H)``, valid when ``H % G == 0``.

    We implement it as ``h // (H // G)`` instead of ``int(h * G / H)``
    because Python integers are exact. The float form can round the
    wrong way at very large H (e.g. h * G / H = 12.99999... truncating
    to 12 when the exact value is 13).

    Example: H=8, G=2 gives the mapping [0, 0, 0, 0, 1, 1, 1, 1].

    What gets rejected, and why
    ---------------------------
    * bools — they're int subclasses (``True`` means 1)
    * H <= 0 or G <= 0 — nonsense head counts
    * H not divisible by G — would give unequal group sizes, and the
      spec only defines equal contiguous groups
    * h outside [0, H)
    """
    num_query_heads = _require_positive_int(num_query_heads, "num_query_heads")
    num_kv_heads = _require_positive_int(num_kv_heads, "num_kv_heads")

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"{num_query_heads} query heads can't be split evenly into "
            f"{num_kv_heads} KV groups. Pick a KV-head count that divides "
            f"{num_query_heads} without a remainder."
        )

    if isinstance(query_head_index, bool) or not isinstance(
        query_head_index, (int, np.integer)
    ):
        raise TypeError(
            f"query_head_index should be a plain integer; "
            f"received {type(query_head_index).__name__}"
        )

    head_index = int(query_head_index)
    if not 0 <= head_index < num_query_heads:
        raise ValueError(
            f"query_head_index {head_index} is out of range. "
            f"Valid range is [0, {num_query_heads})."
        )

    # Integer floor-division: exact and safe.
    return head_index // (num_query_heads // num_kv_heads)


def _stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax along the last axis that doesn't overflow on large logits.

    The trick: subtract the row's maximum before exponentiating. Since
    ``exp(x - max) <= exp(0) = 1``, nothing in the numerator can exceed
    1.0, so float32 (whose exp() limit is around 88.7) handles logits
    up to about 1e10 without producing Inf or NaN.

    After the shift, the denominator is at least 1.0 (the argmax
    contributes exp(0) = 1), so the ``_EPSILON`` clamp is defensive
    only and never fires on finite input. It exists because the project
    rules ask every division to have a small epsilon guard.

    Precondition: ``scores`` is finite. The caller checks this before
    calling; NaN or Inf here would defeat the max-subtraction trick.
    """
    row_max = np.max(scores, axis=-1, keepdims=True)
    exp_shifted = np.exp(scores - row_max)
    denominator = np.sum(exp_shifted, axis=-1, keepdims=True)
    return exp_shifted / np.maximum(denominator, scores.dtype.type(_EPSILON))


class GroupedQueryAttention:
    """One-layer Grouped-Query Attention with learned projections.

    Parameters
    ----------
    d_model : int
        Width of the input embeddings. Must be divisible by
        ``num_query_heads`` so every head gets an integer-sized slice.
    num_query_heads : int
        H — how many query heads. Positive integer, not a bool.
    num_kv_heads : int
        G — how many KV heads. ``1 <= G <= H`` and ``H`` must be
        divisible by ``G``.
    seed : int, optional
        Seed for ``numpy.random.default_rng``. Defaults to 0. Bools
        rejected.
    rng : numpy.random.Generator, optional
        Alternatively, hand in a Generator. Its state gets consumed.
        Cannot be combined with ``seed``.

    Public weights
    --------------
    These are intentionally public so tests can swap in controlled
    values (e.g. identities) for hand-computed examples. Every
    ``forward`` call validates their shape, dtype, and finiteness
    before doing math.

    W_q : (d_model, d_model)
    W_k : (d_model, G * d_k) — note the G, not H
    W_v : (d_model, G * d_k) — note the G, not H
    W_o : (d_model, d_model)

    The fact that W_k and W_v are ``G * d_k`` wide, not ``H * d_k``, is
    what separates "real GQA" from "MHA with the K/V heads duplicated
    and relabelled." A test asserts this directly.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        d_model = _require_positive_int(d_model, "d_model")
        num_query_heads = _require_positive_int(num_query_heads, "num_query_heads")
        num_kv_heads = _require_positive_int(num_kv_heads, "num_kv_heads")

        if d_model % num_query_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_query_heads "
                f"({num_query_heads}) so each head has an integer-sized slice."
            )

        # This single check also rejects G > H for positive inputs:
        # if 0 < H < G then H % G == H > 0, so the divisibility check fails.
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"{num_query_heads} query heads can't be split evenly into "
                f"{num_kv_heads} KV groups. Pick a KV-head count that "
                f"divides {num_query_heads} without a remainder."
            )

        # --- RNG setup ---
        if seed is not None and rng is not None:
            raise ValueError(
                "Pass either seed or rng, not both — we need to know which "
                "one initialises the weights."
            )
        if seed is not None and isinstance(seed, bool):
            raise TypeError("seed should be a plain integer, not a bool")

        if rng is None:
            if seed is None:
                seed = 0
            rng = np.random.default_rng(seed)
        elif not isinstance(rng, np.random.Generator):
            raise TypeError(
                f"rng should be a numpy.random.Generator; received {type(rng).__name__}"
            )

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads  # d_k in the spec

        # Xavier-uniform init in float64 for good precision during init.
        # Forward casts to X.dtype so the "output dtype == input dtype"
        # contract holds without silent promotion.
        self.W_q = self._xavier_uniform_init(rng, d_model, d_model)
        self.W_k = self._xavier_uniform_init(rng, d_model, num_kv_heads * self.head_dim)
        self.W_v = self._xavier_uniform_init(rng, d_model, num_kv_heads * self.head_dim)
        self.W_o = self._xavier_uniform_init(rng, d_model, d_model)

        # Precompute "which KV head does query head h use?" as an int64
        # array for O(1) lookup in the forward loop. The mapping is
        # deterministic and independent of the input.
        self._head_to_kv = np.array(
            [
                compute_kv_head_mapping(h, num_query_heads, num_kv_heads)
                for h in range(num_query_heads)
            ],
            dtype=np.int64,
        )

    @staticmethod
    def _xavier_uniform_init(
        rng: np.random.Generator, fan_in: int, fan_out: int
    ) -> np.ndarray:
        """Xavier-uniform: values in [-sqrt(6/(fan_in+fan_out)), +that]."""
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))

    def _check_weights_are_valid(self) -> None:
        """Make sure nobody corrupted the public weights since construction.

        Because W_q, W_k, W_v, W_o are public, a caller could have
        assigned a NaN or a wrong-shaped array. This method catches that
        before it corrupts downstream math. Cost is O(d_model^2) —
        negligible relative to attention.
        """
        expectations = [
            ("W_q", self.W_q, (self.d_model, self.d_model)),
            ("W_k", self.W_k, (self.d_model, self.num_kv_heads * self.head_dim)),
            ("W_v", self.W_v, (self.d_model, self.num_kv_heads * self.head_dim)),
            ("W_o", self.W_o, (self.d_model, self.d_model)),
        ]
        for weight_name, weight, expected_shape in expectations:
            if not isinstance(weight, np.ndarray):
                raise TypeError(
                    f"{weight_name} should be a numpy array; "
                    f"received {type(weight).__name__}"
                )
            if weight.shape != expected_shape:
                raise ValueError(
                    f"{weight_name} shape is {weight.shape}, but expected "
                    f"{expected_shape}."
                )
            if weight.dtype not in _FLOAT_DTYPES:
                raise TypeError(
                    f"{weight_name} dtype {weight.dtype} is not one of {_FLOAT_DTYPES}."
                )
            if not np.all(np.isfinite(weight)):
                raise ValueError(
                    f"{weight_name} contains non-finite values (NaN or Inf)."
                )

    def _check_input_tensor(self, X: np.ndarray) -> None:
        """Validate the incoming activation tensor before touching it."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X should be a numpy array; received {type(X).__name__}")
        if X.ndim != 3:
            raise ValueError(
                f"X should be a 3D array of shape (B, N, d_model); "
                f"received one with {X.ndim} dimension(s)."
            )

        batch_size, seq_len, feat_dim = X.shape
        if batch_size < 1 or seq_len < 1:
            raise ValueError(
                f"X must have at least one batch and one sequence position; "
                f"received B={batch_size}, N={seq_len}."
            )
        if feat_dim != self.d_model:
            raise ValueError(
                f"X's last dimension is {feat_dim}, but d_model is {self.d_model}."
            )
        if X.dtype not in _FLOAT_DTYPES:
            raise TypeError(
                f"X has dtype {X.dtype}, which is not one of {_FLOAT_DTYPES}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X contains non-finite values (NaN or Inf). "
                "Please clean up the input before calling forward()."
            )

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run GQA on ``X`` of shape ``(B, N, d_model)``.

        Returns
        -------
        np.ndarray of shape ``(B, N, d_model)``, dtype == ``X.dtype``.

        For finite X, either the result is finite, or a
        ``FloatingPointError`` is raised with a stage-specific message.
        Never returns silent NaN/Inf under the tested configuration.

        Design note: we loop over query heads in Python rather than
        materialising a ``(B, H, N, d_k)`` K/V tensor via fancy indexing.
        The loop costs a tiny bit of Python overhead but avoids
        physically copying each KV head H/G times — which would inflate
        peak memory and defeat the whole point of GQA inside a
        reference implementation. Only one Python loop exists in this
        method: over query heads. All inner work is vectorised over
        ``(B, N, d_k)``.
        """
        self._check_input_tensor(X)
        self._check_weights_are_valid()

        # Cast weights to the input dtype once. If they already match,
        # astype(copy=False) is a no-op; if not, it's one allocation.
        W_q = self.W_q.astype(X.dtype, copy=False)
        W_k = self.W_k.astype(X.dtype, copy=False)
        W_v = self.W_v.astype(X.dtype, copy=False)
        W_o = self.W_o.astype(X.dtype, copy=False)

        batch_size, seq_len, _ = X.shape
        num_q_heads = self.num_query_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # --- Step 1: project X into Q, K, V -----------------------------
        # Q is full width (num_q_heads * head_dim = d_model). K and V are
        # only num_kv_heads * head_dim wide — that's the GQA memory
        # saving, and it's observable in the weight shapes. Every matmul
        # is guarded so overflow raises instead of producing silent
        # NaN/Inf.
        try:
            with np.errstate(over="raise", invalid="raise"):
                Q = X @ W_q  # (B, N, d_model)
                K = X @ W_k  # (B, N, num_kv_heads * head_dim)
                V = X @ W_v  # (B, N, num_kv_heads * head_dim)
        except FloatingPointError as error:
            max_abs_x = float(np.max(np.abs(X)))
            max_abs_w = max(
                float(np.max(np.abs(W_q))),
                float(np.max(np.abs(W_k))),
                float(np.max(np.abs(W_v))),
            )
            raise FloatingPointError(
                f"Projection overflow during X @ W (dtype={X.dtype}): "
                f"input or weight magnitudes are too large. "
                f"max |X| = {max_abs_x:.3e}, max |W| = {max_abs_w:.3e}."
            ) from error

        # --- Step 2: split each tensor into per-head slices -------------
        # Layout goes from (B, N, heads*d_k) to (B, heads, N, d_k) so
        # each head's (N, d_k) slice can be batched independently.
        Q = Q.reshape(batch_size, seq_len, num_q_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Do the 1/sqrt(d_k) scale in the input dtype. A bare Python
        # float here would silently promote float32 scores to float64,
        # breaking the "output dtype == input dtype" contract.
        scale = X.dtype.type(1.0 / np.sqrt(head_dim))

        # --- Step 3: attention, one query head at a time ----------------
        head_outputs = np.empty(
            (batch_size, num_q_heads, seq_len, head_dim), dtype=X.dtype
        )

        for head_index in range(num_q_heads):
            # Which KV head does this query head share?
            kv_index = int(self._head_to_kv[head_index])

            Q_head = Q[:, head_index, :, :]  # (B, N, d_k)  — Q_h
            K_kv = K[:, kv_index, :, :]  # (B, N, d_k) — view, no copy
            V_kv = V[:, kv_index, :, :]  # (B, N, d_k) — view, no copy

            # Scores: (B, N, d_k) @ (B, d_k, N) -> (B, N, N), then scale.
            try:
                with np.errstate(over="raise", invalid="raise"):
                    scores = (Q_head @ K_kv.swapaxes(-1, -2)) * scale
            except FloatingPointError as error:
                raise FloatingPointError(
                    f"Attention-score overflow at query head {head_index} "
                    f"(using KV head {kv_index}). Projected Q/K values "
                    f"are too large for dtype {X.dtype}."
                ) from error

            attention_weights = _stable_softmax(scores)  # (B, N, N)

            # --- Attention-value product ---
            # Defensive guard. attention_weights is a convex combination
            # (sums to 1, all entries >= 0) of the finite rows of V_kv,
            # so in exact arithmetic this product is bounded by
            # max |V_kv| and cannot overflow. Under realistic
            # float64/BLAS rounding it also cannot overflow in any case
            # we could construct. The guard stays so the "finite or
            # raise" contract is arithmetic-independent; no test in this
            # suite actually triggers it.
            try:
                with np.errstate(over="raise", invalid="raise"):
                    head_outputs[:, head_index, :, :] = attention_weights @ V_kv
            except FloatingPointError as error:
                raise FloatingPointError(
                    f"Attention-value product overflow at query head "
                    f"{head_index} (using KV head {kv_index})."
                ) from error

        # --- Step 4: concatenate heads and project back to d_model ------
        # (B, H, N, d_k) -> (B, N, H, d_k) -> (B, N, H*d_k) == (B, N, d_model)
        concatenated = head_outputs.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_q_heads * head_dim
        )

        # Final output projection, also guarded. Without this guard, a
        # huge W_o could silently overflow here even though every
        # previous stage stayed finite. (This was the exact bug a
        # reviewer caught in v2 of this file.)
        try:
            with np.errstate(over="raise", invalid="raise"):
                return concatenated @ W_o
        except FloatingPointError as error:
            raise FloatingPointError(
                f"Output projection overflow in concat @ W_o. "
                f"max |concat| = "
                f"{float(np.max(np.abs(concatenated))):.3e}, "
                f"max |W_o| = {float(np.max(np.abs(W_o))):.3e}."
            ) from error
