"""Top-2 Mixture-of-Experts (MoE) router and auxiliary load-balancing loss.

WHAT THIS SOLVES
----------------
Implements the routing half of a sparse MoE layer:
  1. A linear gating projection  H = X @ W_g        (N, E)
  2. Numerically stable per-token softmax over experts
  3. Deterministic top-2 expert selection (ties -> lower index)
  4. Renormalization of the two selected gate weights so each row sums to 1
  5. The auxiliary load-balancing loss
        f_j   = (1/N) * count_i[ j in top2_indices[i] ]
        P_j   = (1/N) * sum_i probs[i, j]
        L_aux = alpha * E * sum_j f_j * P_j

WHAT THIS IS NOT (honest limitations)
-------------------------------------
- Does NOT implement sparse expert MLPs, token dispatch, or output
  combination y_i = sum_j g_ij * Expert_j(x_i). No expert network is
  defined or executed here.
- Does NOT implement capacity factors, token dropping, or all-to-all
  communication.
- Does NOT prove L_aux alone prevents expert collapse.
- Does NOT differentiate through hard top-2 selection. The differentiable
  path is via `probs`; forward value only.
- Does NOT demonstrate distributed MoE throughput.

COMPLEXITY (derived from the implementation)
-------------------------------------------
route():
  - H = X @ W_g              O(N * d_model * E)   dense matmul
  - _stable_softmax          O(N * E)
  - np.argsort(-logits)      O(N * E * log E)     stable full sort
  - gather + renormalize     O(N)
  - Total time               O(N * d_model * E + N * E * log E)
  - Memory (new arrays)      logits O(N*E), probs O(N*E), order O(N*E),
                             top2_indices O(N), top2_weights O(N)
                             -> O(N*E)

Top-2 uses FULL stable argsort, not argpartition. argpartition does not
specify tie-breaking for equal keys, so the policy would be
unspecified. The O(log E) factor is the price of determinism. NOT
BENCHMARKED: no measurement of the argsort-vs-argpartition constant.

REPRODUCIBILITY
---------------
`np.random.default_rng(seed)` (PCG64) is used only to initialize W_g.
`route()` is fully deterministic: no randomness, no hashing, no set
iteration; ties break to the lower expert index via stable argsort.
Default seed=0; pass seed=None for non-deterministic init.

PUBLIC MUTABLE STATE (documented boundary)
------------------------------------------
`W_g` is exposed so tests can install a controlled matrix. It is not
frozen. `route()` validates it on every call in `_check_weights_shape`:
the object must be a numpy.ndarray, have a floating dtype, and have
shape (d_model, num_experts); any violation raises RuntimeError before
the matmul. Non-finite values are caught by the logit finiteness guard
(they propagate through the matmul). These checks are O(1) shape/dtype
comparisons, not O(d*E) data scans.

FLOATING-POINT WARNING POLICY
-----------------------------
Both the matmul (`X @ W_g`) and the softmax subtraction
(`logits - row_max`) can overflow for extreme finite inputs. We wrap
both in `np.errstate(...)` so the expected range events do not emit
`RuntimeWarning`s that would obscure real numerical faults. The public
API still rejects non-finite results explicitly (`RuntimeError`), so
the warning suppression is not a silent failure.

API NOTE (documented change from the starter spec)
--------------------------------------------------
The starter spec returned a 2-tuple (top2_indices, top2_weights). This
implementation returns RoutingResult(top2_indices, top2_weights, probs)
so compute_load_balancing_loss can consume the full probability matrix
without recomputation or hidden mutable state on the router.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_EPS: float = 1e-8  # denominator floor for top-2 renormalization
_ROW_SUM_TOL: float = 1e-5  # row-sum tolerance for loss `probs` input


def _coerce_int(v, name: str, min_value: int) -> int:
    """Validate an integer config parameter, rejecting bool (⊂ int)."""
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a positive integer >= {min_value}, not a bool: got {v!r}"
        )
    if not isinstance(v, (int, np.integer)) or int(v) < min_value:
        raise ValueError(f"{name} must be a positive integer >= {min_value}, got {v!r}")
    return int(v)


class RoutingResult(NamedTuple):
    """Output of :meth:`Top2MoERouter.route`.

    All three arrays are NEW allocations; none is a view of an internal
    router buffer or of the caller's input `X`.

    top2_indices : (N, 2) int64
        Selected expert indices. Column 0 holds the larger logit; equal
        logits break to the LOWER index.
    top2_weights : (N, 2) float64
        Gate weights for the selected experts, renormalized per row to
        sum to 1. `top2_weights[i, j]` corresponds to
        `top2_indices[i, j]`, NOT to `probs[i, j]`.
    probs : (N, E) float64
        Full softmax probability matrix. Rows sum to 1.
    """

    top2_indices: np.ndarray
    top2_weights: np.ndarray
    probs: np.ndarray


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-subtraction.

    logits : (N, E) finite float64.
    Returns (N, E) float64 whose rows sum to 1 up to floating-point error.

    Overflow handling:
      When a logit is near -MAX_FLOAT and row_max is near +MAX_FLOAT,
      the subtraction overflows to -inf. exp(-inf)=0 is the correct
      underflow (that expert's probability is exactly zero). We suppress
      the warning with `np.errstate` so genuine numerical faults are not
      hidden among expected range events.
    """
    row_max = np.max(logits, axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = logits - row_max  # <= 0; may be -inf on overflow
        exp_shifted = np.exp(shifted)  # in [0, 1]
        return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


class Top2MoERouter:
    """Top-2 MoE gating router.

    Attributes
    ----------
    d_model : int
    num_experts : int
    seed : int | None
    W_g : np.ndarray, (d_model, num_experts), float64
        Gating projection. Public so tests can install a controlled
        matrix; treat as read-only in normal use. `route()` validates
        its type, dtype, and shape on every call. See "PUBLIC MUTABLE
        STATE" in the module docstring.
    """

    def __init__(self, d_model: int, num_experts: int, seed: int | None = 0) -> None:
        self.d_model = _coerce_int(d_model, "d_model", 1)
        self.num_experts = _coerce_int(num_experts, "num_experts", 2)
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Xavier-style init keeps pre-softmax logits O(1) at initialization,
        # avoiding the degenerate one-hot regime.
        scale = 1.0 / np.sqrt(self.d_model)
        self.W_g = rng.standard_normal((self.d_model, self.num_experts)) * scale

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy.ndarray, got {type(X).__name__}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, d_model), got ndim={X.ndim}")
        N, d = X.shape
        if N == 0:
            raise ValueError("X must have at least one token (N >= 1)")
        if d != self.d_model:
            raise ValueError(f"X feature count {d} != router d_model {self.d_model}")
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError(f"X must have a floating dtype, got {X.dtype}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf")
        return X.astype(np.float64, copy=False)

    def _check_weights_shape(self) -> None:
        """Validate the publicly mutable W_g before routing.

        Guards three corruptions that would otherwise surface as opaque
        numpy errors: non-ndarray (list/tuple), non-floating dtype
        (object/int), and wrong shape. Each is O(1) metadata access, no
        data scan. Non-finite values are caught later by the logit
        finiteness guard because they propagate through the matmul.
        """
        if not isinstance(self.W_g, np.ndarray):
            raise TypeError(
                "W_g must be a numpy.ndarray; router state is corrupted. "
                "W_g is testing-only mutable."
            )
        if not np.issubdtype(self.W_g.dtype, np.floating):
            raise RuntimeError(
                "W_g must have a floating dtype; router state is corrupted. "
                "W_g is testing-only mutable."
            )
        expected = (self.d_model, self.num_experts)
        if self.W_g.shape != expected:
            raise RuntimeError(
                f"W_g shape {self.W_g.shape} != expected {expected}; "
                "router state is corrupted. W_g is testing-only mutable; "
                "do not replace it with a wrong-shape array."
            )

    def route(self, X: np.ndarray) -> RoutingResult:
        """Compute full softmax probabilities, top-2 expert indices, and
        per-token renormalized gate weights for every row of X.

        Parameters
        ----------
        X : (N, d_model) floating ndarray, finite, N >= 1. Not mutated.

        Returns
        -------
        RoutingResult — all arrays are new allocations.

        Raises
        ------
        TypeError   : X not a floating ndarray.
        ValueError  : X not 2-D, N == 0, d != d_model, contains NaN/Inf.
        RuntimeError: W_g is not an ndarray, not floating, wrong shape,
                      or logits overflowed to non-finite values despite
                      finite X and W_g. The overflow guard is a raise,
                      not an assert, so it survives `python -O`.
        """
        X = self._validate_input(X)
        self._check_weights_shape()

        # (1) Dense gating projection. Finite X and finite W_g can still
        #     overflow; suppress the warning so the range event is not
        #     mistaken for an uncontrolled fault. We detect the failure
        #     explicitly on the next line.
        with np.errstate(over="ignore", invalid="ignore"):
            logits = X @ self.W_g

        if not np.all(np.isfinite(logits)):
            raise RuntimeError(
                "router logits are non-finite; the projection overflowed "
                "or W_g contains NaN/Inf. Reduce input magnitude or "
                "reinitialize the router."
            )

        # (2) Numerically stable softmax. O(N*E).
        probs = _stable_softmax(logits)

        # (3) Deterministic top-2 selection. Stable argsort of -logits;
        #     ties resolve to lower index. O(N*E*log E).
        order = np.argsort(-logits, axis=1, kind="stable")
        top2_indices = order[:, :2].astype(np.int64, copy=True)

        # (4) Gather selected probabilities (NOT probs[:, :2]) and
        #     renormalize along the top-2 axis.
        raw_top2 = np.take_along_axis(probs, top2_indices, axis=1)
        denom = np.maximum(raw_top2.sum(axis=1, keepdims=True), _EPS)
        top2_weights = raw_top2 / denom

        # Post-conditions of the softmax and renormalization math. Rule 8:
        # no NaN/Inf may reach the public API. Raises (not asserts) so
        # the guard survives `python -O`.
        if not np.all(np.isfinite(probs)):
            raise RuntimeError(
                "internal error: softmax produced non-finite probabilities"
            )
        if not np.all(np.isfinite(top2_weights)):
            raise RuntimeError(
                "internal error: renormalization produced non-finite weights"
            )

        return RoutingResult(top2_indices, top2_weights, probs)


def compute_load_balancing_loss(
    probs: np.ndarray,
    top2_indices: np.ndarray,
    num_experts: int,
    alpha: float = 0.01,
) -> float:
    """Auxiliary load-balancing loss for Top-2 routing.

    Formula (all intermediates defined explicitly):

        f_j   = (1/N) * count_i[ j in top2_indices[i] ]      # token fraction
        P_j   = (1/N) * sum_i probs[i, j]                    # mean gate prob
        L_aux = alpha * E * sum_j f_j * P_j

    Because each token selects exactly 2 experts, sum_j f_j = 2 (not 1).
    Under perfectly uniform routing and uniform mean probabilities
    (f_j = 2/E, P_j = 1/E) this evaluates to 2*alpha, NOT alpha.

    Differentiability
    -----------------
    top2_indices is discrete; df_j/dW_g = 0 almost everywhere. The
    differentiable path is P_j = mean_i softmax(X W_g)[i, j]. Forward
    value only; no backward pass is provided.

    Parameters
    ----------
    probs : (N, E) float array. N >= 1, rows sum to 1 within atol=1e-5,
        finite, non-negative.
    top2_indices : (N, 2) integer array. Two DISTINCT indices in [0, E)
        per row.
    num_experts : int, >= 2, not bool. Must equal probs.shape[1].
    alpha : float, finite, >= 0. `alpha=0` supported (yields 0.0).
        `bool` is rejected (subclasses int but is almost certainly a
        caller bug).

    Returns
    -------
    Python float. No epsilon is added to the sum because no division or
    logarithm appears after validation.

    Raises
    ------
    TypeError  : wrong dtype for probs/top2_indices/alpha, or bool alpha.
    ValueError : shape/dtype/range/finiteness contract violated.
    """
    # ---- validate config --------------------------------------------------
    num_experts = _coerce_int(num_experts, "num_experts", 2)

    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError(f"alpha must be a real number, not bool: got {alpha!r}")
    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError(f"alpha must be a real number, got {type(alpha).__name__}")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha!r}")

    # ---- validate probs ---------------------------------------------------
    if not isinstance(probs, np.ndarray):
        raise TypeError("probs must be a numpy.ndarray")
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, E), got ndim={probs.ndim}")
    N, E = probs.shape
    if N == 0:
        raise ValueError("probs must have N >= 1 rows")
    if E != num_experts:
        raise ValueError(f"probs.shape[1]={E} != num_experts={num_experts}")
    if not np.issubdtype(probs.dtype, np.floating):
        raise TypeError(f"probs must have floating dtype, got {probs.dtype}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probs contains NaN or Inf")
    if np.any(probs < 0):
        raise ValueError("probs contains negative entries")
    row_sums = probs.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=_ROW_SUM_TOL):
        raise ValueError(
            f"probs rows must sum to 1 within atol={_ROW_SUM_TOL}; "
            f"got min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )

    # ---- validate top2_indices -------------------------------------------
    if not isinstance(top2_indices, np.ndarray):
        raise TypeError("top2_indices must be a numpy.ndarray")
    if top2_indices.ndim != 2 or top2_indices.shape[1] != 2:
        raise ValueError(
            f"top2_indices must have shape (N, 2), got {top2_indices.shape}"
        )
    if top2_indices.shape[0] != N:
        raise ValueError(
            f"top2_indices.shape[0]={top2_indices.shape[0]} != probs.shape[0]={N}"
        )
    if not np.issubdtype(top2_indices.dtype, np.integer):
        raise TypeError(
            f"top2_indices must have integer dtype, got {top2_indices.dtype}"
        )
    if top2_indices.min() < 0 or top2_indices.max() >= num_experts:
        raise ValueError(
            f"top2_indices out of range [0, {num_experts}); "
            f"min={int(top2_indices.min())}, max={int(top2_indices.max())}"
        )
    if not np.all(top2_indices[:, 0] != top2_indices[:, 1]):
        bad = np.where(top2_indices[:, 0] == top2_indices[:, 1])[0]
        raise ValueError(
            f"top2_indices rows must contain two DISTINCT experts; "
            f"duplicate expert at row(s) {bad[:5].tolist()}"
        )

    # ---- compute ----------------------------------------------------------
    probs64 = probs.astype(np.float64, copy=False)
    counts = np.bincount(
        top2_indices.astype(np.int64, copy=False).ravel(),
        minlength=num_experts,
    ).astype(np.float64)  # (E,), sum == 2*N
    f = counts / N  # sum(f) == 2
    P = probs64.mean(axis=0)  # sum(P) == 1
    loss = alpha * num_experts * float(np.sum(f * P))
    if not np.isfinite(loss):
        raise RuntimeError(f"load-balancing loss is non-finite: {loss!r}")
    return loss
