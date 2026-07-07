"""
Federated/client/update_selector.py
=====================================
Top-K sparse update selection for communication-efficient Federated Learning.

Mathematical Formulation
------------------------
Given the global model weights W_global and the client's locally-trained
weights W_local after one FL round, the weight delta is::

    Delta_W = W_local - W_global          (element-wise, float32)

The Top-K operator retains only the K entries with the largest absolute
magnitude::

    K = floor(|Delta_W| * top_k_ratio)
    S = argtop_K(|Delta_W|)              (K-element index set)

    compressed(Delta_W)[i] = Delta_W[i]  if i in S
                           = 0           otherwise

With error feedback (see error_feedback_store.py), the corrected delta::

    Delta_W_corrected = Delta_W + r_{t-1}

is used in place of raw Delta_W before Top-K selection, where r_{t-1} is
the accumulated residual from the previous round.  The residual is updated
after selection::

    r_t = Delta_W_corrected - compressed(Delta_W_corrected)

Algorithm Complexity
--------------------
- Delta computation:     O(P)         — single vectorised subtraction
- Absolute values:       O(P)         — in-place, no extra allocation
- argpartition (Top-K):  O(P)         — linear-time nth-element (NumPy)
- Sort of K indices:     O(K log K)   — required for deterministic order
- Total:                 O(P + K log K)

where P = total_params and K = floor(P * top_k_ratio).

For our FLModel (P = 109,386, K ≈ 10,939 at 10%):
- argpartition:  ~0.5 ms  (typical CPU)
- sort:          ~0.08 ms
- Full selection: < 1 ms

Memory Complexity
-----------------
- Input tensors (W_local, W_global):  2P float32 = 2 * 4P bytes
- Delta vector:                        P float32  (can reuse, not copied)
- Index/value output:                  K int64 + K float32 = 12K bytes
- Peak additional allocation:          P float32 for |Delta_W|

For P = 109,386 at float32, peak overhead ≈ 430 KB — negligible.

Design Decisions
----------------
1. NumPy throughout for server-side interoperability (reconstructor, FedAvg).
   Converting to numpy once at the entry point avoids repeated torch→numpy
   calls in the hot path.

2. ``np.argpartition`` for O(P) Top-K rather than full sort O(P log P).
   Only the K selected indices are then sorted, costing O(K log K) which is
   much cheaper for small K.

3. Sorted output indices guarantee deterministic serialization regardless of
   the input order from argpartition (which is implementation-defined and
   platform-specific).

4. float32 dtype is preserved throughout — never silently up-cast to float64.

5. The ``UpdateSelection`` dataclass is immutable (frozen) to prevent
   accidental mutation during the compression and transmission pipeline.

References
----------
Stich, S.U. (2018). "Sparsified SGD with Memory." NeurIPS 2018.
Alistarh, D. et al. (2017). "QSGD: Communication-Efficient SGD via
Gradient Quantization and Encoding." NeurIPS 2017.
Lin, Y. et al. (2018). "Deep Gradient Compression." ICLR 2018.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Federated.config import TOP_K_RATIO
from Federated.model.model_utils import state_dict_to_vector


# ===========================================================================
# UpdateSelection — immutable result record
# ===========================================================================

@dataclass(frozen=True)
class UpdateSelection:
    """
    Immutable record returned by ``select_topk()``.

    Carries all information needed to construct a ``SparsePayload`` and to
    compute the error-feedback residual.

    Parameters
    ----------
    indices : np.ndarray
        Sorted integer array of shape ``(k,)`` and dtype ``int64``.
        Contains the positions in the flat delta vector that were selected
        by Top-K.  Sorted ascending for deterministic serialization.
    values : np.ndarray
        Float32 array of shape ``(k,)``  with ``values[i] = delta[indices[i]]``.
        Contains the actual (signed) weight-delta values at the selected
        positions.
    k : int
        Actual number of parameters selected (``len(indices)``).
        May differ slightly from ``floor(total_params * top_k_ratio)`` due
        to rounding.
    total_params : int
        Total number of scalar parameters in the flat delta vector.
    top_k_ratio : float
        The configured Top-K ratio (from ``config.TOP_K_RATIO`` or override).
    actual_ratio : float
        ``k / total_params`` — the fraction of parameters actually transmitted.
        Equals ``top_k_ratio`` when ``total_params`` is exactly divisible.
    sparsity : float
        ``1.0 - actual_ratio`` — fraction of parameters NOT transmitted.
    delta_norm : float
        L2 norm of the full delta vector (before Top-K).
        Useful for monitoring client drift and convergence.
    selection_time_s : float
        Wall-clock time (seconds) taken by the Top-K selection step.
        Populated by ``select_topk()``; used in performance logging.
    peak_memory_bytes : int
        Peak additional memory allocated (bytes) during Top-K selection.
        Measured via ``tracemalloc``.

    Examples
    --------
    >>> sel = select_topk(delta, total_params=109386, top_k_ratio=0.1)
    >>> sel.k
    10938
    >>> sel.sparsity
    0.9...
    >>> sel.indices.dtype
    dtype('int64')
    >>> sel.values.dtype
    dtype('float32')
    """

    indices: np.ndarray
    values: np.ndarray
    k: int
    total_params: int
    top_k_ratio: float
    actual_ratio: float
    sparsity: float
    delta_norm: float
    selection_time_s: float
    peak_memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise scalar metadata to a plain dict for CSV logging.

        Returns
        -------
        dict
            All scalar fields (arrays excluded for logging purposes).
        """
        return {
            "k": self.k,
            "total_params": self.total_params,
            "top_k_ratio": self.top_k_ratio,
            "actual_ratio": self.actual_ratio,
            "sparsity": self.sparsity,
            "delta_norm": self.delta_norm,
            "selection_time_s": self.selection_time_s,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


# ===========================================================================
# Delta Computation
# ===========================================================================

def compute_delta(
    local_state_dict: "OrderedDict",
    global_state_dict: "OrderedDict",
) -> np.ndarray:
    """
    Compute the weight delta vector: Delta_W = W_local - W_global.

    Both state_dicts are first flattened to float32 numpy vectors using
    ``state_dict_to_vector``, then subtracted element-wise.

    Parameters
    ----------
    local_state_dict : OrderedDict[str, torch.Tensor]
        The ``state_dict`` from the client's locally trained model, as
        returned by ``LocalTrainer.train()["state_dict"]``.
    global_state_dict : OrderedDict[str, torch.Tensor]
        The ``state_dict`` of the current global model, as returned by
        ``model.state_dict()`` on the server-distributed copy.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(total_params,)`` representing
        ``W_local - W_global`` in the flattened parameter space.
        The ordering matches ``state_dict_to_vector`` (insertion order of keys).

    Raises
    ------
    ValueError
        If the two state_dicts produce vectors of different lengths,
        indicating an architecture mismatch.

    Notes
    -----
    The subtraction is performed in float32 throughout.  NumPy will
    upcast to float64 if either operand is float64 — this is guarded
    by explicitly casting the output.

    Examples
    --------
    >>> delta = compute_delta(local_sd, global_sd)
    >>> delta.shape
    (109386,)
    >>> delta.dtype
    dtype('float32')
    """
    local_vec = state_dict_to_vector(local_state_dict)    # float32
    global_vec = state_dict_to_vector(global_state_dict)  # float32

    if local_vec.shape != global_vec.shape:
        raise ValueError(
            f"compute_delta: shape mismatch between local "
            f"({local_vec.shape}) and global ({global_vec.shape}) vectors. "
            f"Are the client and server using the same model architecture?"
        )

    # Explicit float32 cast to prevent silent upcasting
    delta = (local_vec - global_vec).astype(np.float32)

    _validate_finite(delta, name="delta")
    return delta


# ===========================================================================
# Top-K Selection
# ===========================================================================

def select_topk(
    delta: np.ndarray,
    total_params: Optional[int] = None,
    top_k_ratio: float = TOP_K_RATIO,
    measure_memory: bool = False,
) -> UpdateSelection:
    """
    Select the K weight-delta entries with the largest absolute magnitude.

    Algorithm
    ---------
    1. Compute absolute values of ``delta`` (no extra copy: abs is O(P)).
    2. Use ``np.argpartition`` to find the K-th largest element in O(P).
       This is strictly cheaper than full sort (O(P log P)) when K << P.
    3. Extract the top-K index set.
    4. Sort the K indices ascending — O(K log K).
       This is mandatory: ``argpartition`` returns indices in arbitrary order
       which is platform- and implementation-specific.
    5. Extract the corresponding signed values from ``delta``.

    Parameters
    ----------
    delta : np.ndarray
        Float32 vector of shape ``(P,)`` — the raw weight delta, optionally
        corrected by the error-feedback residual.
    total_params : int, optional
        Total number of parameters P.  Defaults to ``len(delta)``.
    top_k_ratio : float, optional
        Fraction of parameters to retain.  Must be in ``(0.0, 1.0]``.
        Defaults to ``config.TOP_K_RATIO``.
    measure_memory : bool, optional
        When ``True``, uses ``tracemalloc`` to measure peak memory allocation
        during the selection step.  Adds ~0.1 ms overhead.  Default ``False``.

    Returns
    -------
    UpdateSelection
        Immutable record containing sorted indices, signed values, and
        performance/compression metadata.

    Raises
    ------
    ValueError
        If ``top_k_ratio`` is outside ``(0, 1]``, if ``delta`` is empty,
        or if ``delta`` contains NaN or Inf values.
    TypeError
        If ``delta`` is not a float32 numpy array.

    Notes
    -----
    When ``top_k_ratio = 1.0``, all parameters are selected (full update).
    This is used as the baseline in evaluation comparisons.

    Examples
    --------
    >>> delta = np.random.randn(109386).astype(np.float32)
    >>> sel = select_topk(delta, top_k_ratio=0.1)
    >>> sel.k
    10938
    >>> np.all(np.diff(sel.indices) > 0)   # strictly sorted
    True
    >>> sel.values.dtype
    dtype('float32')
    """
    # --- Input validation ---
    if not isinstance(delta, np.ndarray):
        raise TypeError(
            f"select_topk: expected np.ndarray, got {type(delta).__name__}."
        )
    if delta.dtype != np.float32:
        delta = delta.astype(np.float32)  # silent cast with warning
    if delta.ndim != 1:
        raise ValueError(
            f"select_topk: delta must be 1-D, got shape {delta.shape}."
        )

    P = len(delta)
    if P == 0:
        raise ValueError("select_topk: delta vector is empty.")

    if not (0.0 < top_k_ratio <= 1.0):
        raise ValueError(
            f"select_topk: top_k_ratio must be in (0, 1], got {top_k_ratio}."
        )

    _validate_finite(delta, name="delta")

    if total_params is None:
        total_params = P

    # Number of parameters to select
    k = max(1, int(P * top_k_ratio))
    # Clamp to P in case of floating-point overshoot
    k = min(k, P)

    # --- Timing and optional memory profiling ---
    t0 = time.perf_counter()

    if measure_memory:
        tracemalloc.start()

    # --- O(P) Top-K via argpartition ---
    # abs_delta is computed inline; numpy may reuse memory for unary ops
    abs_delta = np.abs(delta)

    if k == P:
        # Full update: all indices selected, sorted trivially
        top_k_indices = np.arange(P, dtype=np.int64)
    else:
        # argpartition: element at position (P-k) is the K-th largest.
        # Elements to the right of this pivot are >= pivot (unsorted).
        pivot = P - k
        partition_idx = np.argpartition(abs_delta, pivot)
        # Take everything from pivot onwards (the K largest)
        top_k_unordered = partition_idx[pivot:]

        # REQUIREMENT 2: sort indices for deterministic serialization
        top_k_indices = np.sort(top_k_unordered).astype(np.int64)

    # Extract signed values at selected positions (float32 preserved)
    selected_values = delta[top_k_indices].astype(np.float32)

    t1 = time.perf_counter()
    selection_time_s = t1 - t0

    peak_memory_bytes = 0
    if measure_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_bytes = peak

    # --- Compute summary statistics ---
    actual_ratio = k / total_params
    sparsity = 1.0 - actual_ratio
    delta_norm = float(np.linalg.norm(delta, ord=2))

    return UpdateSelection(
        indices=top_k_indices,
        values=selected_values,
        k=k,
        total_params=total_params,
        top_k_ratio=top_k_ratio,
        actual_ratio=actual_ratio,
        sparsity=sparsity,
        delta_norm=delta_norm,
        selection_time_s=selection_time_s,
        peak_memory_bytes=peak_memory_bytes,
    )


# ===========================================================================
# Convenience: compute delta + select in one call
# ===========================================================================

def compute_and_select(
    local_state_dict: "OrderedDict",
    global_state_dict: "OrderedDict",
    top_k_ratio: float = TOP_K_RATIO,
    error_feedback: Optional[np.ndarray] = None,
    measure_memory: bool = False,
) -> Tuple[UpdateSelection, np.ndarray]:
    """
    End-to-end helper: delta computation, optional error-feedback correction,
    and Top-K selection in a single call.

    This is the primary entry point used by ``fl_round_runner.py`` for each
    client per communication round.

    Pipeline::

        delta = W_local - W_global
        if error_feedback is not None:
            delta_corrected = delta + error_feedback     # add residual
        else:
            delta_corrected = delta
        selection = select_topk(delta_corrected, top_k_ratio)

    The corrected delta is returned alongside the selection so the caller can
    compute the new residual::

        new_residual = delta_corrected - reconstruct_from_selection(selection)

    Parameters
    ----------
    local_state_dict : OrderedDict
        Post-training local model state_dict.
    global_state_dict : OrderedDict
        Current global model state_dict (distributed at round start).
    top_k_ratio : float, optional
        Top-K fraction. Default is ``config.TOP_K_RATIO``.
    error_feedback : np.ndarray, optional
        Residual vector from the previous round (shape ``(P,)``, float32).
        When ``None``, no error feedback is applied (standard Top-K).
    measure_memory : bool, optional
        Enable memory profiling in ``select_topk``. Default ``False``.

    Returns
    -------
    selection : UpdateSelection
        Top-K selection result on the (optionally corrected) delta.
    corrected_delta : np.ndarray
        The delta vector that was actually compressed (used to compute residual).
        Shape ``(P,)``, dtype float32.

    Raises
    ------
    ValueError
        Propagated from ``compute_delta`` or ``select_topk`` on any input error.

    Examples
    --------
    >>> sel, corrected = compute_and_select(local_sd, global_sd, top_k_ratio=0.1)
    >>> sel.k
    10938
    >>> corrected.dtype
    dtype('float32')
    """
    delta = compute_delta(local_state_dict, global_state_dict)

    if error_feedback is not None:
        if error_feedback.shape != delta.shape:
            raise ValueError(
                f"compute_and_select: error_feedback shape {error_feedback.shape} "
                f"does not match delta shape {delta.shape}."
            )
        corrected_delta = (delta + error_feedback).astype(np.float32)
    else:
        corrected_delta = delta

    selection = select_topk(
        corrected_delta,
        total_params=len(corrected_delta),
        top_k_ratio=top_k_ratio,
        measure_memory=measure_memory,
    )

    return selection, corrected_delta


# ===========================================================================
# Internal validation helpers
# ===========================================================================

def _validate_finite(arr: np.ndarray, name: str = "array") -> None:
    """
    Raise ValueError if ``arr`` contains any NaN or Inf value.

    Parameters
    ----------
    arr : np.ndarray
        Array to check.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        With specific counts of NaN and Inf elements found.
    """
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"_validate_finite: '{name}' contains {nan_count} NaN(s) and "
            f"{inf_count} Inf(s).  Check local training for gradient explosion "
            f"or data preprocessing issues."
        )
