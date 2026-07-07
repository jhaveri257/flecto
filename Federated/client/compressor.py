"""
Federated/client/compressor.py
================================
Sparse (index, value) serializer for FL model updates.

Mathematical Formulation
------------------------
A sparse update is represented as a pair of parallel arrays::

    transmission = { (i, v) : i in S, v = Delta_W[i] }

where S is the Top-K index set from ``update_selector.select_topk()``.

Byte Cost Model
---------------
Each (index, value) pair is encoded as:

    - index : int64  → 8 bytes
    - value : float32 → 4 bytes
    - pair total   → 12 bytes

Plus a fixed-size header carrying metadata::

    header = { client_id: int32,  round_idx: int32,
               total_params: int32,  k: int32 }
             = 4 × 4 bytes = 16 bytes

Total payload size (bytes) = 16 + 12 × k

The full (dense) update would cost ``4 × P`` bytes (float32).

Compression ratio = payload_bytes / full_bytes
                  = (16 + 12k) / (4P)

For P = 109,386 and k = 10,939 (10%):
    full    = 437,544 bytes (≈ 427 KB)
    sparse  =  131,284 bytes (≈ 128 KB)
    ratio   ≈ 0.30   (70% bandwidth saving)

The savings improve nonlinearly as top_k_ratio decreases:
    1%  → ratio ≈ 0.031   (96.9% saving)
    5%  → ratio ≈ 0.15    (85.0% saving)
    10% → ratio ≈ 0.30    (70.0% saving)

Design Decisions
----------------
1. No entropy coding (Huffman/Arithmetic): excluded by Requirement 3.
   This keeps the implementation auditable and avoids bitstream libraries.

2. No quantization: excluded by Requirement 3.  Values are stored as
   full-precision float32 to avoid introducing quantization error that
   would confound the error-feedback analysis.

3. Int64 indices: safe for models up to 2^63 ≈ 9 × 10^18 parameters.
   For our model (109,386 params), int32 would suffice, but int64 is used
   for forward compatibility with larger models.

4. Sorted index contract: the ``Compressor`` asserts that received indices
   are sorted.  This is guaranteed by ``update_selector.select_topk()``
   (Requirement 2) and allows the reconstructor to use binary search if needed.

5. The ``Compressor`` is stateless — it holds only configuration constants
   and can be reused across rounds and clients without risk of state leakage.

Algorithm Complexity
--------------------
- compress():   O(k)  — one array copy of size k for indices + values
- decompress(): O(k)  — scatter of k values into zero buffer of size P
- validate():   O(k)  — single pass for NaN/Inf/duplicate/bounds checks

Memory Complexity
-----------------
- Input:  k indices (8 bytes each) + k values (4 bytes each)
- Output: SparsePayload ≈ 12k + 16 bytes
- Peak:   O(k) additional allocation (argsort buffer for duplicate check)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from Federated.config import TOP_K_RATIO
from Federated.transport_bridge.payload import CompressionStats, SparsePayload
from Federated.client.update_selector import UpdateSelection


# Bytes per index (int64)
_INDEX_DTYPE = np.int64
_INDEX_BYTES = 8

# Bytes per value (float32)
_VALUE_DTYPE = np.float32
_VALUE_BYTES = 4

# Fixed metadata header: client_id, round_idx, total_params, k  (4 × int32)
_HEADER_BYTES = 16


# ===========================================================================
# Compressor
# ===========================================================================

class Compressor:
    """
    Stateless sparse update compressor.

    Converts a ``UpdateSelection`` (Top-K result) into a ``SparsePayload``
    by recording the selected ``(index, value)`` pairs and computing
    bandwidth-cost metadata.

    Also provides ``decompress()`` to reconstruct a dense update vector
    from a ``SparsePayload`` on the server side.

    Parameters
    ----------
    total_params : int
        Total number of scalar parameters in the model.  Used as the
        reference denominator for the full (dense) byte cost.

    Attributes
    ----------
    total_params : int
        Stored at construction.

    Examples
    --------
    >>> comp = Compressor(total_params=109386)
    >>> payload = comp.compress(selection, client_id=0, round_idx=1)
    >>> dense = comp.decompress(payload)
    >>> dense.shape
    (109386,)
    """

    def __init__(self, total_params: int) -> None:
        if total_params <= 0:
            raise ValueError(
                f"Compressor: total_params must be > 0, got {total_params}."
            )
        self.total_params = total_params
        # Pre-compute the full (uncompressed) payload size for ratio calculations
        self._full_bytes = total_params * _VALUE_BYTES

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(
        self,
        selection: UpdateSelection,
        client_id: int,
        round_idx: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SparsePayload:
        """
        Serialise a ``UpdateSelection`` into a ``SparsePayload``.

        Validates the selection, computes byte costs, and packages the
        result into the immutable ``SparsePayload`` dataclass.

        Parameters
        ----------
        selection : UpdateSelection
            Top-K selection result from ``update_selector.select_topk()``
            or ``compute_and_select()``.
        client_id : int
            Integer identifier of the transmitting client.
        round_idx : int
            Current FL communication round number.
        extra_metadata : dict, optional
            Additional key-value pairs merged into ``payload.metadata``.
            Examples: ``{"error_feedback_applied": True, "delta_norm": 0.04}``.

        Returns
        -------
        SparsePayload
            Immutable payload containing sorted indices, signed values,
            and all bandwidth-cost metadata.

        Raises
        ------
        ValueError
            If the selection contains NaN/Inf values, duplicate indices,
            out-of-bounds indices, or is empty.
        TypeError
            If ``selection`` is not an ``UpdateSelection`` instance.

        Examples
        --------
        >>> payload = comp.compress(selection, client_id=2, round_idx=5)
        >>> payload.num_transmitted_params
        10938
        >>> payload.compression_ratio
        0.09999...
        """
        if not isinstance(selection, UpdateSelection):
            raise TypeError(
                f"compress: expected UpdateSelection, got {type(selection).__name__}."
            )

        t0 = time.perf_counter()

        # Full validation (Requirement 7)
        self._validate_selection(selection)

        # Byte cost calculation
        payload_bytes = _HEADER_BYTES + selection.k * (_INDEX_BYTES + _VALUE_BYTES)
        compression_ratio = selection.actual_ratio  # k / P

        meta: Dict[str, Any] = {
            "compress_time_s": time.perf_counter() - t0,
            "delta_norm": selection.delta_norm,
            "selection_time_s": selection.selection_time_s,
            "peak_memory_bytes": selection.peak_memory_bytes,
            "full_bytes": self._full_bytes,
            "sparsity": selection.sparsity,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return SparsePayload(
            client_id=client_id,
            round_idx=round_idx,
            indices=selection.indices.copy(),    # own the arrays
            values=selection.values.copy(),
            total_params=self.total_params,
            compression_ratio=compression_ratio,
            payload_bytes=payload_bytes,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Decompression (server side)
    # ------------------------------------------------------------------

    def decompress(
        self,
        payload: SparsePayload,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Reconstruct a dense update vector from a ``SparsePayload``.

        This is the inverse of ``compress`` — it scatters the transmitted
        ``(index, value)`` pairs into a zero-filled buffer of length
        ``total_params``.

        Unselected positions remain zero, which is the standard
        zero-fill reconstruction strategy for Top-K FL.

        Parameters
        ----------
        payload : SparsePayload
            Compressed payload as produced by ``compress()``.
        out : np.ndarray, optional
            Pre-allocated output buffer of shape ``(total_params,)`` and
            dtype float32.  When provided, it is zeroed and written in-place
            to avoid heap allocation.  When ``None``, a new array is allocated.

        Returns
        -------
        np.ndarray
            Dense float32 vector of shape ``(total_params,)`` with selected
            values scattered at their original positions.

        Raises
        ------
        ValueError
            If ``out`` is provided with the wrong shape or dtype.

        Examples
        --------
        >>> dense = comp.decompress(payload)
        >>> dense.shape
        (109386,)
        >>> dense[payload.indices].tolist() == payload.values.tolist()
        True
        >>> (dense == 0).sum()           # unselected positions are zero
        98447
        """
        if not isinstance(payload, SparsePayload):
            raise TypeError(
                f"decompress: expected SparsePayload, got {type(payload).__name__}."
            )

        P = self.total_params
        if payload.total_params != P:
            raise ValueError(
                f"decompress: payload.total_params={payload.total_params} "
                f"!= compressor.total_params={P}."
            )

        if out is not None:
            if out.shape != (P,):
                raise ValueError(
                    f"decompress: out has wrong shape {out.shape}, expected ({P},)."
                )
            if out.dtype != np.float32:
                raise ValueError(
                    f"decompress: out has wrong dtype {out.dtype}, expected float32."
                )
            out[:] = 0.0
        else:
            out = np.zeros(P, dtype=np.float32)

        # Scatter: O(k) assignment
        out[payload.indices] = payload.values
        return out

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_bytes(k: int) -> int:
        """
        Estimate payload size in bytes for ``k`` transmitted parameters.

        Parameters
        ----------
        k : int
            Number of Top-K selected parameters.

        Returns
        -------
        int
            Estimated on-wire size: ``16 + 12 * k`` bytes.

        Examples
        --------
        >>> Compressor.estimate_bytes(10938)
        131272
        """
        return _HEADER_BYTES + k * (_INDEX_BYTES + _VALUE_BYTES)

    @staticmethod
    def full_bytes(total_params: int) -> int:
        """
        Byte cost of an uncompressed (dense) float32 update.

        Parameters
        ----------
        total_params : int
            Total number of scalar model parameters.

        Returns
        -------
        int
            ``total_params * 4`` (float32 = 4 bytes per scalar).

        Examples
        --------
        >>> Compressor.full_bytes(109386)
        437544
        """
        return total_params * _VALUE_BYTES

    # ------------------------------------------------------------------
    # Validation (internal)
    # ------------------------------------------------------------------

    def _validate_selection(self, selection: UpdateSelection) -> None:
        """
        Comprehensive validation of a ``UpdateSelection`` before compression.

        Checks (in order):

        1. Non-empty selection.
        2. dtype conformance (int64 indices, float32 values).
        3. Length match between indices and values.
        4. No NaN or Inf in values.
        5. Sorted indices (required for deterministic serialization).
        6. No duplicate indices.
        7. All indices in valid range [0, total_params).

        Parameters
        ----------
        selection : UpdateSelection
            The selection to validate.

        Raises
        ------
        ValueError
            With a descriptive message identifying the first failing check.
        """
        indices = selection.indices
        values = selection.values
        k = selection.k
        P = self.total_params

        # 1. Empty guard
        if k == 0 or len(indices) == 0:
            raise ValueError(
                "Compressor._validate_selection: selection is empty (k=0). "
                "Increase TOP_K_RATIO or check delta computation."
            )

        # 2. dtype conformance
        if indices.dtype != np.int64:
            raise ValueError(
                f"_validate_selection: indices.dtype is {indices.dtype}, "
                f"expected int64."
            )
        if values.dtype != np.float32:
            raise ValueError(
                f"_validate_selection: values.dtype is {values.dtype}, "
                f"expected float32."
            )

        # 3. Length match
        if len(indices) != len(values):
            raise ValueError(
                f"_validate_selection: len(indices)={len(indices)} != "
                f"len(values)={len(values)}."
            )

        # 4. NaN / Inf in values
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"_validate_selection: values contain {nan_count} NaN(s) "
                f"and {inf_count} Inf(s).  Check gradient computation."
            )

        # 5. Sorted indices (REQUIREMENT 2 contract)
        if len(indices) > 1:
            diffs = np.diff(indices)
            if not np.all(diffs > 0):
                n_unsorted = int((diffs <= 0).sum())
                raise ValueError(
                    f"_validate_selection: indices are not strictly sorted ascending "
                    f"({n_unsorted} violation(s) found).  "
                    f"select_topk() must sort indices before compression."
                )

        # 6. Duplicate indices (implied by strict sort, but explicit check for clarity)
        #    Note: this is O(k) given the sorted guarantee above — just check diffs==0
        if len(indices) > 1 and int((np.diff(indices) == 0).sum()) > 0:
            raise ValueError(
                "_validate_selection: duplicate indices found.  "
                "Top-K selection must produce unique indices."
            )

        # 7. Bounds: all indices in [0, P)
        if len(indices) > 0:
            if int(indices.min()) < 0:
                raise ValueError(
                    f"_validate_selection: negative index found: {indices.min()}."
                )
            if int(indices.max()) >= P:
                raise ValueError(
                    f"_validate_selection: index {indices.max()} >= total_params={P}."
                )


# ===========================================================================
# Compression Statistics Helper
# ===========================================================================

def compute_round_stats(
    payloads: list,    # List[SparsePayload]
    round_idx: int,
) -> CompressionStats:
    """
    Aggregate ``SparsePayload`` objects from all clients into round-level stats.

    Parameters
    ----------
    payloads : list[SparsePayload]
        All client payloads collected during one FL round.
    round_idx : int
        Current round index.

    Returns
    -------
    CompressionStats
        Aggregated statistics for logging.

    Raises
    ------
    ValueError
        If ``payloads`` is empty.
    """
    return CompressionStats.from_payloads(round_idx=round_idx, payloads=payloads)
