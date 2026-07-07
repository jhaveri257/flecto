"""
Federated/transport_bridge/payload.py
=======================================
Lightweight communication payload abstraction for the FL transport bridge.

This module defines ``SparsePayload``, a data class that represents a single
client-to-server communication payload containing a sparse model update.

Design intent
-------------
The ``SparsePayload`` object is the single artifact that flows through the
entire compression-transmission-reconstruction pipeline:

1. **Compressor** (client side) creates a ``SparsePayload`` from a sparse
   ``(indices, values)`` update.
2. **Payload adapter** serialises it to raw bytes for transmission.
3. **Controller bridge** passes the byte buffer plus metadata to the
   existing ``TCPAdapter`` / ``FCAdapter`` for simulated transport.
4. **Reconstructor** (server side) deserialises the ``SparsePayload`` and
   fills the zero-padded dense weight vector.

This module deliberately contains **no transport logic**.  It is a pure
data container.  All network-level operations live in ``controller_bridge.py``
and ``payload_adapter.py``.

Classes
-------
SparsePayload
    Immutable record representing one compressed weight-update transmission.
CompressionStats
    Lightweight named summary of compression metrics (used for logging).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ===========================================================================
# SparsePayload
# ===========================================================================

@dataclass
class SparsePayload:
    """
    Represents a single client-to-server communication payload.

    A ``SparsePayload`` carries a sparse model update encoded as parallel
    arrays of ``(indices, values)`` pairs, plus metadata needed by the server
    for reconstruction, aggregation, and logging.

    Parameters
    ----------
    client_id : int
        Integer identifier of the client that generated this payload.
    round_idx : int
        FL communication round number during which this payload was created.
    indices : np.ndarray
        Integer array of shape ``(k,)`` containing the positions in the
        flat weight-delta vector that were selected by Top-K.
        dtype should be ``np.int64`` or ``np.int32``.
    values : np.ndarray
        Float32 array of shape ``(k,)`` containing the update magnitudes at
        the selected ``indices``.  ``values[i]`` is the delta at position
        ``indices[i]``.
    total_params : int
        Total number of scalar parameters in the full (dense) weight vector.
        Required by the server to allocate the zero-filled reconstruction
        buffer of the correct size.
    compression_ratio : float
        Fraction of parameters transmitted relative to the full model:
        ``len(indices) / total_params``.  Equals ``TOP_K_RATIO`` in the
        standard implementation but may differ if additional filtering is
        applied.
    payload_bytes : int
        Estimated on-wire size in bytes of this payload.
        Computed by ``PayloadAdapter.estimate_bytes()``.
        Used for communication cost tracking and bandwidth logging.
    timestamp : float
        Unix timestamp (``time.time()``) at payload creation.
        Used for per-round timing analysis.
    metadata : dict, optional
        Arbitrary key-value pairs for extensibility.  Examples:
        ``{"error_feedback_applied": True, "norm_delta": 0.032}``.

    Attributes
    ----------
    num_transmitted_params : int
        Read-only property.  Equal to ``len(indices)``.
    bandwidth_saved_fraction : float
        Read-only property.  Fraction of parameters NOT transmitted:
        ``1.0 - compression_ratio``.

    Examples
    --------
    >>> payload = SparsePayload(
    ...     client_id=3,
    ...     round_idx=7,
    ...     indices=np.array([10, 42, 8800], dtype=np.int64),
    ...     values=np.array([0.012, -0.003, 0.047], dtype=np.float32),
    ...     total_params=109386,
    ...     compression_ratio=3 / 109386,
    ...     payload_bytes=3 * (8 + 4),   # 3 pairs * (int64 + float32)
    ... )
    >>> payload.num_transmitted_params
    3
    >>> payload.bandwidth_saved_fraction
    0.9999...
    """

    # --- Required fields ---
    client_id: int
    round_idx: int
    indices: np.ndarray
    values: np.ndarray
    total_params: int
    compression_ratio: float
    payload_bytes: int

    # --- Optional fields with defaults ---
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_transmitted_params(self) -> int:
        """
        Number of parameters actually transmitted (length of sparse update).

        Returns
        -------
        int
            ``len(self.indices)``.
        """
        return int(self.indices.shape[0])

    @property
    def bandwidth_saved_fraction(self) -> float:
        """
        Fraction of model parameters NOT transmitted in this payload.

        Returns
        -------
        float
            Value in ``[0.0, 1.0)``.  Higher means more compression.
            For ``TOP_K_RATIO=0.1`` this will be approximately 0.9.
        """
        return 1.0 - self.compression_ratio

    @property
    def full_payload_size_bytes(self) -> int:
        """
        Hypothetical size of an uncompressed (dense) payload in bytes.

        Assumes float32 (4 bytes per parameter).

        Returns
        -------
        int
            ``total_params * 4``.
        """
        return self.total_params * 4

    @property
    def size_reduction_factor(self) -> float:
        """
        Ratio of uncompressed to compressed payload size.

        Returns
        -------
        float
            ``full_payload_size_bytes / max(1, payload_bytes)``.
            Values > 1 indicate compression achieved.
        """
        return self.full_payload_size_bytes / max(1, self.payload_bytes)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate array shapes and value consistency on construction."""
        if self.indices.ndim != 1:
            raise ValueError(
                f"SparsePayload: indices must be 1-D, got shape {self.indices.shape}."
            )
        if self.values.ndim != 1:
            raise ValueError(
                f"SparsePayload: values must be 1-D, got shape {self.values.shape}."
            )
        if self.indices.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"SparsePayload: indices and values must have the same length. "
                f"Got {self.indices.shape[0]} indices and {self.values.shape[0]} values."
            )
        if not (0.0 < self.compression_ratio <= 1.0):
            raise ValueError(
                f"SparsePayload: compression_ratio must be in (0, 1], "
                f"got {self.compression_ratio}."
            )
        if self.num_transmitted_params > self.total_params:
            raise ValueError(
                f"SparsePayload: cannot transmit more params ({self.num_transmitted_params}) "
                f"than exist in the model ({self.total_params})."
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable one-line summary string.

        Returns
        -------
        str
            Formatted summary suitable for logging.

        Examples
        --------
        >>> print(payload.summary())
        [Round  7 | Client  3] params=3/109386 | ratio=0.0000 | saved=100.0% | bytes=36
        """
        return (
            f"[Round {self.round_idx:2d} | Client {self.client_id:2d}] "
            f"params={self.num_transmitted_params}/{self.total_params} | "
            f"ratio={self.compression_ratio:.4f} | "
            f"saved={self.bandwidth_saved_fraction * 100:.1f}% | "
            f"bytes={self.payload_bytes}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the payload metadata to a plain Python dictionary.

        Arrays are converted to lists for JSON compatibility.
        Intended for logging to CSV or JSON.

        Returns
        -------
        dict
            Flat dictionary with all payload fields except raw numpy arrays.
        """
        return {
            "client_id": self.client_id,
            "round_idx": self.round_idx,
            "num_transmitted_params": self.num_transmitted_params,
            "total_params": self.total_params,
            "compression_ratio": self.compression_ratio,
            "bandwidth_saved_fraction": self.bandwidth_saved_fraction,
            "payload_bytes": self.payload_bytes,
            "full_payload_bytes": self.full_payload_size_bytes,
            "size_reduction_factor": self.size_reduction_factor,
            "timestamp": self.timestamp,
            **self.metadata,
        }


# ===========================================================================
# CompressionStats
# ===========================================================================

@dataclass
class CompressionStats:
    """
    Lightweight record summarising compression metrics for a single round.

    Aggregated from all client ``SparsePayload`` objects at the end of
    each FL round.  Written to the CSV log by the metrics module.

    Parameters
    ----------
    round_idx : int
        Communication round number.
    num_clients : int
        Number of clients that contributed payloads this round.
    total_params : int
        Total model parameters (same for all clients).
    avg_transmitted_params : float
        Mean number of parameters transmitted per client.
    avg_compression_ratio : float
        Mean compression ratio across clients.
    total_bytes_transmitted : int
        Sum of ``payload_bytes`` across all client payloads this round.
    total_bytes_uncompressed : int
        Hypothetical total bytes if all clients sent full dense updates.
    avg_bandwidth_saved : float
        Mean fraction of bandwidth saved across clients.
    """

    round_idx: int
    num_clients: int
    total_params: int
    avg_transmitted_params: float
    avg_compression_ratio: float
    total_bytes_transmitted: int
    total_bytes_uncompressed: int
    avg_bandwidth_saved: float

    @classmethod
    def from_payloads(
        cls,
        round_idx: int,
        payloads: list,   # List[SparsePayload]
    ) -> "CompressionStats":
        """
        Compute aggregated compression statistics from a list of payloads.

        Parameters
        ----------
        round_idx : int
            Current FL round index.
        payloads : list[SparsePayload]
            All client payloads for this round.

        Returns
        -------
        CompressionStats
            Aggregated statistics object.
        """
        if not payloads:
            raise ValueError("Cannot compute stats from an empty payload list.")

        n = len(payloads)
        total_params = payloads[0].total_params

        avg_tx = sum(p.num_transmitted_params for p in payloads) / n
        avg_ratio = sum(p.compression_ratio for p in payloads) / n
        total_bytes_tx = sum(p.payload_bytes for p in payloads)
        total_bytes_uncomp = sum(p.full_payload_size_bytes for p in payloads)
        avg_saved = sum(p.bandwidth_saved_fraction for p in payloads) / n

        return cls(
            round_idx=round_idx,
            num_clients=n,
            total_params=total_params,
            avg_transmitted_params=avg_tx,
            avg_compression_ratio=avg_ratio,
            total_bytes_transmitted=total_bytes_tx,
            total_bytes_uncompressed=total_bytes_uncomp,
            avg_bandwidth_saved=avg_saved,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dict for CSV logging.

        Returns
        -------
        dict
        """
        return {
            "round_idx": self.round_idx,
            "num_clients": self.num_clients,
            "total_params": self.total_params,
            "avg_transmitted_params": self.avg_transmitted_params,
            "avg_compression_ratio": self.avg_compression_ratio,
            "total_bytes_transmitted": self.total_bytes_transmitted,
            "total_bytes_uncompressed": self.total_bytes_uncompressed,
            "avg_bandwidth_saved": self.avg_bandwidth_saved,
        }


# ===========================================================================
# Self-Test
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("SparsePayload self-test")
    print("-" * 40)

    k = 10946       # 10% of 109,386 total params
    total = 109386

    payload = SparsePayload(
        client_id=0,
        round_idx=1,
        indices=np.arange(k, dtype=np.int64),
        values=np.random.randn(k).astype(np.float32),
        total_params=total,
        compression_ratio=k / total,
        payload_bytes=k * (8 + 4),    # int64 index + float32 value
    )

    print(f"  num_transmitted_params : {payload.num_transmitted_params}")
    print(f"  bandwidth_saved        : {payload.bandwidth_saved_fraction * 100:.1f}%")
    print(f"  size_reduction_factor  : {payload.size_reduction_factor:.1f}x")
    print(f"  summary: {payload.summary()}")

    d = payload.to_dict()
    assert "compression_ratio" in d
    print(f"  to_dict keys: {list(d.keys())}")

    # CompressionStats
    stats = CompressionStats.from_payloads(round_idx=1, payloads=[payload])
    print(f"  CompressionStats: avg_saved={stats.avg_bandwidth_saved * 100:.1f}%")

    # Validation guard
    try:
        bad = SparsePayload(
            client_id=0, round_idx=0,
            indices=np.array([1, 2], dtype=np.int64),
            values=np.array([0.1], dtype=np.float32),   # shape mismatch
            total_params=total,
            compression_ratio=0.1,
            payload_bytes=100,
        )
        print("[FAIL] Should have raised ValueError")
        sys.exit(1)
    except ValueError:
        print("  [PASS] Shape mismatch correctly caught.")

    print("\n[PASS] SparsePayload self-test complete.")
    sys.exit(0)
