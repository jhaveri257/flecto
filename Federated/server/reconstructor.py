"""
Federated/server/reconstructor.py
====================================
Server-side dense update reconstruction from SparsePayload objects.

Mathematical Formulation
------------------------
Given a SparsePayload with index set S and values v, the reconstructor
produces the zero-fill dense reconstruction::

    delta_hat[i] = v_i   if  i in S
    delta_hat[i] = 0     otherwise

Equivalently, using the scatter operation::

    delta_hat = zeros(P)
    delta_hat[S] = v

This is the standard Top-K reconstruction strategy described in:
    Lin et al. (2018). "Deep Gradient Compression." ICLR 2018.
    Stich et al. (2018). "Sparsified SGD with Memory." NeurIPS 2018.

Reconstruction Error
--------------------
Given access to the original corrected delta (available in simulation),
the normalised L2 reconstruction error is::

    epsilon_i = ||delta_corrected_i - delta_hat_i||_2
              / max(eps, ||delta_corrected_i||_2)

where delta_corrected_i = Delta_W_i + r_{t-1,i}  (error-feedback-corrected).

This is exactly the L2 norm of the portion discarded by Top-K selection,
normalised by the full-update norm.  For error-feedback-enabled runs,
epsilon approaches zero as residuals accumulate and eventually transmit
all gradient information.

Algorithm Complexity
--------------------
per payload:
    validate:       O(k)  — index bounds check
    scatter:        O(k)  — numpy advanced indexing
    recon_error:    O(P)  — two L2 norm computations

Total per round (N clients): O(N * (k + P))
With k = round(P * top_k_ratio): O(N * P * (1 + top_k_ratio))

Memory Complexity
-----------------
One pre-allocated buffer per client:  N * P * 4 bytes
For N=10, P=109386: 10 * 109386 * 4 ≈ 4.27 MB  (negligible)

The buffer is zeroed and rewritten each round, never reallocated.

Communication Complexity
------------------------
The Reconstructor operates entirely server-side.  No additional bytes
are transmitted beyond the SparsePayload already counted during compression.
Communication cost = sum of payload_bytes across all participating clients
                   = N * (16 + 12k) bytes per round.

Design Decisions
----------------
1. Pre-allocated buffer pool (REQUIREMENT 1/8):
   One numpy array per client, allocated lazily on first use, reused every
   round via `buf[:] = 0.0; buf[indices] = values`.  This avoids O(P)
   allocation cost on every reconstruction call.

2. Compressor.decompress() reuse:
   The scatter logic is delegated to the existing ``Compressor.decompress()``
   method via its ``out=`` parameter, avoiding code duplication and ensuring
   a single authoritative scatter implementation.

3. Optional corrected_delta argument:
   When provided (simulation context), the reconstruction error is computed
   and returned.  When None (real deployment where the server cannot access
   client gradients), the error defaults to NaN.

4. Fault tolerance (REQUIREMENT 7):
   Invalid payloads raise ``ReconstructionError`` rather than crashing the
   round.  The caller catches this and skips the offending client.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from Federated.client.compressor import Compressor
from Federated.config import NUM_CLIENTS
from Federated.transport_bridge.payload import SparsePayload


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ReconstructionError(ValueError):
    """Raised when a SparsePayload cannot be reconstructed."""


# ---------------------------------------------------------------------------
# ReconstructionResult — per-client result record
# ---------------------------------------------------------------------------

class ReconstructionResult:
    """
    Holds the output of one ``Reconstructor.reconstruct()`` call.

    Parameters
    ----------
    client_id : int
    round_idx : int
    dense_delta : np.ndarray
        Zero-fill reconstructed dense update, shape ``(P,)``, float32.
    reconstruction_error : float
        Normalised L2 error: ``||corrected_delta - delta_hat|| / ||corrected_delta||``.
        ``float('nan')`` when corrected_delta is not provided.
    recon_time_s : float
        Wall-clock time for this reconstruction call.
    """

    __slots__ = (
        "client_id", "round_idx", "dense_delta",
        "reconstruction_error", "recon_time_s",
    )

    def __init__(
        self,
        client_id: int,
        round_idx: int,
        dense_delta: np.ndarray,
        reconstruction_error: float,
        recon_time_s: float,
    ) -> None:
        self.client_id = client_id
        self.round_idx = round_idx
        self.dense_delta = dense_delta
        self.reconstruction_error = reconstruction_error
        self.recon_time_s = recon_time_s


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------

class Reconstructor:
    """
    Server-side sparse-to-dense reconstruction engine.

    Maintains a pre-allocated dense buffer per client to avoid
    per-round numpy allocation.  Uses ``Compressor.decompress()`` for
    the actual scatter operation.

    Parameters
    ----------
    total_params : int
        Total number of scalar model parameters.
    num_clients : int, optional
        Expected number of clients (used only for logging).

    Examples
    --------
    >>> rec = Reconstructor(total_params=109386)
    >>> result = rec.reconstruct(payload, corrected_delta=corrected)
    >>> result.dense_delta.shape
    (109386,)
    >>> result.reconstruction_error
    0.894...
    """

    def __init__(
        self,
        total_params: int,
        num_clients: int = NUM_CLIENTS,
    ) -> None:
        if total_params <= 0:
            raise ValueError(
                f"Reconstructor: total_params must be > 0, got {total_params}."
            )
        self.total_params = total_params
        self.num_clients = num_clients

        # Stateless compressor for scatter logic
        self._compressor = Compressor(total_params)

        # Pre-allocated dense buffers keyed by client_id (REQUIREMENT 1/8)
        self._buffers: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        payload: SparsePayload,
        corrected_delta: Optional[np.ndarray] = None,
    ) -> ReconstructionResult:
        """
        Reconstruct a dense update from a ``SparsePayload``.

        Pipeline::

            validate(payload)
            buf = get_or_create_buffer(payload.client_id)
            buf[:] = 0
            buf[payload.indices] = payload.values      # scatter
            if corrected_delta is not None:
                err = ||corrected_delta - buf|| / ||corrected_delta||
            return ReconstructionResult(dense_delta=buf.copy(), ...)

        Parameters
        ----------
        payload : SparsePayload
            Compressed update from one client.
        corrected_delta : np.ndarray, optional
            The error-feedback-corrected delta on the client side
            (shape ``(P,)``, float32).  When provided, the normalised
            L2 reconstruction error is computed and stored in the result.
            When ``None``, reconstruction_error is ``float('nan')``.

        Returns
        -------
        ReconstructionResult
            Contains the dense update array and diagnostic metrics.

        Raises
        ------
        ReconstructionError
            If the payload fails any validation check.
        """
        t0 = time.perf_counter()

        self._validate_payload(payload)

        buf = self._get_or_create_buffer(payload.client_id)

        # Scatter into pre-allocated buffer (in-place, avoids allocation)
        self._compressor.decompress(payload, out=buf)

        # Reconstruction error (simulation-only; requires ground-truth delta)
        recon_error = float("nan")
        if corrected_delta is not None:
            recon_error = self._compute_reconstruction_error(
                corrected_delta, buf
            )

        recon_time_s = time.perf_counter() - t0

        # Return a COPY of the buffer so the caller can safely store it
        # while the buffer is reused next round
        return ReconstructionResult(
            client_id=payload.client_id,
            round_idx=payload.round_idx,
            dense_delta=buf.copy(),
            reconstruction_error=recon_error,
            recon_time_s=recon_time_s,
        )

    def reconstruct_batch(
        self,
        payloads: List[SparsePayload],
        corrected_deltas: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[ReconstructionResult], List[int]]:
        """
        Reconstruct a batch of payloads, skipping any that fail validation.

        Fault tolerance: if any single payload raises ``ReconstructionError``,
        that client is skipped and its index is recorded in ``failed_indices``.

        Parameters
        ----------
        payloads : list[SparsePayload]
            One payload per participating client.
        corrected_deltas : list[np.ndarray], optional
            Parallel list of corrected delta vectors.  If provided, must have
            the same length as ``payloads``.

        Returns
        -------
        results : list[ReconstructionResult]
            Successfully reconstructed updates (may be shorter than ``payloads``).
        failed_indices : list[int]
            Indices (into ``payloads``) that raised ``ReconstructionError``.
        """
        if corrected_deltas is not None and len(corrected_deltas) != len(payloads):
            raise ValueError(
                "reconstruct_batch: payloads and corrected_deltas must have "
                "the same length."
            )

        results: List[ReconstructionResult] = []
        failed: List[int] = []

        for i, payload in enumerate(payloads):
            cd = corrected_deltas[i] if corrected_deltas is not None else None
            try:
                result = self.reconstruct(payload, cd)
                results.append(result)
            except ReconstructionError as exc:
                failed.append(i)
                print(
                    f"[Reconstructor] Skipping client {payload.client_id} "
                    f"(round {payload.round_idx}): {exc}"
                )

        return results, failed

    def reset(self) -> None:
        """
        Zero all pre-allocated buffers in-place.

        Called between independent experiment runs.  Does NOT deallocate
        the buffers — they are reused in the next experiment.
        """
        for buf in self._buffers.values():
            buf[:] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_buffer(self, client_id: int) -> np.ndarray:
        """
        Return the pre-allocated buffer for a client, creating it if needed.

        Allocated ONCE per client_id.  Subsequent calls return the same array.

        Parameters
        ----------
        client_id : int

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(total_params,)``.
        """
        if client_id not in self._buffers:
            self._buffers[client_id] = np.zeros(
                self.total_params, dtype=np.float32
            )
        return self._buffers[client_id]

    def _validate_payload(self, payload: SparsePayload) -> None:
        """
        Validate a SparsePayload before reconstruction.

        Parameters
        ----------
        payload : SparsePayload

        Raises
        ------
        ReconstructionError
            On any validation failure.
        """
        if not isinstance(payload, SparsePayload):
            raise ReconstructionError(
                f"Expected SparsePayload, got {type(payload).__name__}."
            )
        if payload.total_params != self.total_params:
            raise ReconstructionError(
                f"payload.total_params={payload.total_params} does not match "
                f"reconstructor.total_params={self.total_params}."
            )
        if payload.num_transmitted_params == 0:
            raise ReconstructionError(
                f"Empty payload from client {payload.client_id} "
                f"(round {payload.round_idx})."
            )
        if np.any(np.isnan(payload.values)):
            raise ReconstructionError(
                f"NaN values in payload from client {payload.client_id}."
            )
        if np.any(np.isinf(payload.values)):
            raise ReconstructionError(
                f"Inf values in payload from client {payload.client_id}."
            )
        if payload.indices.max() >= self.total_params:
            raise ReconstructionError(
                f"Out-of-bounds index {payload.indices.max()} >= "
                f"total_params={self.total_params}."
            )

    @staticmethod
    def _compute_reconstruction_error(
        corrected_delta: np.ndarray,
        reconstructed: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute the normalised L2 reconstruction error.

        error = ||corrected_delta - reconstructed||_2
              / max(eps, ||corrected_delta||_2)

        Parameters
        ----------
        corrected_delta : np.ndarray
            Ground-truth corrected delta (client-side).
        reconstructed : np.ndarray
            Dense zero-fill reconstruction from the payload.
        eps : float
            Stability epsilon for division.

        Returns
        -------
        float
            Value in [0, inf).  Ideal = 0 (lossless).
            For Top-K at 10%: expected ≈ sqrt(0.90) ≈ 0.949.
        """
        residual_norm = float(np.linalg.norm(corrected_delta - reconstructed, ord=2))
        delta_norm = float(np.linalg.norm(corrected_delta, ord=2))
        return residual_norm / max(eps, delta_norm)
