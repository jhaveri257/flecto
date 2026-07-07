"""
Federated/server/aggregator.py
================================
Federated Averaging (FedAvg) aggregation for the FL server.

Mathematical Formulation
------------------------
Let there be N participating clients in round t, each with:
    - W_local_i  : client i's locally-trained model weights
    - n_i        : number of local training samples
    - Delta_W_i  : reconstructed weight update = W_local_i - W_global(t)

FedAvg (McMahan et al., 2017) computes the new global model as::

    W_global(t+1) = Σᵢ (n_i / N_total) * W_local_i
                  = W_global(t) + Σᵢ (n_i / N_total) * Delta_W_i

where N_total = Σᵢ n_i.

Equivalently, with delta accumulation (our implementation)::

    aggregated_delta = Σᵢ (n_i * Delta_W_i) / Σᵢ n_i
    W_global(t+1)   = W_global(t) + aggregated_delta

Unweighted mode (equal client contribution)::

    aggregated_delta = (1/N) * Σᵢ Delta_W_i

Weighted mode is the default and the standard in FL literature because
clients with more local data should contribute more to the global model.

Algorithm Complexity
--------------------
- add_client_update():   O(P)       — one in-place += on P-length vector
- finalize_round():      O(P)       — one divide + one vector_to_state_dict
- Total per round (N):   O(N*P + P) = O(N*P)

where P = total_params.

Memory Complexity
-----------------
- One accumulator buffer: P * 4 bytes (float32)
- One output state_dict:  P * 4 bytes
- Peak:                   2P * 4 ≈ 855 KB for P=109,386

The accumulator is allocated ONCE and reused across all rounds.

Communication Complexity
------------------------
The Aggregator is entirely server-side.  Its memory usage is O(P) — the
same as storing the global model.  No additional inter-node communication
is required beyond what the Reconstructor has already processed.

Design Decisions
----------------
1. Numpy accumulation (REQUIREMENT 8):
   All operations use float32 numpy arrays.  In-place ``+=`` avoids heap
   allocation.  The accumulator is reset in-place (``fill(0.0)``).

2. API contract (REQUIREMENT 3):
   - add_client_update()  : accumulate one client at a time
   - aggregate()          : alias for finalize_round (backward-compat)
   - finalize_round()     : apply accumulated updates to global model,
                            return new state_dict, reset for next round
   - reset()              : zero without advancing round

3. NaN guard:
   If the aggregated delta contains NaN after accumulation, a ValueError
   is raised immediately with the count of NaN elements — rather than
   silently corrupting the global model.

4. Transport independence (REQUIREMENT 3):
   The Aggregator receives only dense numpy deltas.  It has zero
   dependency on SparsePayload, Compressor, or transport layers.

References
----------
McMahan, H.B. et al. (2017). "Communication-Efficient Learning of Deep
Networks from Decentralized Data." AISTATS 2017.
https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from Federated.config import WEIGHTED_FEDAVG
from Federated.model.model_utils import state_dict_to_vector, vector_to_state_dict


# ---------------------------------------------------------------------------
# AggregationResult — round-level summary
# ---------------------------------------------------------------------------

class AggregationResult:
    """
    Holds the output of one ``Aggregator.finalize_round()`` call.

    Parameters
    ----------
    new_state_dict : OrderedDict
        The new global model state_dict after FedAvg.
    aggregated_delta_norm : float
        L2 norm of the aggregated delta vector.  Monitors convergence:
        this should decrease as training progresses.
    num_clients_aggregated : int
        Number of client updates that were aggregated.
    total_weight : float
        Sum of client sample counts (or N for unweighted).
    agg_time_s : float
        Wall-clock aggregation time.
    weighted : bool
        Whether weighted FedAvg was applied.
    """

    __slots__ = (
        "new_state_dict", "aggregated_delta_norm",
        "num_clients_aggregated", "total_weight",
        "agg_time_s", "weighted",
    )

    def __init__(
        self,
        new_state_dict: dict,
        aggregated_delta_norm: float,
        num_clients_aggregated: int,
        total_weight: float,
        agg_time_s: float,
        weighted: bool,
    ) -> None:
        self.new_state_dict = new_state_dict
        self.aggregated_delta_norm = aggregated_delta_norm
        self.num_clients_aggregated = num_clients_aggregated
        self.total_weight = total_weight
        self.agg_time_s = agg_time_s
        self.weighted = weighted

    def to_dict(self) -> dict:
        """Serialise scalar metrics for CSV logging."""
        return {
            "aggregated_delta_norm": self.aggregated_delta_norm,
            "num_clients_aggregated": self.num_clients_aggregated,
            "total_weight": self.total_weight,
            "agg_time_s": self.agg_time_s,
            "weighted": self.weighted,
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class Aggregator:
    """
    Implements weighted and unweighted Federated Averaging (FedAvg).

    Client updates are accumulated incrementally via ``add_client_update()``.
    When all participating clients have submitted their updates,
    ``finalize_round()`` applies the weighted average to the global model
    and returns the new state_dict.

    The accumulator buffer is allocated ONCE and reused across all rounds
    to avoid repeated heap allocation.

    Parameters
    ----------
    total_params : int
        Number of scalar model parameters.
    weighted : bool, optional
        If ``True`` (default), use FedAvg weighted by client sample counts.
        If ``False``, all clients contribute equally regardless of shard size.

    Attributes
    ----------
    total_params : int
    weighted : bool
    num_updates : int
        Number of client updates added to the current round (read-only).
    total_weight : float
        Accumulated weight denominator for the current round (read-only).

    Examples
    --------
    >>> agg = Aggregator(total_params=109386, weighted=True)
    >>> agg.add_client_update(client_id=0, dense_delta=d0, num_samples=600)
    >>> agg.add_client_update(client_id=1, dense_delta=d1, num_samples=600)
    >>> result = agg.finalize_round(global_model.state_dict())
    >>> global_model.load_state_dict(result.new_state_dict)
    """

    def __init__(
        self,
        total_params: int,
        weighted: bool = WEIGHTED_FEDAVG,
    ) -> None:
        if total_params <= 0:
            raise ValueError(
                f"Aggregator: total_params must be > 0, got {total_params}."
            )
        self.total_params = total_params
        self.weighted = weighted

        # Pre-allocated accumulator: Σ(n_i * Delta_W_i) in weighted mode
        #                             Σ(Delta_W_i)       in unweighted mode
        self._accumulator = np.zeros(total_params, dtype=np.float32)

        # Metadata per round
        self._num_updates: int = 0
        self._total_weight: float = 0.0
        self._client_ids_this_round: List[int] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_updates(self) -> int:
        """Number of client updates in the current round."""
        return self._num_updates

    @property
    def total_weight(self) -> float:
        """Accumulated weight denominator (Σ n_i or N)."""
        return self._total_weight

    # ------------------------------------------------------------------
    # Primary API (REQUIREMENT 3)
    # ------------------------------------------------------------------

    def add_client_update(
        self,
        client_id: int,
        dense_delta: np.ndarray,
        num_samples: int,
    ) -> None:
        """
        Accumulate one client's reconstructed weight update.

        Performs an in-place weighted accumulation::

            accumulator += n_i * Delta_W_i   (weighted mode)
            accumulator += Delta_W_i          (unweighted mode)

        Parameters
        ----------
        client_id : int
            Client identifier (used for duplicate detection and logging).
        dense_delta : np.ndarray
            Dense reconstructed weight update, shape ``(total_params,)``,
            dtype float32.  Produced by ``Reconstructor.reconstruct()``.
        num_samples : int
            Number of local training samples for this client.  Used as the
            weight in weighted FedAvg.  Must be > 0.

        Raises
        ------
        ValueError
            If ``dense_delta`` has the wrong shape, wrong dtype, or contains
            NaN/Inf values.
        ValueError
            If ``num_samples <= 0``.
        ValueError
            If ``client_id`` has already been added in this round.

        Examples
        --------
        >>> agg.add_client_update(0, dense_delta=d0, num_samples=600)
        """
        # Validation (REQUIREMENT 7)
        if client_id in self._client_ids_this_round:
            raise ValueError(
                f"add_client_update: client {client_id} has already been "
                f"added in this round.  Call finalize_round() first."
            )
        if num_samples <= 0:
            raise ValueError(
                f"add_client_update: num_samples must be > 0, got {num_samples}."
            )
        self._validate_delta(dense_delta, name=f"client_{client_id}_delta")

        # Accumulate (in-place, REQUIREMENT 8)
        weight = float(num_samples) if self.weighted else 1.0
        np.add(self._accumulator, weight * dense_delta, out=self._accumulator)

        self._total_weight += weight
        self._num_updates += 1
        self._client_ids_this_round.append(client_id)

    def finalize_round(
        self,
        global_state_dict: dict,
    ) -> AggregationResult:
        """
        Compute the FedAvg update and apply it to the global model.

        Computes::

            aggregated_delta = accumulator / total_weight
            new_global_vec   = global_vec + aggregated_delta
            new_state_dict   = vector_to_state_dict(new_global_vec, reference)

        Then resets the accumulator for the next round.

        Parameters
        ----------
        global_state_dict : OrderedDict
            Current global model ``state_dict()``.  Used as both the base
            for the update and the reference architecture for tensor shapes.

        Returns
        -------
        AggregationResult
            Contains the new state_dict and aggregation diagnostics.

        Raises
        ------
        RuntimeError
            If no client updates have been added (cannot average zero clients).
        ValueError
            If the aggregated delta contains NaN or Inf after averaging.

        Examples
        --------
        >>> result = agg.finalize_round(global_model.state_dict())
        >>> global_model.load_state_dict(result.new_state_dict)
        """
        if self._num_updates == 0:
            raise RuntimeError(
                "finalize_round: no client updates have been added.  "
                "Call add_client_update() before finalize_round()."
            )

        t0 = time.perf_counter()

        # --- Weighted average of deltas ---
        aggregated_delta = (self._accumulator / self._total_weight).astype(np.float32)

        # NaN guard (REQUIREMENT 7)
        nan_count = int(np.isnan(aggregated_delta).sum())
        inf_count = int(np.isinf(aggregated_delta).sum())
        if nan_count > 0 or inf_count > 0:
            raise ValueError(
                f"finalize_round: aggregated delta contains {nan_count} NaN(s) "
                f"and {inf_count} Inf(s).  Check individual client updates."
            )

        # --- Apply delta to global model (in flat-vector space) ---
        global_vec = state_dict_to_vector(global_state_dict)  # float32 numpy
        new_global_vec = (global_vec + aggregated_delta).astype(np.float32)

        delta_norm = float(np.linalg.norm(aggregated_delta, ord=2))

        # --- Reconstruct new state_dict ---
        new_state_dict = vector_to_state_dict(new_global_vec, global_state_dict)

        agg_time_s = time.perf_counter() - t0

        n_clients = self._num_updates
        weight = self._total_weight

        # Reset for next round
        self._reset_buffers()

        return AggregationResult(
            new_state_dict=new_state_dict,
            aggregated_delta_norm=delta_norm,
            num_clients_aggregated=n_clients,
            total_weight=weight,
            agg_time_s=agg_time_s,
            weighted=self.weighted,
        )

    def aggregate(self, global_state_dict: dict) -> AggregationResult:
        """
        Alias for ``finalize_round()`` — provided for API completeness.

        Parameters
        ----------
        global_state_dict : OrderedDict

        Returns
        -------
        AggregationResult
        """
        return self.finalize_round(global_state_dict)

    def reset(self) -> None:
        """
        Zero the accumulator and clear round metadata without advancing.

        Use this to abort a partially-accumulated round (e.g., if too
        many clients failed and the round should be discarded entirely).
        """
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        """Zero the accumulator in-place and clear per-round state."""
        self._accumulator.fill(0.0)
        self._num_updates = 0
        self._total_weight = 0.0
        self._client_ids_this_round = []

    def _validate_delta(self, delta: np.ndarray, name: str = "delta") -> None:
        """
        Validate shape, dtype, and finiteness of a client delta.

        Parameters
        ----------
        delta : np.ndarray
        name : str

        Raises
        ------
        ValueError
        """
        if not isinstance(delta, np.ndarray):
            raise ValueError(
                f"Aggregator.add_client_update ({name}): "
                f"expected np.ndarray, got {type(delta).__name__}."
            )
        if delta.shape != (self.total_params,):
            raise ValueError(
                f"Aggregator.add_client_update ({name}): "
                f"expected shape ({self.total_params},), got {delta.shape}."
            )
        if delta.dtype != np.float32:
            raise ValueError(
                f"Aggregator.add_client_update ({name}): "
                f"expected float32, got {delta.dtype}."
            )
        nan_n = int(np.isnan(delta).sum())
        inf_n = int(np.isinf(delta).sum())
        if nan_n > 0 or inf_n > 0:
            raise ValueError(
                f"Aggregator.add_client_update ({name}): "
                f"delta contains {nan_n} NaN(s) and {inf_n} Inf(s)."
            )
