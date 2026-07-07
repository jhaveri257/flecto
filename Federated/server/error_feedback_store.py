"""
Federated/server/error_feedback_store.py
==========================================
Per-client error feedback residual store for communication-efficient FL.

Mathematical Formulation
------------------------
Error feedback (also known as "gradient memory" or "error correction")
accumulates the compression error from each round so that information
discarded by Top-K in round t is not permanently lost — it is added back
to the raw delta in round t+1 before the next Top-K selection.

Let:
    - t           : communication round index
    - Delta_W(t)  : raw weight delta for a client in round t
                    = W_local(t) - W_global(t)
    - r(t)        : residual vector for this client at the START of round t
                    (initialised to zero vector at t=0)
    - C(x)        : compression operator (Top-K applied to x)

Round-t pipeline with error feedback::

    (1) Corrected delta:
            delta_corrected(t) = Delta_W(t) + r(t)

    (2) Top-K selection:
            compressed(t) = C(delta_corrected(t))
                          = Top-K(|delta_corrected(t)|) applied to delta_corrected(t)

    (3) Residual update (stored for use in round t+1):
            r(t+1) = delta_corrected(t) - compressed(t)

The key insight is that the residual r captures the information discarded
by Top-K.  Over multiple rounds, every parameter update eventually gets
transmitted — small-but-persistent gradients accumulate in r until their
magnitude crosses the Top-K selection threshold.

This is identical to the "error feedback SGD" scheme of Stich et al. (2018),
adapted for the Federated Learning communication-round context.

Convergence Guarantee
---------------------
Under standard smoothness and bounded-gradient assumptions, error feedback
Top-K SGD converges at the same asymptotic rate as full-gradient SGD.
Without error feedback, Top-K SGD can stagnate for small ratios (< 5%).
See Stich et al. (2018), Theorem 2 for the formal bound.

Design Decisions
----------------
1. One-time allocation (REQUIREMENT 5):
   Each client's residual vector is allocated exactly ONCE via
   ``np.zeros(total_params, dtype=np.float32)`` on the first call.
   Subsequent rounds update the buffer IN-PLACE using numpy operations
   (``r[:] = delta_corrected - compressed``), avoiding repeated heap
   allocation.  This is critical for low-latency simulation with many
   clients and rounds.

2. Per-client isolation:
   Residuals are stored in a dict keyed by ``client_id``.  Each client
   owns its own independent residual vector.  There is no cross-client
   sharing or contamination.

3. float32 throughout (REQUIREMENT 6):
   All operations preserve float32.  The in-place update
   ``r[:] = delta_corrected - compressed`` is performed in float32.
   NumPy would silently upcast to float64 if either input is float64,
   so both inputs are explicitly cast before subtraction.

4. Lazy initialisation:
   Residuals are only allocated when a client first interacts with the
   store.  This avoids allocating P × 4 bytes for every possible client
   at startup — important when NUM_CLIENTS is large and only a fraction
   participate each round.

5. Round tracking:
   The store records, per client, the last round in which the residual
   was updated.  This enables detection of "skipped" rounds (a client
   that did not participate) and supports partial-participation FL
   experiments.

Algorithm Complexity
--------------------
- apply_residual():   O(P) — one vector addition
- update_residual():  O(P) — one vector subtraction + one in-place write
- reset():            O(P) — one numpy zeros fill
- get_residual():     O(1) — dict lookup

where P = total_params (e.g., 109,386 for FLModel).

Memory Complexity
-----------------
- Per client: P × 4 bytes (float32 residual vector)
              = 109,386 × 4 ≈ 427 KB per client
- For N clients: N × 427 KB  (e.g., 10 clients ≈ 4.27 MB total)

This is fully acceptable for simulation.  In a real deployment, residuals
would be stored on the client device, not the server.

References
----------
Stich, S.U. (2018). "Sparsified SGD with Memory." NeurIPS 2018.
https://arxiv.org/abs/1809.10767

Lin, Y. et al. (2018). "Deep Gradient Compression: Reducing the
Communication Bandwidth for Distributed Training." ICLR 2018.
https://arxiv.org/abs/1712.01887

Karimireddy, S.P. et al. (2019). "Error Feedback Fixes SignSGD and other
Gradient Compression Schemes." ICML 2019.
https://arxiv.org/abs/1901.09847
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from Federated.config import NUM_CLIENTS, USE_ERROR_FEEDBACK


# ===========================================================================
# ErrorFeedbackStore
# ===========================================================================

class ErrorFeedbackStore:
    """
    Manages per-client error-feedback residual vectors across FL rounds.

    Each client owns one residual vector of shape ``(total_params,)`` in
    float32.  The vector is allocated lazily on first access and updated
    in-place every round to avoid heap allocation overhead.

    The store is designed to be held by the FL round runner and passed
    to the compression pipeline each round.  It is NOT thread-safe: if
    clients are processed in parallel, use one store per worker process
    or add external locking.

    Parameters
    ----------
    total_params : int
        Number of scalar parameters in the flattened model vector.
        All residual buffers will have this length.
    num_clients : int, optional
        Expected number of clients.  Used only for validation in
        ``register_clients()`` and logging.  Default is ``NUM_CLIENTS``.
    enabled : bool, optional
        Master switch for error feedback.  When ``False``, all methods
        operate as if residuals are permanently zero — equivalent to
        standard Top-K without memory.  Default is ``USE_ERROR_FEEDBACK``.

    Attributes
    ----------
    total_params : int
    num_clients : int
    enabled : bool
    _residuals : dict[int, np.ndarray]
        Internal residual buffer dictionary keyed by client_id.
    _round_updated : dict[int, int]
        Maps client_id to the last round in which its residual was updated.
    _update_count : dict[int, int]
        Counts how many times each client's residual has been updated
        (useful for detecting dropped clients in non-full-participation FL).

    Examples
    --------
    >>> store = ErrorFeedbackStore(total_params=109386, num_clients=10)
    >>> delta = np.random.randn(109386).astype(np.float32)
    >>> corrected = store.apply_residual(client_id=0, delta=delta)
    >>> # ... run Top-K, compress ...
    >>> compressed = np.zeros(109386, dtype=np.float32)
    >>> compressed[sel.indices] = sel.values
    >>> store.update_residual(client_id=0, corrected_delta=corrected,
    ...                        compressed=compressed, round_idx=0)
    """

    def __init__(
        self,
        total_params: int,
        num_clients: int = NUM_CLIENTS,
        enabled: bool = USE_ERROR_FEEDBACK,
    ) -> None:
        if total_params <= 0:
            raise ValueError(
                f"ErrorFeedbackStore: total_params must be > 0, got {total_params}."
            )
        if num_clients <= 0:
            raise ValueError(
                f"ErrorFeedbackStore: num_clients must be > 0, got {num_clients}."
            )

        self.total_params = total_params
        self.num_clients = num_clients
        self.enabled = enabled

        # Residual buffers — allocated lazily on first client access
        self._residuals: Dict[int, np.ndarray] = {}

        # Metadata for diagnostics and convergence monitoring
        self._round_updated: Dict[int, int] = {}   # client_id -> last round
        self._update_count: Dict[int, int] = {}    # client_id -> total updates

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def apply_residual(
        self,
        client_id: int,
        delta: np.ndarray,
    ) -> np.ndarray:
        """
        Add the stored residual to a raw delta to produce the corrected delta.

        Implements step (1) of the error feedback pipeline::

            delta_corrected(t) = Delta_W(t) + r(t)

        When ``enabled=False``, returns ``delta`` unchanged (no copy).

        Parameters
        ----------
        client_id : int
            Client identifier.  If not yet seen, a zero residual is
            lazily allocated and returned (equivalent to no correction
            in the first round).
        delta : np.ndarray
            Float32 vector of shape ``(total_params,)`` — the raw weight
            delta Delta_W(t) = W_local - W_global for this round.

        Returns
        -------
        np.ndarray
            Float32 vector of shape ``(total_params,)`` — the corrected
            delta ``delta + r(t)``.  This is the vector that must be
            passed to Top-K selection.

        Raises
        ------
        ValueError
            If ``delta.shape != (total_params,)`` or ``delta`` contains
            NaN or Inf values.

        Notes
        -----
        The return value is a NEW array (not in-place on ``delta`` or ``r``).
        This is intentional: the caller needs both the corrected delta and
        the raw delta for the residual update.

        Examples
        --------
        >>> corrected = store.apply_residual(client_id=3, delta=raw_delta)
        >>> corrected.dtype
        dtype('float32')
        """
        self._validate_delta(delta, "apply_residual")

        if not self.enabled:
            return delta  # no copy — caller must not mutate

        residual = self._get_or_create(client_id)

        # Explicit float32: prevents silent upcast if residual somehow drifts
        corrected = (delta + residual).astype(np.float32)
        return corrected

    def update_residual(
        self,
        client_id: int,
        corrected_delta: np.ndarray,
        compressed: np.ndarray,
        round_idx: int,
    ) -> None:
        """
        Update the stored residual after a compression step.

        Implements step (3) of the error feedback pipeline::

            r(t+1) = delta_corrected(t) - compressed(t)

        The update is performed IN-PLACE on the stored buffer to avoid
        allocating a new float32 array of size P each round.

        Parameters
        ----------
        client_id : int
            Client identifier.
        corrected_delta : np.ndarray
            The corrected delta vector as returned by ``apply_residual()``
            — i.e., ``Delta_W(t) + r(t)``.  Shape ``(total_params,)``.
        compressed : np.ndarray
            The DENSE reconstruction of the compressed update — i.e., the
            zero-filled vector with Top-K values scattered at their indices.
            Shape ``(total_params,)``.  Produced by ``Compressor.decompress()``.
        round_idx : int
            Current communication round.  Stored for diagnostics.

        Returns
        -------
        None
            The residual buffer is updated in-place.

        Raises
        ------
        ValueError
            If either input has the wrong shape, wrong dtype, or contains
            NaN or Inf values.

        Notes
        -----
        After this call, ``r(t+1) = corrected_delta - compressed`` is
        stored in the buffer.  In round t+1, ``apply_residual()`` will
        add this residual to the new raw delta.

        Examples
        --------
        >>> # dense_compressed is the output of Compressor.decompress(payload)
        >>> store.update_residual(
        ...     client_id=0,
        ...     corrected_delta=corrected,
        ...     compressed=dense_compressed,
        ...     round_idx=3,
        ... )
        """
        if not self.enabled:
            return  # no-op: residuals disabled

        self._validate_delta(corrected_delta, "update_residual:corrected_delta")
        self._validate_delta(compressed, "update_residual:compressed")

        if corrected_delta.shape != compressed.shape:
            raise ValueError(
                f"update_residual: corrected_delta.shape={corrected_delta.shape} "
                f"!= compressed.shape={compressed.shape}."
            )

        residual = self._get_or_create(client_id)

        # In-place subtraction: r[:] = corrected_delta - compressed
        # numpy in-place subtraction preserves the buffer identity (no realloc)
        np.subtract(corrected_delta, compressed, out=residual)

        # Clamp residuals to float32 representable range to prevent overflow
        np.clip(residual, -3.4e38, 3.4e38, out=residual)

        # Diagnostics
        self._round_updated[client_id] = round_idx
        self._update_count[client_id] = self._update_count.get(client_id, 0) + 1

    def get_residual(self, client_id: int) -> np.ndarray:
        """
        Return the current residual vector for a client (read-only view).

        Parameters
        ----------
        client_id : int
            Client identifier.

        Returns
        -------
        np.ndarray
            Float32 vector of shape ``(total_params,)``.
            Returns a zero vector if the client has never been seen.
            The returned array is the INTERNAL buffer — do not modify it.

        Examples
        --------
        >>> r = store.get_residual(client_id=5)
        >>> r.shape
        (109386,)
        >>> r.dtype
        dtype('float32')
        """
        return self._get_or_create(client_id)

    def reset_client(self, client_id: int) -> None:
        """
        Zero the residual vector for a specific client in-place.

        Use this to simulate a client rejoining the federation after an
        absence (its stale residual should be discarded).

        Parameters
        ----------
        client_id : int
            Client to reset.

        Returns
        -------
        None

        Examples
        --------
        >>> store.reset_client(client_id=2)
        >>> np.all(store.get_residual(2) == 0)
        True
        """
        if client_id in self._residuals:
            self._residuals[client_id][:] = 0.0
            self._round_updated.pop(client_id, None)
            self._update_count.pop(client_id, None)

    def reset_all(self) -> None:
        """
        Zero all residual buffers in-place.

        Called between independent experiment runs to ensure clean state
        without reallocating the buffers.

        Returns
        -------
        None

        Examples
        --------
        >>> store.reset_all()
        >>> all(np.all(store.get_residual(i) == 0) for i in range(10))
        True
        """
        for buf in self._residuals.values():
            buf[:] = 0.0
        self._round_updated.clear()
        self._update_count.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def residual_norms(self) -> Dict[int, float]:
        """
        Compute the L2 norm of each stored residual vector.

        Useful for monitoring residual magnitude over rounds to detect
        convergence or divergence of the error feedback mechanism.

        Returns
        -------
        dict[int, float]
            Mapping of ``{client_id: l2_norm}``.  Clients with zero
            residuals (or not yet seen) are not included.

        Examples
        --------
        >>> norms = store.residual_norms()
        >>> norms[0]
        0.041...
        """
        return {
            cid: float(np.linalg.norm(r, ord=2))
            for cid, r in self._residuals.items()
        }

    def residual_summary(self, round_idx: Optional[int] = None) -> str:
        """
        Return a one-line diagnostic string summarising all residuals.

        Parameters
        ----------
        round_idx : int, optional
            If provided, prepended to the output.

        Returns
        -------
        str
            Human-readable summary of residual norms and client count.

        Examples
        --------
        >>> print(store.residual_summary(round_idx=5))
        [Round 5] EF residuals: 10 clients | mean_norm=0.0312 | max_norm=0.0871
        """
        norms = self.residual_norms()
        if not norms:
            prefix = f"[Round {round_idx}] " if round_idx is not None else ""
            return f"{prefix}EF residuals: no clients registered"

        values = list(norms.values())
        mean_norm = float(np.mean(values))
        max_norm = float(np.max(values))
        prefix = f"[Round {round_idx}] " if round_idx is not None else ""
        return (
            f"{prefix}EF residuals: {len(norms)} clients | "
            f"mean_norm={mean_norm:.4f} | max_norm={max_norm:.4f}"
        )

    def memory_usage_bytes(self) -> int:
        """
        Return total bytes used by all residual buffers.

        Returns
        -------
        int
            ``num_registered_clients * total_params * 4``.

        Examples
        --------
        >>> store.memory_usage_bytes()
        4375440  # 10 clients * 109386 * 4
        """
        return len(self._residuals) * self.total_params * 4

    @property
    def num_registered(self) -> int:
        """Number of clients with allocated residual buffers."""
        return len(self._residuals)

    def client_stats(self, client_id: int) -> Dict[str, object]:
        """
        Return per-client diagnostic information.

        Parameters
        ----------
        client_id : int
            Target client.

        Returns
        -------
        dict
            Keys: ``client_id``, ``registered``, ``last_round``,
            ``update_count``, ``residual_norm``.
        """
        registered = client_id in self._residuals
        return {
            "client_id": client_id,
            "registered": registered,
            "last_round": self._round_updated.get(client_id, None),
            "update_count": self._update_count.get(client_id, 0),
            "residual_norm": (
                float(np.linalg.norm(self._residuals[client_id], ord=2))
                if registered else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, client_id: int) -> np.ndarray:
        """
        Return the residual buffer for a client, allocating it if necessary.

        The buffer is allocated ONCE (REQUIREMENT 5).  All subsequent
        accesses return the same numpy array object.

        Parameters
        ----------
        client_id : int

        Returns
        -------
        np.ndarray
            Float32 zero-initialised buffer of shape ``(total_params,)``.
        """
        if client_id not in self._residuals:
            # One-time allocation — NOT recreated each round
            self._residuals[client_id] = np.zeros(
                self.total_params, dtype=np.float32
            )
        return self._residuals[client_id]

    def _validate_delta(self, arr: np.ndarray, name: str) -> None:
        """
        Validate shape, dtype, and finiteness of an input array.

        Parameters
        ----------
        arr : np.ndarray
        name : str

        Raises
        ------
        TypeError
            If ``arr`` is not an ndarray.
        ValueError
            On shape mismatch, wrong dtype, or NaN/Inf values.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"ErrorFeedbackStore.{name}: expected np.ndarray, "
                f"got {type(arr).__name__}."
            )
        if arr.shape != (self.total_params,):
            raise ValueError(
                f"ErrorFeedbackStore.{name}: expected shape ({self.total_params},), "
                f"got {arr.shape}."
            )
        if arr.dtype != np.float32:
            raise ValueError(
                f"ErrorFeedbackStore.{name}: expected float32, got {arr.dtype}. "
                f"Cast to float32 before passing."
            )
        nan_n = int(np.isnan(arr).sum())
        inf_n = int(np.isinf(arr).sum())
        if nan_n > 0 or inf_n > 0:
            raise ValueError(
                f"ErrorFeedbackStore.{name}: array contains {nan_n} NaN(s) "
                f"and {inf_n} Inf(s)."
            )
