"""
Federated/simulation/fl_round_runner.py
=========================================
Complete Federated Learning communication round orchestrator.

Round Pipeline
--------------
Each communication round executes the following sequence::

    Global Model (W_global)
         |
         +---- sample M clients uniformly at random
         |
    [for each sampled client i]:
         |
         +-- clone global model
         |
         +-- LocalTrainer.train()  ->  W_local_i, metrics_i
         |
         +-- compute_delta()       ->  Delta_W_i = W_local_i - W_global
         |
         +-- ErrorFeedbackStore.apply_residual()  ->  Delta_W_corrected_i
         |
         +-- select_topk()         ->  UpdateSelection_i
         |
         +-- Compressor.compress() ->  SparsePayload_i
         |
         +-- ErrorFeedbackStore.update_residual()  (update r_i)
         |
    [server side]:
         |
         +-- Reconstructor.reconstruct(payload_i) -> dense_delta_hat_i
         |
         +-- Aggregator.add_client_update(dense_delta_hat_i, n_i)
         |
         +-- Aggregator.finalize_round()  ->  W_global(t+1)
         |
         +-- Evaluate on test set         ->  val_accuracy, val_loss
         |
         +-- Checkpoint if scheduled
         |
         +-- Log CSV metrics
         |
    Next Round

Communication Complexity
------------------------
Per round, N clients each transmit::

    payload_bytes_i = 16 + 12 * k_i   bytes

Total uplink per round:
    C_round = Σᵢ (16 + 12 * k_i)
            = N * (16 + 12 * K)       [when all clients have equal K]
            = N * (16 + 12 * floor(P * top_k_ratio))

Full (dense, uncompressed) baseline:
    C_dense = N * P * 4 bytes

Compression factor:
    f = C_dense / C_round
      = (N * P * 4) / (N * (16 + 12*K))
      ≈ P / (3K)  = 1 / (3 * top_k_ratio)

For top_k_ratio=0.10: f ≈ 3.33×  (matches the 3.33× from Milestone 3 tests)
For top_k_ratio=0.01: f ≈ 33.3×

Algorithm Complexity (per round)
---------------------------------
- Client local training: O(E * B * P)  where E=local_epochs, B=num_batches
- Delta computation:     O(P)  per client
- Top-K selection:       O(P + K*log(K))  per client
- Compression:           O(K)  per client
- Reconstruction:        O(K)  per client  (scatter)
- FedAvg:                O(N * P)
- Evaluation:            O(|test_set| * P / B_test)

Dominant term: local training  >>  compression  >>  aggregation

Memory Complexity
-----------------
- Global model:          P * 4 bytes
- Per-client buffer:     N * P * 4 bytes  (reconstructor + EF store)
- Aggregator:            P * 4 bytes
- Total simulation:      (2N + 3) * P * 4 bytes
  For N=10, P=109386: ≈ 9.4 MB

Design Decisions
----------------
1. Sequential client processing (not parallel):
   Simulates the federated setting without inter-process communication.
   Each client's step is fault-isolated with try/except.

2. Client sampling:
   Each round samples M = max(1, floor(N * sample_fraction)) clients
   uniformly without replacement.  The RNG is seeded with SEED + round_idx
   for deterministic reproducibility.

3. Minimum viable round:
   If fewer than 2 clients succeed (local training + compression), the
   round is skipped (aggregation is undefined for N=1 in research).

4. CSV metrics log:
   Appended one row per successful round.  File is created with header
   on the first write, then opened in append mode each subsequent round,
   so crash recovery never re-writes headers.

5. Best model tracking:
   The model that achieved the highest test accuracy so far is saved to
   ``CHECKPOINTS_DIR/best_model.pt``.  Uses float comparison with the
   running maximum.

References
----------
McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks
from Decentralized Data." AISTATS 2017. https://arxiv.org/abs/1602.05629
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Federated.client.compressor import Compressor
from Federated.client.local_trainer import LocalTrainer
from Federated.client.update_selector import compute_and_select
from Federated.config import (
    BATCH_SIZE,
    CHECKPOINT_EVERY_N_ROUNDS,
    CHECKPOINTS_DIR,
    CLIENT_SAMPLE_FRACTION,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    LOGS_DIR,
    MOMENTUM,
    NUM_CLIENTS,
    NUM_ROUNDS,
    SEED,
    TOP_K_RATIO,
    USE_ERROR_FEEDBACK,
    VERBOSITY,
    WEIGHTED_FEDAVG,
    WEIGHT_DECAY,
)
from Federated.model.fl_model import FLModel
from Federated.model.model_utils import (
    clone_model,
    copy_weights,
    get_flat_param_count,
    save_checkpoint,
)
from Federated.server.aggregator import Aggregator
from Federated.server.error_feedback_store import ErrorFeedbackStore
from Federated.server.reconstructor import ReconstructionError, Reconstructor
from Federated.transport_bridge.payload import CompressionStats, SparsePayload


# ---------------------------------------------------------------------------
# RoundMetrics — typed record for one complete round
# ---------------------------------------------------------------------------

class RoundMetrics:
    """
    Structured record of all metrics collected during one FL round.

    Provides a ``to_dict()`` method for CSV serialisation.

    Parameters
    ----------
    round_idx : int
    val_accuracy : float
    val_loss : float
    train_loss : float
        Mean training loss averaged over sampled clients.
    local_accuracy : float
        Mean local training accuracy over sampled clients.
    comm_bytes_total : int
        Total uplink bytes transmitted by all clients.
    avg_compression_ratio : float
        Mean compression ratio across sampled clients.
    avg_reconstruction_error : float
        Mean normalised L2 reconstruction error.
    aggregated_delta_norm : float
        L2 norm of the FedAvg-aggregated delta.
    round_time_s : float
        Total wall-clock time for the entire round.
    agg_time_s : float
        Wall-clock time for the FedAvg aggregation step.
    avg_recon_time_s : float
        Mean per-client reconstruction time.
    num_sampled_clients : int
        Number of clients sampled this round.
    num_successful_clients : int
        Number of clients that completed without error.
    """

    def __init__(
        self,
        round_idx: int,
        val_accuracy: float,
        val_loss: float,
        train_loss: float,
        local_accuracy: float,
        comm_bytes_total: int,
        avg_compression_ratio: float,
        avg_reconstruction_error: float,
        aggregated_delta_norm: float,
        round_time_s: float,
        agg_time_s: float,
        avg_recon_time_s: float,
        num_sampled_clients: int,
        num_successful_clients: int,
    ) -> None:
        self.round_idx = round_idx
        self.val_accuracy = val_accuracy
        self.val_loss = val_loss
        self.train_loss = train_loss
        self.local_accuracy = local_accuracy
        self.comm_bytes_total = comm_bytes_total
        self.avg_compression_ratio = avg_compression_ratio
        self.avg_reconstruction_error = avg_reconstruction_error
        self.aggregated_delta_norm = aggregated_delta_norm
        self.round_time_s = round_time_s
        self.agg_time_s = agg_time_s
        self.avg_recon_time_s = avg_recon_time_s
        self.num_sampled_clients = num_sampled_clients
        self.num_successful_clients = num_successful_clients

    @staticmethod
    def csv_header() -> List[str]:
        """Return the CSV column header list."""
        return [
            "round_idx",
            "val_accuracy",
            "val_loss",
            "train_loss",
            "local_accuracy",
            "comm_bytes_total",
            "avg_compression_ratio",
            "avg_reconstruction_error",
            "aggregated_delta_norm",
            "round_time_s",
            "agg_time_s",
            "avg_recon_time_s",
            "num_sampled_clients",
            "num_successful_clients",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dict suitable for ``csv.DictWriter``."""
        return {
            "round_idx": self.round_idx,
            "val_accuracy": f"{self.val_accuracy:.6f}",
            "val_loss": f"{self.val_loss:.6f}",
            "train_loss": f"{self.train_loss:.6f}",
            "local_accuracy": f"{self.local_accuracy:.6f}",
            "comm_bytes_total": self.comm_bytes_total,
            "avg_compression_ratio": f"{self.avg_compression_ratio:.6f}",
            "avg_reconstruction_error": f"{self.avg_reconstruction_error:.6f}",
            "aggregated_delta_norm": f"{self.aggregated_delta_norm:.6f}",
            "round_time_s": f"{self.round_time_s:.4f}",
            "agg_time_s": f"{self.agg_time_s:.4f}",
            "avg_recon_time_s": f"{self.avg_recon_time_s:.4f}",
            "num_sampled_clients": self.num_sampled_clients,
            "num_successful_clients": self.num_successful_clients,
        }


# ---------------------------------------------------------------------------
# ClientStepResult — typed result from one client's round step
# ---------------------------------------------------------------------------

class ClientStepResult:
    """Internal record from one client's training + compression step."""

    __slots__ = (
        "client_id", "payload", "corrected_delta",
        "dense_compressed", "train_loss", "local_accuracy",
        "train_time_s",
    )

    def __init__(
        self,
        client_id: int,
        payload: SparsePayload,
        corrected_delta: np.ndarray,
        dense_compressed: np.ndarray,
        train_loss: float,
        local_accuracy: float,
        train_time_s: float,
    ) -> None:
        self.client_id = client_id
        self.payload = payload
        self.corrected_delta = corrected_delta
        self.dense_compressed = dense_compressed
        self.train_loss = train_loss
        self.local_accuracy = local_accuracy
        self.train_time_s = train_time_s


# ---------------------------------------------------------------------------
# FLRoundRunner
# ---------------------------------------------------------------------------

class FLRoundRunner:
    """
    Orchestrates the complete Federated Learning simulation.

    Executes the full round pipeline: local training -> compression ->
    reconstruction -> FedAvg -> evaluation -> checkpointing -> CSV logging.

    Parameters
    ----------
    global_model : nn.Module
        The initial global model (modified in-place each round).
    client_loaders : list[DataLoader]
        One DataLoader per client, from ``build_partition()``.
    client_sizes : list[int]
        Number of local samples per client (for weighted FedAvg).
    test_loader : DataLoader or None
        Global test set loader.  Pass ``None`` to skip evaluation.
    device : torch.device, optional
        Computation device.  Defaults to CPU.
    top_k_ratio : float, optional
        Top-K compression ratio.  Default is ``config.TOP_K_RATIO``.
    use_error_feedback : bool, optional
        Enable error feedback residuals.  Default is ``config.USE_ERROR_FEEDBACK``.
    client_sample_fraction : float, optional
        Fraction of clients per round.  Default is ``config.CLIENT_SAMPLE_FRACTION``.
    weighted_fedavg : bool, optional
        Use weighted FedAvg.  Default is ``config.WEIGHTED_FEDAVG``.
    verbosity : int, optional
        0=silent, 1=per-round summary, 2=per-client detail.
    experiment_name : str, optional
        Prefix for CSV log filename.  Default is ``"fl_run"``.

    Examples
    --------
    >>> runner = FLRoundRunner(
    ...     global_model=model,
    ...     client_loaders=loaders,
    ...     client_sizes=sizes,
    ...     test_loader=test_loader,
    ... )
    >>> all_metrics = runner.run(num_rounds=10)
    >>> len(all_metrics)
    10
    """

    def __init__(
        self,
        global_model: nn.Module,
        client_loaders: List[DataLoader],
        client_sizes: List[int],
        test_loader: Optional[DataLoader],
        device: Optional[torch.device] = None,
        top_k_ratio: float = TOP_K_RATIO,
        use_error_feedback: bool = USE_ERROR_FEEDBACK,
        client_sample_fraction: float = CLIENT_SAMPLE_FRACTION,
        weighted_fedavg: bool = WEIGHTED_FEDAVG,
        verbosity: int = VERBOSITY,
        experiment_name: str = "fl_run",
    ) -> None:
        if len(client_loaders) != len(client_sizes):
            raise ValueError(
                "FLRoundRunner: client_loaders and client_sizes must have "
                "the same length."
            )

        self.global_model = global_model
        self.client_loaders = client_loaders
        self.client_sizes = client_sizes
        self.test_loader = test_loader
        self.device = device or torch.device("cpu")
        self.top_k_ratio = top_k_ratio
        self.use_error_feedback = use_error_feedback
        self.client_sample_fraction = client_sample_fraction
        self.weighted_fedavg = weighted_fedavg
        self.verbosity = verbosity
        self.experiment_name = experiment_name

        self.num_clients = len(client_loaders)
        self.total_params = get_flat_param_count(global_model)

        # --- Sub-components ---
        self._compressor = Compressor(self.total_params)
        self._reconstructor = Reconstructor(self.total_params, self.num_clients)
        self._aggregator = Aggregator(self.total_params, weighted=weighted_fedavg)
        self._ef_store = ErrorFeedbackStore(
            self.total_params,
            self.num_clients,
            enabled=use_error_feedback,
        )

        # Per-client LocalTrainer instances (stateless; reuse across rounds)
        self._trainers: Dict[int, LocalTrainer] = {
            cid: LocalTrainer(client_id=cid, device=self.device)
            for cid in range(self.num_clients)
        }

        # Checkpointing and logging state
        self._best_val_acc: float = -1.0
        self._csv_path: Optional[str] = None
        self._csv_header_written: bool = False

        # Ensure output directories exist
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, num_rounds: int = NUM_ROUNDS) -> List[RoundMetrics]:
        """
        Execute ``num_rounds`` communication rounds.

        Parameters
        ----------
        num_rounds : int, optional
            Number of FL communication rounds. Default is ``config.NUM_ROUNDS``.

        Returns
        -------
        list[RoundMetrics]
            One ``RoundMetrics`` object per successfully completed round.
            Skipped rounds (e.g., too many client failures) are omitted.
        """
        # Set up CSV log path using timestamp for uniqueness
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(
            LOGS_DIR, f"{self.experiment_name}_{ts}.csv"
        )
        self._csv_header_written = False

        all_metrics: List[RoundMetrics] = []

        if self.verbosity >= 1:
            ef_str = "ON" if self.use_error_feedback else "OFF"
            print(
                f"\n{'=' * 64}\n"
                f"  FL Simulation Start\n"
                f"  Rounds={num_rounds} | Clients={self.num_clients} | "
                f"TopK={self.top_k_ratio:.0%} | EF={ef_str}\n"
                f"  Params={self.total_params:,} | "
                f"Device={self.device} | Weighted={self.weighted_fedavg}\n"
                f"{'=' * 64}"
            )

        for round_idx in range(num_rounds):
            metrics = self._run_one_round(round_idx)
            if metrics is not None:
                all_metrics.append(metrics)
                self._log_round_metrics(metrics)
                if self.verbosity >= 1:
                    self._print_round_summary(metrics)

        if self.verbosity >= 1:
            print(f"\n{'=' * 64}")
            print(f"  Simulation Complete — {len(all_metrics)} rounds logged")
            print(f"  CSV: {self._csv_path}")
            print(f"  Best val_acc: {self._best_val_acc:.4f}")
            print(f"{'=' * 64}\n")

        return all_metrics

    # ------------------------------------------------------------------
    # Round orchestration
    # ------------------------------------------------------------------

    def _run_one_round(self, round_idx: int) -> Optional[RoundMetrics]:
        """
        Execute one complete FL communication round.

        Parameters
        ----------
        round_idx : int

        Returns
        -------
        RoundMetrics or None
            None if the round is skipped (< 2 clients succeeded).
        """
        t_round_start = time.perf_counter()

        # Sample participating clients (deterministic per round)
        sampled_ids = self._sample_clients(round_idx)
        global_sd = self.global_model.state_dict()

        # --- Client steps ---
        client_results: List[ClientStepResult] = []
        for cid in sampled_ids:
            try:
                result = self._client_step(cid, global_sd, round_idx)
                client_results.append(result)
                if self.verbosity >= 2:
                    print(
                        f"  [Round {round_idx:3d} | Client {cid:2d}] "
                        f"loss={result.train_loss:.4f} "
                        f"acc={result.local_accuracy:.3f} "
                        f"bytes={result.payload.payload_bytes}"
                    )
            except Exception as exc:  # fault tolerance (REQUIREMENT 7)
                print(
                    f"[WARNING] Round {round_idx}: client {cid} failed — "
                    f"{type(exc).__name__}: {exc}"
                )

        # Minimum viable round guard
        if len(client_results) < 2:
            print(
                f"[WARNING] Round {round_idx}: only {len(client_results)} "
                f"client(s) succeeded — round skipped."
            )
            return None

        # --- Server reconstruction ---
        payloads = [r.payload for r in client_results]
        corrected_deltas = [r.corrected_delta for r in client_results]

        recon_results, failed_idx = self._reconstructor.reconstruct_batch(
            payloads, corrected_deltas
        )

        # Remove failed reconstructions from client_results
        successful_ids = set(r.client_id for r in recon_results)
        client_results = [r for r in client_results if r.client_id in successful_ids]

        if len(client_results) < 2:
            print(
                f"[WARNING] Round {round_idx}: reconstruction failures "
                f"left < 2 clients — round skipped."
            )
            return None

        # --- FedAvg aggregation ---
        t_agg_start = time.perf_counter()
        self._aggregator.reset()  # ensure clean state
        for rr in recon_results:
            cid = rr.client_id
            n_samples = self.client_sizes[cid]
            try:
                self._aggregator.add_client_update(cid, rr.dense_delta, n_samples)
            except ValueError as exc:
                print(
                    f"[WARNING] Round {round_idx}: aggregation skipped "
                    f"client {cid} — {exc}"
                )

        if self._aggregator.num_updates < 2:
            self._aggregator.reset()
            print(f"[WARNING] Round {round_idx}: < 2 valid updates — round skipped.")
            return None

        agg_result = self._aggregator.finalize_round(global_sd)

        # Apply updated weights to global model
        self.global_model.load_state_dict(agg_result.new_state_dict)

        # --- Evaluation ---
        eval_results = self._evaluate_global_model()

        # --- Checkpointing (REQUIREMENT 5) ---
        self._maybe_checkpoint(round_idx, eval_results["val_accuracy"])

        # --- Aggregate round-level metrics (REQUIREMENT 6) ---
        round_time_s = time.perf_counter() - t_round_start
        avg_recon_time = float(np.mean([r.recon_time_s for r in recon_results]))
        comm_bytes = sum(r.payload.payload_bytes for r in client_results)
        avg_ratio = float(np.mean([r.payload.compression_ratio for r in client_results]))
        avg_recon_err = float(np.nanmean([r.reconstruction_error for r in recon_results]))
        mean_train_loss = float(np.mean([r.train_loss for r in client_results]))
        mean_local_acc = float(np.mean([r.local_accuracy for r in client_results]))

        return RoundMetrics(
            round_idx=round_idx,
            val_accuracy=eval_results["val_accuracy"],
            val_loss=eval_results["val_loss"],
            train_loss=mean_train_loss,
            local_accuracy=mean_local_acc,
            comm_bytes_total=comm_bytes,
            avg_compression_ratio=avg_ratio,
            avg_reconstruction_error=avg_recon_err,
            aggregated_delta_norm=agg_result.aggregated_delta_norm,
            round_time_s=round_time_s,
            agg_time_s=agg_result.agg_time_s,
            avg_recon_time_s=avg_recon_time,
            num_sampled_clients=len(sampled_ids),
            num_successful_clients=len(client_results),
        )

    # ------------------------------------------------------------------
    # Client step
    # ------------------------------------------------------------------

    def _client_step(
        self,
        client_id: int,
        global_sd: dict,
        round_idx: int,
    ) -> ClientStepResult:
        """
        Execute one client's local training, compression, and residual update.

        Parameters
        ----------
        client_id : int
        global_sd : dict
            Current global model state_dict (shared read-only reference).
        round_idx : int

        Returns
        -------
        ClientStepResult

        Raises
        ------
        Any exception from LocalTrainer, select_topk, or Compressor is
        propagated to ``_run_one_round`` for fault-tolerant handling.
        """
        # Clone global model for local training
        local_model = clone_model(self.global_model)

        # Local training
        trainer = self._trainers[client_id]
        train_metrics = trainer.train(local_model, self.client_loaders[client_id])

        # Error feedback: get residual from previous round
        ef_residual = None
        if self.use_error_feedback:
            ef_residual = self._ef_store.get_residual(client_id).copy()

        # Compute delta + Top-K selection
        selection, corrected_delta = compute_and_select(
            local_state_dict=train_metrics["state_dict"],
            global_state_dict=global_sd,
            top_k_ratio=self.top_k_ratio,
            error_feedback=ef_residual,
        )

        # Compress to SparsePayload
        payload = self._compressor.compress(
            selection,
            client_id=client_id,
            round_idx=round_idx,
            extra_metadata={"error_feedback_applied": self.use_error_feedback},
        )

        # Reconstruct dense compressed update (for residual computation)
        dense_compressed = np.zeros(self.total_params, dtype=np.float32)
        dense_compressed[selection.indices] = selection.values

        # Update error feedback residual: r(t+1) = corrected_delta - compressed
        self._ef_store.update_residual(
            client_id, corrected_delta, dense_compressed, round_idx
        )

        return ClientStepResult(
            client_id=client_id,
            payload=payload,
            corrected_delta=corrected_delta,
            dense_compressed=dense_compressed,
            train_loss=train_metrics["loss"],
            local_accuracy=train_metrics["accuracy"],
            train_time_s=train_metrics["train_time_s"],
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate the global model on the test set.

        Returns
        -------
        dict
            Keys: ``"val_accuracy"`` (float), ``"val_loss"`` (float).
            Both are ``float('nan')`` when ``test_loader`` is None.
        """
        if self.test_loader is None:
            return {"val_accuracy": float("nan"), "val_loss": float("nan")}

        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.global_model(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.global_model.train()

        val_accuracy = correct / max(1, total)
        val_loss = total_loss / max(1, n_batches)
        return {"val_accuracy": val_accuracy, "val_loss": val_loss}

    # ------------------------------------------------------------------
    # Checkpointing (REQUIREMENT 5)
    # ------------------------------------------------------------------

    def _maybe_checkpoint(self, round_idx: int, val_accuracy: float) -> None:
        """
        Save periodic and best-model checkpoints.

        Saves to ``CHECKPOINTS_DIR``:
        - Periodic: every ``CHECKPOINT_EVERY_N_ROUNDS`` rounds.
        - Best model: whenever ``val_accuracy > self._best_val_acc``.
        """
        # Periodic checkpoint
        if CHECKPOINT_EVERY_N_ROUNDS > 0 and (round_idx + 1) % CHECKPOINT_EVERY_N_ROUNDS == 0:
            ckpt_path = save_checkpoint(
                self.global_model,
                round_idx=round_idx,
                models_dir=CHECKPOINTS_DIR,
                extra_meta={"val_accuracy": val_accuracy},
            )
            if self.verbosity >= 2:
                print(f"  [Checkpoint] Round {round_idx}: saved {os.path.basename(ckpt_path)}")

        # Best model checkpoint
        if not np.isnan(val_accuracy) and val_accuracy > self._best_val_acc:
            self._best_val_acc = val_accuracy
            best_path = save_checkpoint(
                self.global_model,
                round_idx=round_idx,
                filename="best_model.pt",
                models_dir=CHECKPOINTS_DIR,
                extra_meta={"val_accuracy": val_accuracy},
            )
            if self.verbosity >= 1:
                print(
                    f"  [Best Model] Round {round_idx}: "
                    f"val_acc={val_accuracy:.4f} -> {os.path.basename(best_path)}"
                )

    # ------------------------------------------------------------------
    # CSV logging (REQUIREMENT 6)
    # ------------------------------------------------------------------

    def _log_round_metrics(self, metrics: RoundMetrics) -> None:
        """
        Append one row to the CSV metrics log.

        Creates the file with header on first call; appends on subsequent
        calls so crash recovery never duplicates the header.

        Parameters
        ----------
        metrics : RoundMetrics
        """
        if self._csv_path is None:
            return

        write_header = not self._csv_header_written

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RoundMetrics.csv_header())
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(metrics.to_dict())

    # ------------------------------------------------------------------
    # Client sampling
    # ------------------------------------------------------------------

    def _sample_clients(self, round_idx: int) -> List[int]:
        """
        Sample M clients uniformly at random without replacement.

        Uses a round-specific RNG seed (``SEED + round_idx``) for
        deterministic reproducibility across independent runs.

        Parameters
        ----------
        round_idx : int

        Returns
        -------
        list[int]
            Sorted list of sampled client IDs.
        """
        m = max(1, int(self.num_clients * self.client_sample_fraction))
        m = min(m, self.num_clients)
        rng = np.random.default_rng(SEED + round_idx)
        sampled = rng.choice(self.num_clients, size=m, replace=False)
        return sorted(sampled.tolist())

    # ------------------------------------------------------------------
    # Verbosity
    # ------------------------------------------------------------------

    def _print_round_summary(self, metrics: RoundMetrics) -> None:
        """Print a one-line round summary to stdout."""
        print(
            f"  Round {metrics.round_idx:3d}/{metrics.round_idx} | "
            f"val_acc={metrics.val_accuracy:.4f} | "
            f"val_loss={metrics.val_loss:.4f} | "
            f"train_loss={metrics.train_loss:.4f} | "
            f"bytes={metrics.comm_bytes_total:,} | "
            f"ratio={metrics.avg_compression_ratio:.3f} | "
            f"recon_err={metrics.avg_reconstruction_error:.3f} | "
            f"time={metrics.round_time_s:.2f}s"
        )


if __name__ == "__main__":
    import json
    import os
    import time
    import traceback
    
    import torch
    
    from Federated.config import (
        set_seed, NUM_CLIENTS, NUM_ROUNDS, TOP_K_RATIO, USE_ERROR_FEEDBACK,
        PARTITION_MODE, LOGS_DIR, CHECKPOINTS_DIR, SEED, BATCH_SIZE
    )
    from Federated.model.fl_model import FLModel
    from Federated.simulation.data_partitioner import (
        load_mnist, build_partition, make_test_loader
    )

    try:
        print("=" * 50)
        print("Federated Learning Simulation")
        print("=" * 50)
        print("Dataset: MNIST")
        print(f"Clients: {NUM_CLIENTS}")
        print(f"Rounds: {NUM_ROUNDS}")
        print(f"Top-K Ratio: {TOP_K_RATIO * 100:.0f}%")
        print(f"Error Feedback: {'Enabled' if USE_ERROR_FEEDBACK else 'Disabled'}")
        print("=" * 50)

        # 1. Set seed
        set_seed(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Download/load MNIST dataset
        train_ds, test_ds = load_mnist()

        # 3. Partition dataset
        client_loaders, client_sizes, _ = build_partition(
            train_ds, mode=PARTITION_MODE, num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE
        )
        test_loader = make_test_loader(test_ds)

        # 4. Create global FLModel
        global_model = FLModel().to(device)

        # 5. Instantiate FLRoundRunner
        runner = FLRoundRunner(
            global_model=global_model,
            client_loaders=client_loaders,
            client_sizes=client_sizes,
            test_loader=test_loader,
            device=device,
            top_k_ratio=TOP_K_RATIO,
            use_error_feedback=USE_ERROR_FEEDBACK,
            verbosity=1,
            experiment_name="cli_run"
        )

        # 6. Execute runner
        metrics = runner.run(num_rounds=NUM_ROUNDS)
        
        if not metrics:
            print("Simulation skipped all rounds or failed.")
            exit(1)

        # Compute reduction
        baseline_bytes = sum(m.num_sampled_clients for m in metrics) * runner.total_params * 4
        actual_bytes = sum(m.comm_bytes_total for m in metrics)
        comm_reduction = ((baseline_bytes - actual_bytes) / baseline_bytes) * 100 if baseline_bytes > 0 else 0.0
        
        best_acc = max(m.val_accuracy for m in metrics)
        final_acc = metrics[-1].val_accuracy

        # 7. Print summary
        print("=" * 50)
        print("Simulation Complete")
        print(f"Best Accuracy: {best_acc:.4f}")
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Communication Reduction: {comm_reduction:.2f}%")
        print(f"Checkpoint Location: {os.path.abspath(CHECKPOINTS_DIR)}")
        print(f"CSV Location: {os.path.abspath(runner._csv_path)}")
        print("=" * 50)

        # 8. Save experiment config
        config_data = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "top_k_ratio": TOP_K_RATIO,
            "use_error_feedback": USE_ERROR_FEEDBACK,
            "partition_mode": PARTITION_MODE,
            "seed": SEED
        }
        os.makedirs(LOGS_DIR, exist_ok=True)
        config_path = os.path.join(LOGS_DIR, "experiment_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"Simulation Failed: {type(e).__name__}")
        print(str(e))
        print("=" * 50)
        traceback.print_exc()
        exit(1)
