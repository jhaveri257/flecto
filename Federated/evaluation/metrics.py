"""
Federated/evaluation/metrics.py
================================
Statistical aggregation and metric computation for FL evaluation.

Purpose
-------
Computes derived metrics (e.g., Bytes Saved, Percentage Reduction)
and statistical aggregates (Mean, Std Dev, 95% CI) across multiple
independent runs of an experiment mode.

Design Decisions
----------------
1. Independent from simulation state: Consumes plain dictionaries/lists
   to remain decoupled from the active training loop.
2. SciPy dependency avoided: Uses standard NumPy for means, standard
   deviations, and approximate 95% CIs (using 1.96 multiplier) to minimize
   heavy dependencies.

Algorithm Complexity
--------------------
O(R * T) where R is the number of runs and T is the number of rounds.
Since R <= 10 and T <= 100 typically, this is negligible O(1) in practice.

Memory Complexity
-----------------
O(R * T) to hold the raw metrics in memory before aggregation.

Output Files Generated
----------------------
None directly (used by evaluator.py to write CSVs).

Research Significance
---------------------
Proper statistical aggregation across multiple random seeds is mandatory
in ML research to prove that algorithmic gains (like Error Feedback) are
statistically significant and not artifacts of a "lucky" initialization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Tuple

import numpy as np


class MetricsAggregator:
    """
    Computes statistical aggregates across multiple FL simulation runs.

    Supports computation of mean, standard deviation, and 95% Confidence
    Intervals for per-round time-series data.
    """

    @staticmethod
    def aggregate_runs(runs: List[List[Dict[str, Any]]]) -> List[Dict[str, float]]:
        """
        Aggregate a list of runs (each a list of round metrics) into a
        single list of averaged round metrics with std devs.

        Parameters
        ----------
        runs : list of list of dict
            ``runs[i][t]`` is the metrics dict for run ``i``, round ``t``.
            All runs must have the same number of rounds.

        Returns
        -------
        list of dict
            Aggregated metrics per round. Keys will have ``_mean`` and
            ``_std`` suffixes, along with ``_ci95``.
        """
        if not runs:
            return []

        num_runs = len(runs)
        num_rounds = len(runs[0])

        aggregated_rounds: List[Dict[str, float]] = []

        for t in range(num_rounds):
            round_data: Dict[str, List[float]] = {}
            # Collect data for round t across all runs
            for run_idx in range(num_runs):
                metrics = runs[run_idx][t]
                for k, v in metrics.items():
                    if k == "round_idx":
                        continue
                    try:
                        float_v = float(v)
                        if k not in round_data:
                            round_data[k] = []
                        round_data[k].append(float_v)
                    except (ValueError, TypeError):
                        pass

            # Compute statistics
            agg_metrics: Dict[str, float] = {"round_idx": float(t)}
            for k, values in round_data.items():
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if num_runs > 1 else 0.0
                ci95_val = 1.96 * (std_val / math.sqrt(num_runs)) if num_runs > 1 else 0.0

                agg_metrics[f"{k}_mean"] = mean_val
                agg_metrics[f"{k}_std"] = std_val
                agg_metrics[f"{k}_ci95"] = ci95_val

            aggregated_rounds.append(agg_metrics)

        return aggregated_rounds

    @staticmethod
    def compute_communication_savings(
        compressed_bytes: int,
        uncompressed_bytes: int
    ) -> Tuple[int, float]:
        """
        Compute total bytes saved and percentage reduction.

        Parameters
        ----------
        compressed_bytes : int
            Total bytes transmitted in the compressed regime.
        uncompressed_bytes : int
            Total bytes that would have been transmitted (baseline).

        Returns
        -------
        tuple
            (bytes_saved: int, percentage_reduction: float)
        """
        if uncompressed_bytes == 0:
            return 0, 0.0
            
        saved = max(0, uncompressed_bytes - compressed_bytes)
        pct = (saved / uncompressed_bytes) * 100.0
        return saved, pct
