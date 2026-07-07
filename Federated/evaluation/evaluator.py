"""
Federated/evaluation/evaluator.py
==================================
Primary orchestration script for Federated Learning research evaluation.

Purpose
-------
Automates the execution of multiple experimental modes (Full FL, Top-K, 
Top-K with Error Feedback) across multiple deterministic runs to evaluate
the statistical significance of communication reduction algorithms.
Outputs comprehensive CSV logs, Markdown tables, JSON configs, and plots.

Design Decisions
----------------
1. Clean State Isolation: Each run of each mode reinstantiates the model,
   trainers, datasets, and RNG seeds from scratch to prevent state leakage.
2. Pipelined Architecture: 
   Execution -> Raw Metrics -> Aggregation -> Tables/Plots
3. Automated Baselines: Dynamically determines the "Full FL" baseline
   by inspecting the results to compute percentage reductions automatically.

Algorithm Complexity
--------------------
O(M * R * T) where M is modes, R is runs, T is rounds.
Highly dominated by the FL simulation loop (Local Training).

Memory Complexity
-----------------
O(P) for active simulation state, plus O(M * R * T) for storing the raw
metrics dicts in memory until the end of the experiment block.

Output Files Generated
----------------------
- results/logs/experiment_config.json
- results/logs/summary_metrics.csv
- results/logs/summary_metrics.md
- results/plots/*.png (via plot_generator.py)

Research Significance
---------------------
Provides a 1-click reproduction pipeline that generates all artifacts
required for a research paper on Federated Learning gradient compression.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Federated.config import (
    set_seed, SEED, NUM_CLIENTS, EVAL_ROUNDS, EVAL_MODES,
    LOGS_DIR, CHECKPOINTS_DIR, PLOTS_DIR, BATCH_SIZE, TOP_K_RATIO
)
from Federated.model.fl_model import FLModel
from Federated.simulation.fl_round_runner import FLRoundRunner
from Federated.evaluation.metrics import MetricsAggregator
from Federated.evaluation.plot_generator import PlotGenerator


class Evaluator:
    """
    Orchestrates automated evaluation of FL gradient compression.
    """

    def __init__(self, num_runs: int = 1) -> None:
        """
        Parameters
        ----------
        num_runs : int
            Number of independent runs per mode for statistical aggregation.
            Default is 1 (fast execution). Set to 5 for publication.
        """
        self.num_runs = num_runs
        self.modes = EVAL_MODES
        self.output_dir = LOGS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.aggregated_results: Dict[str, List[Dict[str, float]]] = {}
        self.summary_table: List[Dict[str, Any]] = []

    def run_all_evaluations(self) -> None:
        """Execute all modes, aggregate metrics, and generate reports."""
        print(f"\n{'='*64}")
        print(f"Starting Evaluation Pipeline ({self.num_runs} runs/mode)")
        print(f"{'='*64}")
        
        # 1. Run Simulations
        for mode_name, top_k, use_ef in self.modes:
            mode_runs_data: List[List[Dict[str, Any]]] = []
            print(f"\n>> Evaluating Mode: {mode_name}")
            
            for run_idx in range(self.num_runs):
                seed = SEED + run_idx
                print(f"   Run {run_idx+1}/{self.num_runs} (Seed: {seed})")
                
                # Full state isolation per run
                set_seed(seed)
                model, client_loaders, client_sizes, test_loader = self._prepare_data()
                
                runner = FLRoundRunner(
                    global_model=model,
                    client_loaders=client_loaders,
                    client_sizes=client_sizes,
                    test_loader=test_loader,
                    top_k_ratio=top_k,
                    use_error_feedback=use_ef,
                    experiment_name=f"eval_{mode_name.replace(' ', '_')}_run{run_idx}",
                    verbosity=0
                )
                
                metrics = runner.run(num_rounds=EVAL_ROUNDS)
                mode_runs_data.append([m.to_dict() for m in metrics])
                
            # 2. Aggregate Metrics for this mode
            agg_data = MetricsAggregator.aggregate_runs(mode_runs_data)
            self.aggregated_results[mode_name] = agg_data

        # 3. Generate Reports & Plots
        self._generate_summary_table()
        self._export_summary_csv()
        self._export_summary_markdown()
        self._export_config()
        
        plotter = PlotGenerator()
        plotter.generate_all_plots(self.aggregated_results)
        
        print(f"\n{'='*64}")
        print("Evaluation Complete. Artifacts generated in results/logs and results/plots.")
        print(f"{'='*64}\n")

    def _prepare_data(self) -> Tuple[nn.Module, List[DataLoader], List[int], DataLoader]:
        """Mock data factory to isolate state between runs."""
        # Note: In a real environment, this would call data_partitioner.py
        # Here we use synthetic data for robust, standalone CI/CD capability.
        X_all = torch.randn(500, 1, 28, 28)
        y_all = torch.randint(0, 10, (500,))
        
        loaders = []
        sizes = []
        for i in range(NUM_CLIENTS):
            start = i * 50
            end = start + 50
            ds = TensorDataset(X_all[start:end], y_all[start:end])
            loaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))
            sizes.append(50)
            
        X_test = torch.randn(200, 1, 28, 28)
        y_test = torch.randint(0, 10, (200,))
        test_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = FLModel()
        return model, loaders, sizes, test_loader

    def _generate_summary_table(self) -> None:
        """Compute final scalar comparative metrics across all modes."""
        # Find baseline bytes
        baseline_bytes = 1.0
        for name, data in self.aggregated_results.items():
            if "Full Update" in name or "Full FL" in name:
                baseline_bytes = sum(r["comm_bytes_total_mean"] for r in data)
                
        for mode_name, data in self.aggregated_results.items():
            if not data:
                continue
                
            final_acc = data[-1]["val_accuracy_mean"]
            total_bytes = sum(r["comm_bytes_total_mean"] for r in data)
            avg_ratio = np.mean([r["avg_compression_ratio_mean"] for r in data])
            bytes_saved, pct_red = MetricsAggregator.compute_communication_savings(
                int(total_bytes), int(baseline_bytes)
            )
            total_time = sum(r["round_time_s_mean"] for r in data)
            avg_recon_err = np.nanmean([r["avg_reconstruction_error_mean"] for r in data])
            
            self.summary_table.append({
                "Mode": mode_name,
                "Final Accuracy": final_acc,
                "Comm Cost (MB)": total_bytes / (1024*1024),
                "Compression Ratio": avg_ratio,
                "Bytes Saved (MB)": bytes_saved / (1024*1024),
                "Comm Reduction (%)": pct_red,
                "Total Time (s)": total_time,
                "Avg Recon Error": avg_recon_err
            })

    def _export_summary_csv(self) -> None:
        """Export summary table to CSV."""
        path = os.path.join(self.output_dir, "summary_metrics.csv")
        if not self.summary_table:
            return
        keys = self.summary_table[0].keys()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.summary_table)

    def _export_summary_markdown(self) -> None:
        """Export summary table to Markdown for easy GitHub viewing."""
        path = os.path.join(self.output_dir, "summary_metrics.md")
        if not self.summary_table:
            return
            
        keys = list(self.summary_table[0].keys())
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Federated Learning Evaluation Summary\n\n")
            
            # Header
            f.write("| " + " | ".join(keys) + " |\n")
            f.write("|" + "|".join(["---"] * len(keys)) + "|\n")
            
            # Rows
            for row in self.summary_table:
                formatted_row = []
                for k in keys:
                    v = row[k]
                    if isinstance(v, float):
                        formatted_row.append(f"{v:.4f}")
                    else:
                        formatted_row.append(str(v))
                f.write("| " + " | ".join(formatted_row) + " |\n")

    def _export_config(self) -> None:
        """Export experiment configuration and reproducibility metadata."""
        config = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "num_runs": self.num_runs,
            "base_seed": SEED,
            "eval_rounds": EVAL_ROUNDS,
            "num_clients": NUM_CLIENTS,
            "batch_size": BATCH_SIZE,
            "modes_evaluated": [m[0] for m in self.modes]
        }
        path = os.path.join(self.output_dir, "experiment_config.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)


if __name__ == "__main__":
    evaluator = Evaluator(num_runs=1)
    evaluator.run_all_evaluations()
