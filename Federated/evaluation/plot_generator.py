"""
Federated/evaluation/plot_generator.py
=======================================
Publication-quality graph generation for FL evaluation.

Purpose
-------
Consumes aggregated metrics (from metrics.py) and generates 14 required
research plots using matplotlib.  Enforces consistent styling, high-DPI
output, and clear legends.

Design Decisions
----------------
1. Matplotlib OO API: Uses the Object-Oriented ``fig, ax = plt.subplots()``
   API rather than stateful ``plt.*`` to prevent cross-contamination
   between graphs in a multi-plot pipeline.
2. Error Bands: Plots the mean as a solid line and fills the 95% Confidence
   Interval as a shaded region (``fill_between``).
3. Standardized Styling: Enforces 300 DPI, grid lines, and legible fonts.

Algorithm Complexity
--------------------
O(M * T) where M is the number of modes and T is the number of rounds.
Negligible execution time.

Memory Complexity
-----------------
O(1) memory footprint beyond matplotlib's internal buffers. Figures are
closed immediately after saving to prevent memory leaks.

Output Files Generated
----------------------
14 PNG files in ``Federated/results/plots/`` (e.g., ``accuracy_vs_round.png``).

Research Significance
---------------------
Visualizing communication savings against model degradation (Accuracy vs Cost)
is the gold standard for evaluating gradient compression techniques.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

from Federated.config import PLOTS_DIR, PLOT_DPI, PLOT_FIGSIZE


class PlotGenerator:
    """Generates and saves publication-quality evaluation graphs."""

    def __init__(self, output_dir: str = PLOTS_DIR) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply standard research styling
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Helper to save and safely close a matplotlib figure."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)

    def generate_all_plots(self, aggregated_results: Dict[str, List[Dict[str, float]]]) -> None:
        """
        Generate all 14 requested graphs.

        Parameters
        ----------
        aggregated_results : dict
            Mapping from mode_name (e.g., "Full FL", "Top-K") to its
            aggregated metrics list (produced by MetricsAggregator).
        """
        # Line plots vs Round
        self._plot_metric_vs_round(aggregated_results, "val_accuracy", "Accuracy", "Accuracy vs Communication Round", "1_accuracy_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "train_loss", "Training Loss", "Training Loss vs Round", "2_train_loss_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "val_loss", "Validation Loss", "Validation Loss vs Round", "3_val_loss_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "comm_bytes_total", "Bytes", "Communication Cost vs Round", "4_comm_cost_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "avg_compression_ratio", "Ratio", "Compression Ratio vs Round", "5_comp_ratio_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "avg_reconstruction_error", "Norm. L2 Error", "Reconstruction Error vs Round", "6_recon_error_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "agg_time_s", "Seconds", "Aggregation Time vs Round", "7_agg_time_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "round_time_s", "Seconds", "Round Execution Time vs Round", "8_round_time_vs_round.png")
        self._plot_metric_vs_round(aggregated_results, "num_successful_clients", "Clients", "Client Participation per Round", "13_client_participation_vs_round.png")

        # Custom / Cumulative Plots
        self._plot_cumulative_savings(aggregated_results, "9_communication_savings.png")
        self._plot_accuracy_vs_cost(aggregated_results, "10_accuracy_vs_cost.png")
        self._plot_bar_chart(aggregated_results, "bandwidth_saved", "11_bandwidth_saved.png")
        self._plot_bar_chart(aggregated_results, "model_size_reduction", "12_model_size_reduction.png")
        self._plot_compression_distribution(aggregated_results, "14_compression_ratio_distribution.png")

    def _plot_metric_vs_round(
        self, 
        results: Dict[str, List[Dict[str, float]]], 
        metric_key: str, 
        ylabel: str, 
        title: str, 
        filename: str
    ) -> None:
        """Plot a single metric against rounds with 95% CI bands."""
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        has_data = False
        for mode_name, rounds_data in results.items():
            if not rounds_data or f"{metric_key}_mean" not in rounds_data[0]:
                continue
                
            has_data = True
            x = [r["round_idx"] for r in rounds_data]
            y = [r[f"{metric_key}_mean"] for r in rounds_data]
            ci = [r[f"{metric_key}_ci95"] for r in rounds_data]
            
            y_lower = [max(0, val - c) if metric_key != "val_accuracy" else max(0, val - c) for val, c in zip(y, ci)]
            y_upper = [val + c for val, c in zip(y, ci)]

            line, = ax.plot(x, y, label=mode_name, marker='o', markersize=4, linewidth=2)
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=line.get_color())

        if has_data:
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel("Communication Round", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend(loc="best", frameon=True, shadow=True)
            ax.grid(True, linestyle='--', alpha=0.7)
            self._save_plot(fig, filename)
        else:
            plt.close(fig)

    def _plot_cumulative_savings(self, results: Dict[str, List[Dict[str, float]]], filename: str) -> None:
        """Plot cumulative bytes transmitted over rounds."""
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for mode_name, rounds_data in results.items():
            if not rounds_data:
                continue
            x = [r["round_idx"] for r in rounds_data]
            y_raw = [r["comm_bytes_total_mean"] for r in rounds_data]
            y_cum = np.cumsum(y_raw) / (1024 * 1024) # MB
            ax.plot(x, y_cum, label=mode_name, marker='s', markersize=4, linewidth=2)
            
        ax.set_title("Cumulative Communication Cost", fontsize=14, pad=10)
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Total Transmitted (MB)", fontsize=12)
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.7)
        self._save_plot(fig, filename)

    def _plot_accuracy_vs_cost(self, results: Dict[str, List[Dict[str, float]]], filename: str) -> None:
        """Plot validation accuracy against cumulative communication cost."""
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for mode_name, rounds_data in results.items():
            if not rounds_data:
                continue
            acc = [r["val_accuracy_mean"] for r in rounds_data]
            cost_raw = [r["comm_bytes_total_mean"] for r in rounds_data]
            cost_cum = np.cumsum(cost_raw) / (1024 * 1024) # MB
            ax.plot(cost_cum, acc, label=mode_name, marker='^', markersize=4, linewidth=2)
            
        ax.set_title("Validation Accuracy vs Communication Cost", fontsize=14, pad=10)
        ax.set_xlabel("Cumulative Transmitted (MB)", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.7)
        self._save_plot(fig, filename)

    def _plot_bar_chart(self, results: Dict[str, List[Dict[str, float]]], chart_type: str, filename: str) -> None:
        """Plot simple bar charts for final comparative metrics."""
        fig, ax = plt.subplots(figsize=(8, 6))
        modes = []
        values = []
        
        baseline_bytes = 0.0
        # Find baseline (Full Update usually has largest bytes)
        for mode_name, rounds_data in results.items():
            if not rounds_data:
                continue
            total_bytes = sum(r["comm_bytes_total_mean"] for r in rounds_data)
            if total_bytes > baseline_bytes:
                baseline_bytes = total_bytes
        
        for mode_name, rounds_data in results.items():
            if not rounds_data:
                continue
            modes.append(mode_name)
            
            if chart_type == "bandwidth_saved":
                total_bytes = sum(r["comm_bytes_total_mean"] for r in rounds_data)
                pct_saved = 0.0
                if baseline_bytes > 0:
                    pct_saved = ((baseline_bytes - total_bytes) / baseline_bytes) * 100.0
                values.append(pct_saved)
            elif chart_type == "model_size_reduction":
                avg_ratio = np.mean([r["avg_compression_ratio_mean"] for r in rounds_data])
                values.append((1.0 - avg_ratio) * 100.0)

        bars = ax.bar(modes, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(modes)])
        
        title = "Bandwidth Saved (%) vs Baseline" if chart_type == "bandwidth_saved" else "Average Model Size Reduction (%)"
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        ax.set_ylim(0, max(100, max(values + [0]) * 1.1))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        self._save_plot(fig, filename)

    def _plot_compression_distribution(self, results: Dict[str, List[Dict[str, float]]], filename: str) -> None:
        """Plot boxplots of compression ratios."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = []
        labels = []
        for mode_name, rounds_data in results.items():
            if not rounds_data:
                continue
            ratios = [r["avg_compression_ratio_mean"] for r in rounds_data]
            data.append(ratios)
            labels.append(mode_name)
            
        if data:
            ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_title("Compression Ratio Distribution Across Rounds", fontsize=14, pad=10)
            ax.set_ylabel("Compression Ratio", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            self._save_plot(fig, filename)
        else:
            plt.close(fig)
