import os
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend BEFORE importing pyplot.
                       # This ensures savefig() always writes to disk, even when
                       # there is no display environment (e.g. SSH, subprocess, or
                       # a GUI backend that silently drops file writes).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Controller.controller_environment import ControllerEnv

def evaluate_controller(csv_path: str = "stats_cubic_aioquic.csv", model_path: str = "Controller/models/controller_ppo_agent.zip", results_dir: str = "Controller/results"):
    """
    Evaluates the Adaptive Controller against static TCP and static FC.
    Generates comparison tables and graphs.
    """
    # Resolve to an absolute path so savefig() always writes to the correct
    # location regardless of how or from where this script is invoked.
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to: {results_dir}")
    
    env = ControllerEnv(csv_path, is_eval=True)
    
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Using random agent for evaluation.")
        model = None
        
    num_episodes = 1 # Evaluate over 1 full pass of the dataset
    
    static_tcp_results = []
    static_fc_results = []
    adaptive_results = []
    
    # 1. Run Adaptive Controller
    obs, _ = env.reset()
    done = False
    
    # Tracking for Adaptive
    adaptive_modes = []
    
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        action_scalar = int(action)
        obs, reward, terminated, truncated, info = env.step(action_scalar)
        
        adaptive_modes.append(action_scalar)
        adaptive_results.append({
            "reward": reward,
            "latency": info.get("latency", 0),
            "packet_loss": info.get("packet_loss", 0),
            "overhead": info.get("overhead", 0),
            "success_rate": info.get("success_rate", 0),
            "mode": "TCP" if action_scalar == 0 else "FC"
        })
        done = terminated or truncated

    # 2. Run Static TCP (Always Action 0)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        static_tcp_results.append({
            "reward": reward,
            "latency": info.get("latency", 0),
            "packet_loss": info.get("packet_loss", 0),
            "overhead": info.get("overhead", 0),
            "success_rate": info.get("success_rate", 0),
            "mode": "TCP"
        })
        done = terminated or truncated
        
    # 3. Run Static FC (Always Action 1)
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(1)
        static_fc_results.append({
            "reward": reward,
            "latency": info.get("latency", 0),
            "packet_loss": info.get("packet_loss", 0),
            "overhead": info.get("overhead", 0),
            "success_rate": info.get("success_rate", 0),
            "mode": "FC"
        })
        done = terminated or truncated

    # --- Compute Summaries ---
    def summarize(results_list):
        df = pd.DataFrame(results_list)
        return {
            "avg_reward": df["reward"].mean(),
            "avg_latency": df["latency"].mean(),
            "avg_packet_loss": df["packet_loss"].mean(),
            "avg_overhead": df["overhead"].mean(),
            "avg_success_rate": df["success_rate"].mean(),
            "raw_df": df
        }
        
    tcp_summary = summarize(static_tcp_results)
    fc_summary = summarize(static_fc_results)
    adapt_summary = summarize(adaptive_results)
    
    adapt_df = adapt_summary["raw_df"]
    switch_count = (adapt_df["mode"] != adapt_df["mode"].shift()).sum() - 1 # first item is not a switch
    switch_freq = max(0, switch_count) / len(adapt_df)
    
    tcp_usage = (adapt_df["mode"] == "TCP").sum() / len(adapt_df)
    fc_usage = (adapt_df["mode"] == "FC").sum() / len(adapt_df)
    
    # --- Generate Comparison Table ---
    comparison_df = pd.DataFrame({
        "Metric": ["Avg Reward", "Avg Latency", "Avg Packet Loss", "Avg Success Rate", "Switch Freq", "TCP Usage %", "FC Usage %"],
        "Static TCP": [f'{tcp_summary["avg_reward"]:.3f}', f'{tcp_summary["avg_latency"]:.2f}', f'{tcp_summary["avg_packet_loss"]:.4f}', f'{tcp_summary["avg_success_rate"]:.4f}', "0.0", "100%", "0%"],
        "Static FC": [f'{fc_summary["avg_reward"]:.3f}', f'{fc_summary["avg_latency"]:.2f}', f'{fc_summary["avg_packet_loss"]:.4f}', f'{fc_summary["avg_success_rate"]:.4f}', "0.0", "0%", "100%"],
        "Adaptive Controller": [f'{adapt_summary["avg_reward"]:.3f}', f'{adapt_summary["avg_latency"]:.2f}', f'{adapt_summary["avg_packet_loss"]:.4f}', f'{adapt_summary["avg_success_rate"]:.4f}', f'{switch_freq:.3f}', f'{tcp_usage*100:.1f}%', f'{fc_usage*100:.1f}%']
    })
    
    table_path = os.path.join(results_dir, "comparison_table.csv")
    comparison_df.to_csv(table_path, index=False)
    print(f"\n--- Evaluation Results ---")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison table to {table_path}")
    
    # --- Helper Functions for Plotting ---
    def _save(filename):
        """Save current figure to results_dir and verify the file was actually written."""
        path = os.path.join(results_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        if not os.path.isfile(path):
            print(f"  WARNING: {filename} was not written to disk!")
        else:
            print(f"  Saved: {path}")

    def plot_bar_chart(filename, title, ylabel, labels, values, colors):
        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=colors)
        plt.title(title)
        plt.ylabel(ylabel)
        _save(filename)

    def plot_timeline(filename, title, ylabel, data, moving_avg=False):
        plt.figure(figsize=(10, 5))
        plt.plot(data, label="Adaptive Controller", alpha=0.7)
        if moving_avg:
            plt.plot(pd.Series(data).rolling(10).mean(), label="Moving Average", color="red")
        plt.title(title)
        plt.xlabel("Evaluation Step")
        plt.ylabel(ylabel)
        plt.legend()
        _save(filename)

    def plot_pie_chart(filename, title, labels, sizes, colors):
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title(title)
        _save(filename)

    # --- Generate Plots ---
    labels_bar = ['Static TCP', 'Static FC', 'Adaptive']
    colors_bar = ['blue', 'orange', 'green']
    
    # 1. Reward Comparison
    plot_bar_chart("reward_comparison.png", "Average Reward Comparison", "Average Reward", 
                   labels_bar, [tcp_summary["avg_reward"], fc_summary["avg_reward"], adapt_summary["avg_reward"]], colors_bar)
    
    # 2. Average Latency Comparison
    plot_bar_chart("latency_comparison.png", "Average Latency Comparison", "Latency (ms)", 
                   labels_bar, [tcp_summary["avg_latency"], fc_summary["avg_latency"], adapt_summary["avg_latency"]], colors_bar)
    
    # 3. Packet Loss Comparison
    plot_bar_chart("packet_loss_comparison.png", "Average Packet Loss Comparison", "Packet Loss", 
                   labels_bar, [tcp_summary["avg_packet_loss"], fc_summary["avg_packet_loss"], adapt_summary["avg_packet_loss"]], colors_bar)
    
    # 4. Success Rate Comparison
    plot_bar_chart("success_rate_comparison.png", "Success Rate Comparison", "Delivery / Decode Success Rate", 
                   labels_bar, [tcp_summary["avg_success_rate"], fc_summary["avg_success_rate"], adapt_summary["avg_success_rate"]], colors_bar)
    
    # 5. Mode Usage
    plot_pie_chart("mode_usage.png", "Adaptive Controller Mode Usage", 
                   ['TCP', 'FC'], [tcp_usage, fc_usage], ['blue', 'orange'])
    
    # 6. Decision Timeline
    plt.figure(figsize=(10, 3))
    mode_numeric = [0 if m == "TCP" else 1 for m in adapt_df["mode"]]
    plt.plot(mode_numeric, drawstyle='steps-mid', color='purple')
    plt.yticks([0, 1], ["TCP (0)", "Fountain Code (1)"])
    plt.title("Decision Timeline")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Protocol")
    _save("decision_timeline.png")
    
    # 7. Switching Timeline
    plt.figure(figsize=(10, 3))
    switches = [i for i in range(1, len(mode_numeric)) if mode_numeric[i] != mode_numeric[i-1]]
    switch_y = [mode_numeric[i] for i in switches]
    plt.scatter(switches, switch_y, color='red', marker='x', s=100, label="Switch Event")
    plt.yticks([0, 1], ["TCP (0)", "Fountain Code (1)"])
    plt.title("Switching Timeline")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Protocol")
    plt.legend()
    _save("switch_events.png")
    
    # 8. Reward Timeline
    plot_timeline("reward_timeline.png", "Adaptive Controller Reward over Time", "Reward", adapt_df["reward"], moving_avg=True)
    
    # 9. Latency Timeline
    plot_timeline("latency_timeline.png", "Adaptive Controller Latency over Time", "Latency (ms)", adapt_df["latency"])
    
    # 10. Packet Loss Timeline
    plot_timeline("packet_loss_timeline.png", "Adaptive Controller Packet Loss over Time", "Packet Loss", adapt_df["packet_loss"])
    
    # BONUS: Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].bar(labels_bar, [tcp_summary["avg_reward"], fc_summary["avg_reward"], adapt_summary["avg_reward"]], color=colors_bar)
    axs[0, 0].set_title("Reward Comparison")
    axs[0, 0].set_ylabel("Average Reward")
    
    axs[0, 1].bar(labels_bar, [tcp_summary["avg_latency"], fc_summary["avg_latency"], adapt_summary["avg_latency"]], color=colors_bar)
    axs[0, 1].set_title("Latency Comparison")
    axs[0, 1].set_ylabel("Latency (ms)")
    
    axs[1, 0].bar(labels_bar, [tcp_summary["avg_packet_loss"], fc_summary["avg_packet_loss"], adapt_summary["avg_packet_loss"]], color=colors_bar)
    axs[1, 0].set_title("Packet Loss Comparison")
    axs[1, 0].set_ylabel("Packet Loss")
    
    axs[1, 1].pie([tcp_usage, fc_usage], labels=['TCP', 'FC'], autopct='%1.1f%%', colors=['blue', 'orange'])
    axs[1, 1].set_title("Mode Usage")
    
    plt.suptitle("Adaptive Controller Dashboard", fontsize=16)
    plt.tight_layout()
    _save("controller_dashboard.png")
    
    print("\nGenerated:")
    print("reward_comparison.png")
    print("latency_comparison.png")
    print("packet_loss_comparison.png")
    print("success_rate_comparison.png")
    print("mode_usage.png")
    print("decision_timeline.png")
    print("switch_events.png")
    print("reward_timeline.png")
    print("latency_timeline.png")
    print("packet_loss_timeline.png")
    print("controller_dashboard.png")
    print("comparison_table.csv")

if __name__ == "__main__":
    evaluate_controller()
