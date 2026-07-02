import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from FC.fc_environment import FCEnv

def run_evaluation_episode(env, model=None, static_action=None, max_steps=500):
    obs, _ = env.reset()
    
    metrics = {
        "decode_success": [],
        "redundancy": [],
        "k": [],
        "loss_rate": [],
        "n_encoded": [],
        "n_received": [],
        "reward": [],
        "bw": [],
        "rtt": [],
        "raw_bler": []
    }
    
    steps = 0
    while steps < max_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = static_action
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        metrics["decode_success"].append(int(info["decode_success"]))
        metrics["redundancy"].append(info["redundancy_ratio"])
        metrics["k"].append(info["k"])
        metrics["loss_rate"].append(info["loss_rate"])
        metrics["n_encoded"].append(info["n_encoded"])
        metrics["n_received"].append(info["n_received"])
        metrics["reward"].append(reward)
        metrics["bw"].append(info.get("raw_bw", 0.0))
        metrics["rtt"].append(info.get("raw_rtt", 0.0))
        metrics["raw_bler"].append(info.get("raw_bler", 0.0))
        
        steps += 1
        if terminated or truncated:
            break
            
    return pd.DataFrame(metrics)

def evaluate():
    print("Loading environment and models...")
    env = FCEnv("stats_reno_aioquic.csv", is_eval=True) # Using reno dataset for robust evaluation
    
    # Load models
    model_path_v1 = "FC/models/fc_ppo_agent.zip"
    model_path_v2 = "FC/models/fc_ppo_agent_v2.zip"
    
    if not os.path.exists(model_path_v2):
        print(f"Error: Model {model_path_v2} not found. Please ensure training is complete.")
        return
    
    model_v1 = PPO.load(model_path_v1)
    model_v2 = PPO.load(model_path_v2)
    
    n_episodes = 5
    max_steps = 200
    
    print(f"Running {n_episodes} evaluation episodes ({max_steps} steps each)...")
    
    static_results = []
    v1_results = []
    v2_results = []
    
    for i in range(n_episodes):
        print(f"  Episode {i+1}/{n_episodes}...")
        
        # We need to ensure both evaluate on the exact same sequence of channel states.
        # So we reset the dataset with a specific seed, but FCEnv currently just shuffles.
        # We will manually set the seed to get the same shuffle.
        seed = 1000 + i
        
        # --- Run Static (action 4: k=128, ratio=1.3) ---
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.dataset.shuffle_data() # Force deterministic shuffle based on torch seed
        df_static = run_evaluation_episode(env, static_action=4, max_steps=max_steps)
        static_results.append(df_static)
        
        # --- Run Adaptive (PPO v1) ---
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.dataset.shuffle_data()
        df_v1 = run_evaluation_episode(env, model=model_v1, max_steps=max_steps)
        v1_results.append(df_v1)
        
        # --- Run Adaptive (PPO v2) ---
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.dataset.shuffle_data()
        df_v2 = run_evaluation_episode(env, model=model_v2, max_steps=max_steps)
        v2_results.append(df_v2)
        
    print("\nCompiling metrics...")
    
    static_concat = pd.concat(static_results, ignore_index=True)
    v1_concat = pd.concat(v1_results, ignore_index=True)
    v2_concat = pd.concat(v2_results, ignore_index=True)
    
    # Calculate aggregated metrics
    def calc_metrics(df):
        return {
            "DSR": df["decode_success"].mean(),
            "Packet Loss": df["loss_rate"].mean(),
            "Avg Redundancy": df["redundancy"].mean(),
            "Avg k": df["k"].mean(),
            "Transmission Overhead": df["n_encoded"].sum() / df["k"].sum(),
            "Latency (ms)": df["rtt"].mean(),
            "Avg Reward": df["reward"].mean()
        }
    
    m_stat = calc_metrics(static_concat)
    m_v1 = calc_metrics(v1_concat)
    m_v2 = calc_metrics(v2_concat)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"| Metric | Static FC | PPO v1 | PPO v2 |")
    print(f"|--------|-----------|--------|--------|")
    print(f"| Decode Success Rate | {m_stat['DSR']:.1%} | {m_v1['DSR']:.1%} | {m_v2['DSR']:.1%} |")
    print(f"| Packet Loss | {m_stat['Packet Loss']:.2%} | {m_v1['Packet Loss']:.2%} | {m_v2['Packet Loss']:.2%} |")
    print(f"| Avg Redundancy | {m_stat['Avg Redundancy']:.2f} | {m_v1['Avg Redundancy']:.2f} | {m_v2['Avg Redundancy']:.2f} |")
    print(f"| Avg k | {m_stat['Avg k']:.1f} | {m_v1['Avg k']:.1f} | {m_v2['Avg k']:.1f} |")
    print(f"| Transmission Overhead | {m_stat['Transmission Overhead']:.2f} | {m_v1['Transmission Overhead']:.2f} | {m_v2['Transmission Overhead']:.2f} |")
    print(f"| Latency (ms) | {m_stat['Latency (ms)']:.2f} | {m_v1['Latency (ms)']:.2f} | {m_v2['Latency (ms)']:.2f} |")
    print(f"| Avg Reward | {m_stat['Avg Reward']:.3f} | {m_v1['Avg Reward']:.3f} | {m_v2['Avg Reward']:.3f} |")
    print("="*80 + "\n")
    
    # Save evaluation graphs
    graphs_dir = "evaluation/graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    
    # 1. Reward vs Episode
    plt.figure(figsize=(10, 5))
    stat_rews = [df["reward"].sum() for df in static_results]
    v1_rews = [df["reward"].sum() for df in v1_results]
    v2_rews = [df["reward"].sum() for df in v2_results]
    plt.plot(range(1, n_episodes+1), stat_rews, marker='o', label='Static FC')
    plt.plot(range(1, n_episodes+1), v1_rews, marker='x', label='PPO v1')
    plt.plot(range(1, n_episodes+1), v2_rews, marker='s', label='PPO v2')
    plt.title("Cumulative Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(f"{graphs_dir}/reward_vs_episode.png")
    plt.close()
    
    # 2. DSR vs Episode
    plt.figure(figsize=(10, 5))
    stat_dsr = [df["decode_success"].mean() for df in static_results]
    v1_dsr = [df["decode_success"].mean() for df in v1_results]
    v2_dsr = [df["decode_success"].mean() for df in v2_results]
    plt.plot(range(1, n_episodes+1), stat_dsr, marker='o', label='Static FC')
    plt.plot(range(1, n_episodes+1), v1_dsr, marker='x', label='PPO v1')
    plt.plot(range(1, n_episodes+1), v2_dsr, marker='s', label='PPO v2')
    plt.title("Decode Success Rate per Episode")
    plt.xlabel("Episode")
    plt.ylabel("DSR")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.savefig(f"{graphs_dir}/dsr_vs_episode.png")
    plt.close()
    
    # 3. Redundancy Distribution
    plt.figure(figsize=(12, 5))
    
    v1_counts = v1_concat["redundancy"].value_counts().sort_index()
    v2_counts = v2_concat["redundancy"].value_counts().sort_index()
    
    all_indices = sorted(list(set(v1_counts.index) | set(v2_counts.index)))
    
    bar_width = 0.35
    index = np.arange(len(all_indices))
    
    v1_vals = [v1_counts.get(idx, 0) for idx in all_indices]
    v2_vals = [v2_counts.get(idx, 0) for idx in all_indices]
    
    plt.bar(index, v1_vals, bar_width, label='PPO v1', alpha=0.7)
    plt.bar(index + bar_width, v2_vals, bar_width, label='PPO v2', alpha=0.7)
    
    plt.title("Selected Redundancy Distribution Comparison")
    plt.xlabel("Redundancy Ratio")
    plt.ylabel("Frequency")
    plt.xticks(index + bar_width / 2, all_indices)
    plt.legend()
    plt.savefig(f"{graphs_dir}/redundancy_dist.png")
    plt.close()
    
    # 4. Redundancy Selection Analysis (Bucketed by Channel Condition)
    def bucket_channel(bler):
        if bler < 0.01: return "Excellent (<1%)"
        elif bler < 0.05: return "Good (1-5%)"
        elif bler < 0.10: return "Moderate (5-10%)"
        elif bler < 0.15: return "Poor (10-15%)"
        else: return "Very Poor (>15%)"
        
    v2_concat["Channel Condition"] = v2_concat["raw_bler"].apply(bucket_channel)
    
    # Order for display
    cat_order = ["Excellent (<1%)", "Good (1-5%)", "Moderate (5-10%)", "Poor (10-15%)", "Very Poor (>15%)"]
    v2_concat["Channel Condition"] = pd.Categorical(v2_concat["Channel Condition"], categories=cat_order, ordered=True)
    
    # Group by category
    grouped = v2_concat.groupby("Channel Condition", observed=False)
    
    print("\n" + "="*80)
    print("REDUNDANCY SELECTION ANALYSIS (PPO v2)")
    print("="*80)
    print(f"| Condition | Obs | Avg BLER | Avg RTT | Most Freq Ratio | Most Freq k | DSR |")
    print(f"|-----------|-----|----------|---------|-----------------|-------------|-----|")
    
    heatmap_data = []
    for cat in cat_order:
        group = v2_concat[v2_concat["Channel Condition"] == cat]
        if len(group) > 0:
            obs = len(group)
            avg_bler = group["raw_bler"].mean()
            avg_rtt = group["rtt"].mean()
            freq_ratio = group["redundancy"].mode().iloc[0]
            freq_k = group["k"].mode().iloc[0]
            dsr = group["decode_success"].mean()
            print(f"| {cat:10} | {obs:3d} | {avg_bler:.2%} | {avg_rtt:5.1f}ms | {freq_ratio:15.2f} | {freq_k:11.0f} | {dsr:.1%} |")
            
            # For heatmap
            ratios = group["redundancy"].value_counts(normalize=True).to_dict()
            ratios["Condition"] = cat
            heatmap_data.append(ratios)
        else:
            print(f"| {cat:10} |   0 |      N/A |     N/A |             N/A |         N/A | N/A |")
            
    print("="*80 + "\n")
    
    # Generate Heatmap for Redundancy vs Condition
    if len(heatmap_data) > 0:
        import seaborn as sns
        df_heat = pd.DataFrame(heatmap_data).set_index("Condition").fillna(0)
        # Sort columns to have increasing redundancy ratios
        df_heat = df_heat.reindex(sorted(df_heat.columns), axis=1)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".1%", cbar_kws={'label': 'Selection Frequency'})
        plt.title("PPO v2 Action Selection by Channel Condition")
        plt.xlabel("Selected Redundancy Ratio")
        plt.ylabel("Channel Condition (Raw BLER)")
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/redundancy_heatmap.png")
        plt.close()
        
    print(f"All graphs saved to {graphs_dir}/")

if __name__ == "__main__":
    evaluate()
