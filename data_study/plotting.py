import pandas as pd
import os
import sys
import inspect
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

# Setup directories relatively
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(script_dir)
sys.path.insert(0, parentdir)

# Save path for plots
save_path = os.path.normpath(os.path.join(script_dir, "..", "plots")) + os.sep
os.makedirs(save_path, exist_ok=True)

# Define dataset paths relative to the workspace
file_path_final = os.path.normpath(os.path.join(script_dir, "..", "stats_cubic_quiche.csv"))

def plot_plot(data, feature_name):
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    plt.figure()
    plt.plot(data["timestamp"], data[feature_name])
    plt.xlabel("Time")
    plt.ylabel("Feature Value")
    plt.title(f"Time Series Plot of {data[feature_name].name}")
    plt.savefig(os.path.join(save_path, f"plot_{feature_name}.png"))
    plt.close()

def plot_multiple_features(data, feature_names):
    data["timestamp"] = pd.to_datetime(data["timestamp"])  # Ensure timestamp is datetime
    plt.figure(figsize=(12, 6))

    for i, feature_name in enumerate(feature_names):
        plt.subplot(len(feature_names), 1, i + 1)
        plt.plot(data["timestamp"], data[feature_name])
        plt.xlabel("Time")
        plt.ylabel(feature_name)
        plt.title(feature_name)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "multiple_features.png"))
    plt.close()

if __name__ == '__main__':
    print("Loading cubic data from:", file_path_final)
    data = pd.read_csv(file_path_final, on_bad_lines='skip')
    numeric_data = data.apply(pd.to_numeric, errors='coerce')
    correlation_matrix = numeric_data.corr()

    # Mask the matrix to keep only positive values
    positive_correlation_matrix = correlation_matrix.mask(correlation_matrix > 0.8)

    # Drop rows and columns that don't contain any positive correlations
    filtered_correlation_matrix = positive_correlation_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

    path = os.path.normpath(os.path.join(script_dir, "..", "training.csv"))
    print("Loading training data from:", path)
    df = pd.read_csv(path)

    rew = df["bw_avg"] * 1024 / df["rtt"]

    path_testing = os.path.normpath(os.path.join(script_dir, "..", "testing.csv"))
    print("Loading testing data from:", path_testing)
    df_test = pd.read_csv(path_testing, on_bad_lines='skip')[100:200]
    my_bw = df_test["bw_avg"] * 1024
    print("My Algorithm BW length:", len(my_bw))

    path_cubic = os.path.normpath(os.path.join(script_dir, "..", "stats_cubic_quiche.csv"))
    print("Loading cubic statistics from:", path_cubic)
    df_cubic = pd.read_csv(path_cubic, on_bad_lines='skip')[2500:2600]

    cubic_bw = df_cubic["bw_avg"]
    print("Cubic BW length:", len(cubic_bw))
    cubic_rtt = df_cubic["smoothed_rtt"]
    cubic_bw_mean = cubic_bw.mean()
    cubic_rtt_mean = cubic_rtt.mean()
    cubic_bw_std = cubic_bw.std()
    cubic_rtt_std = cubic_rtt.std()
    print(f"cubic rtt avg: {cubic_rtt_mean}")
    print(f"cubic rtt std: {cubic_rtt_std}")
    print(f"cubic bw avg: {cubic_bw.mean()}")
    print(f"cubic bw median: {cubic_bw.median()}")
    print(f"cubic bw std: {cubic_bw_std}")

    path_reno = os.path.normpath(os.path.join(script_dir, "..", "stats_reno_aioquic.csv"))
    print("Loading reno statistics from:", path_reno)
    df_reno = pd.read_csv(path_reno, on_bad_lines='skip')[30000:30100]

    reno_bler = df_reno["BLER_UL"]
    reno_bw = df_reno["bw_avg"]
    reno_bw_mean = reno_bw.mean()
    reno_rtt_mean = df_reno["smoothed_rtt"].mean()
    reno_bw_std = reno_bw.std()
    reno_rtt_std = df_reno["smoothed_rtt"].std()
    print(f"reno rtt avg: {reno_rtt_mean}")
    print(f"reno rtt std: {reno_rtt_std}")
    print(f"reno bw avg: {reno_bw_mean}")
    print(f"reno bw median: {reno_bw.median()}")
    print(f"reno bw std: {reno_bw_std}")
    reno_rtt = df_reno["smoothed_rtt"]
    mine_rtt = df_test["rtt"]

    # Bandwidth comparison plot
    plt.figure(figsize=(10,6))
    plt.plot(range(len(cubic_bw)), cubic_bw, label="Cubic Bandwidth", lw=2)
    plt.plot(range(len(cubic_bw)), reno_bw, label="New Reno Bandwidth", lw=2)
    plt.plot(range(len(cubic_bw)), my_bw, label="Our Algorithm Bandwidth", lw=2)
    plt.xlabel("Timesteps", fontsize=20)
    plt.ylabel("Bandwidth (KB/s)",  fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(save_path, "bw_comparison_100.png"))
    plt.close()
    print("Saved bandwidth comparison plot.")

    # RTT comparison plot
    path_testing_all = os.path.normpath(os.path.join(script_dir, "..", "testing.csv"))
    df_test_all = pd.read_csv(path_testing_all, on_bad_lines='skip')
    bw_all = df_test_all["bw_avg"] * 1024
    mine_bw_mean = bw_all.mean()
    mine_rtt_mean = df_test_all["rtt"].mean()
    mine_rtt_std = df_test_all["rtt"].std()
    print(f"mine rtt avg: {mine_rtt_mean}")
    print(f"mine rtt std: {mine_rtt_std}")
    print(f"mine bw median: {bw_all.median()}")
    print(f"mine bw avg: {mine_bw_mean}")
    print(f"mine bw std: {bw_all.std()}")
    bler = df_test_all["BLER_UL"]
    rtt = df_test_all["rtt"][0:100]
    print(f"Mine vs reno: {mine_bw_mean / reno_bw_mean}")
    print(f"Mine vs cubic: {mine_bw_mean / cubic_bw_mean}")

    plt.figure(figsize=(10,6))
    plt.plot(range(len(cubic_bw)), cubic_rtt, label="Cubic RTT", lw=2)
    plt.plot(range(len(cubic_bw)), reno_rtt, label="New Reno RTT", lw=2)
    plt.plot(range(len(cubic_bw)), rtt, label="Our Algorithm RTT", lw=2)
    plt.xlabel("Timesteps", fontsize=20)
    plt.ylabel("Round Trip Time (ms)", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(save_path, "rtt_comparison_100.png"))
    plt.close()
    print("Saved RTT comparison plot.")

    path_training_all = os.path.normpath(os.path.join(script_dir, "..", "training.csv"))
    df_train_all = pd.read_csv(path_training_all)
    bw_train = df_train_all["bw_avg"]
    rtt_train = df_train_all["rtt"]
    max_bw = bw_train.max()
    max_rtt = rtt_train.max()
    reward = bw_train * 1024 / rtt_train
    reward_max = reward.max()

    plt.figure(figsize=(10,6))
    plt.plot(bw_train / max_bw, label="Normalized Bandwidth")
    plt.plot(rtt_train / max_rtt, label="Normalized RTT")
    plt.plot(reward / reward_max, label="Normalized Reward")
    plt.xlabel("Timesteps", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(os.path.join(save_path, "bw_rtt_rew.png"))
    plt.close()
    print("Saved BW, RTT and Reward plot.")
