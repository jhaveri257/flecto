import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import os
from matplotlib.patches import Ellipse

def bootstrap_ci(data, statfunction=np.mean, alpha=0.05, n_samples=1000):
    """
    Calculates a bootstrap confidence interval for a given statistic.

    Args:
        data: The input data (can be a Pandas Series or NumPy array).
        statfunction: The function to calculate the statistic (e.g., np.mean, np.median).
        alpha: Significance level (e.g., 0.05 for 95% confidence).
        n_samples: Number of bootstrap samples.

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    nvals = int(alpha/2 * n_samples)  # Corrected calculation for nvals
    boot_indexes = np.random.randint(0, len(data), size=(n_samples, len(data)))
    boot_samples = []
    for i in range(n_samples):
        sample_indices = boot_indexes[i]
        boot_samples.append(data[sample_indices]) 
    boot_stats = np.apply_along_axis(statfunction, 1, boot_samples) 
    boot_stats.sort() 
    lower_bound = boot_stats[nvals]
    upper_bound = boot_stats[-nvals-1]  # Corrected index for upper bound
    return lower_bound, upper_bound


def round_up_to_nearest_50(num):
  remainder = num % 50
  if remainder == 0:
    return num
  else:
    return num + (50 - remainder)

def plot_from_list_training(data, feature_name, line_colors, plot_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, feature in enumerate(feature_name):
        ax.plot(data[i], label=feature, color=line_colors[i])
    ax.set_xlabel("Timesteps")  # Set the x-axis label
    ax.set_ylabel("Normalized Value")  # Set the y-axis
    ax.set_xlim(-50, round_up_to_nearest_50(len(data[0])))
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', fontsize=13)
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout')
    ax.tick_params(which='minor', length=5, width=1, direction='in')
    ax.set_xticks(np.arange(0, 1200, 200), minor=True)
    ax.set_yticks(np.arange(0, 1, 0.2), minor=True)
    ax.grid(which='both', linewidth=0.5)
    
    save_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"))

def plot_from_list_testing(data, feature_name, line_styles, line_colors, y_desc, plot_name):
    max = -np.inf
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, feature in enumerate(feature_name):
        ax.plot(data[i], label=feature, linestyle=line_styles[i], color=line_colors[i], lw=1.5)
        if np.max(data[i]) > max:
            max = np.max(data[i])
    ax.set_xlabel("Timesteps")  # Set the x-axis label
    ax.set_ylabel(y_desc)  # Set the y-axis
    if "Mbps" in y_desc:
        ax.legend(loc='lower right', fontsize=13)
        step = 2
    else:
        ax.legend(fontsize=13)
        step = 100
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout')
    ax.tick_params(which='minor', length=5, width=1, direction='in')
    ax.set_xticks(np.arange(0, len(data[0]), 20), minor=True)
    ax.set_yticks(np.arange(0, max, step), minor=True)
    ax.set_yticks(np.arange(0, max, step))

    ax.grid(which='both', linewidth=0.5)
    
    save_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"))

"""
def plot_bw_vs_rtt(bw_data, rtt_data, bw_err, rtt_err, feature_name, markers, colors, plot_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, feature in enumerate(feature_name):
        print(feature)
        rtt_lower = rtt_data[i] - rtt_err[i][0]
        rtt_upper = rtt_err[i][1] - rtt_data[i]
        bw_lower = bw_data[i] - bw_err[i][0]
        bw_upper = bw_err[i][1] - bw_data[i]
        print("RTT and BW means", rtt_data[i], bw_data[i])
        print("RTT and BW CI", rtt_err[i], bw_err[i])
        print(rtt_lower, rtt_upper, bw_lower, bw_upper)
        ax.errorbar(
            rtt_data[i],
            bw_data[i],
            xerr=np.array([rtt_lower, rtt_upper]).reshape(2,1),  # 2D structure for asymmetric xerr
            yerr=np.array([bw_lower, bw_upper]).reshape(2,1),  # 2D structure for asymmetric yerr
            label=feature,
            lw=2,
            marker=markers[i],
            color=colors[i],
        )

    ax.set_xlabel("RTT (ms)")  # Set the x-axis label
    ax.set_ylabel("Bandwidth (Mbps)")  # Set the y-axis
    ax.legend(fontsize=13)
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout')
    ax.tick_params(which='minor', length=5, width=1, direction='in')
    ax.grid(which='both', linewidth=0.5)

    plt.savefig(f"/home/cristiano/Desktop/USA/Thesis_Final/New_Plots/{plot_name}.png")
"""

def plot_bw_vs_rtt(bw_data, rtt_data, bw_err, rtt_err, feature_name, markers, colors, plot_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, feature in enumerate(feature_name):
        ax.scatter(rtt_data[i], bw_data[i], label=feature, marker=markers[i], color=colors[i])
        rtt_lower = rtt_data[i] - rtt_err[i][0]
        rtt_upper = rtt_err[i][1] - rtt_data[i]
        bw_lower = bw_data[i] - bw_err[i][0]
        bw_upper = bw_err[i][1] - bw_data[i]
        width = rtt_lower + rtt_upper
        height = bw_lower + bw_upper
        #print(feature, width, height)
        ellipse = Ellipse((rtt_data[i], bw_data[i]), width=width, height=height, color=colors[i], alpha=0.2)
        ax.add_patch(ellipse)

    ax.set_xlabel("RTT (ms)")  # Set the x-axis label
    ax.set_ylabel("Throughput (Mbps)")  # Set the y-axis
    ax.legend(fontsize=13)
    ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=1, direction='inout')
    ax.tick_params(which='minor', length=5, width=1, direction='in')
    ax.set_xticks(np.arange(10, 180, 40), minor=True)
    ax.set_xticks(np.arange(10, 180, 40))
    auto_ylim = ax.get_ylim()
    ax.set_yticks(np.arange(2.5, auto_ylim[1], 0.5), minor=True)
    ax.set_yticks(np.arange(2.5, auto_ylim[1], 0.5))
    ax.grid(which='both', linewidth=0.5)

    save_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots"))
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{plot_name}.png"))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(script_dir, ".."))

    training_data = pd.read_csv(os.path.join(base_dir, "training.csv"))
    training_bw = training_data["bw_avg"] / training_data["bw_avg"].max()
    training_rtt = training_data["rtt"] / training_data["rtt"].max()
    training_rew = training_bw / training_rtt
    training_rew = training_rew / training_rew.max()
    plot_from_list_training(data=[training_bw, training_rtt, training_rew], feature_name=["Avg BW", "RTT", "Reward"],
                             line_colors=["red", "green", "blue"], plot_name="training")

    testing_data = pd.read_csv(os.path.join(base_dir, "testing.csv"), on_bad_lines='skip')
    cubic_data = pd.read_csv(os.path.join(base_dir, "stats_cubic_quiche.csv"), on_bad_lines='skip')
    reno_data = pd.read_csv(os.path.join(base_dir, "stats_reno_aioquic.csv"), on_bad_lines='skip')
    number_of_values = 100
    testing_bw = pd.to_numeric(testing_data["bw_avg"][100:100+number_of_values]).to_numpy() * 1024 / 1000
    cubic_bw = cubic_data["bw_avg"][2500:2500+number_of_values].to_numpy() / 1000
    reno_bw = reno_data["bw_avg"][30000:30000+number_of_values].to_numpy() / 1000
    plot_from_list_testing(data=[testing_bw, cubic_bw, reno_bw], feature_name=["Flecto", "Cubic", "New Reno"], y_desc="Throughput (Mbps)",
                            line_styles=["-", "dashed", "dotted"], line_colors=["red", "green", "blue"], plot_name="bw_comparison")
    testing_rtt = testing_data["rtt"][100:100+number_of_values].to_numpy()
    cubic_rtt = cubic_data["smoothed_rtt"][2500:2500+number_of_values].to_numpy()
    reno_rtt = reno_data["smoothed_rtt"][30000:30000+number_of_values].to_numpy()
    plot_from_list_testing(data=[testing_rtt, cubic_rtt, reno_rtt], feature_name=["Flecto", "Cubic", "New Reno"], y_desc="RTT (ms)",
                            line_styles=["-", "dashed", "dotted"], line_colors=["red", "green", "blue"], plot_name="rtt_comparison")
    
    testing_rtt_avg = testing_data["rtt"].mean()
    cubic_rtt_avg = cubic_data["smoothed_rtt"].mean()
    reno_rtt_avg = reno_data["smoothed_rtt"].mean()

    testing_bw = testing_data["bw_avg"].to_numpy()
    testing_bw_avg = testing_bw.mean()
    cubic_bw = cubic_data["bw_avg"].to_numpy() / 1000
    cubic_bw_avg = cubic_bw.mean()
    reno_bw = reno_data["bw_avg"].to_numpy() / 1000
    reno_bw_avg = reno_bw.mean()

    int_conf_perc = 0.9
    alpha = 1 - int_conf_perc

    #plt.hist(reno_bw, bins=100)
    #plt.show()

    int_conf_testing_bw = st.norm.interval(int_conf_perc, loc=np.mean(testing_bw), scale=st.sem(testing_bw))
    int_conf_cubic_bw = st.norm.interval(int_conf_perc, loc=np.mean(cubic_bw), scale=st.sem(cubic_bw))
    int_conf_reno_bw = st.norm.interval(int_conf_perc, loc=np.mean(reno_bw), scale=st.sem(reno_bw))

    int_conf_testing_rtt = st.norm.interval(int_conf_perc, loc=np.mean(testing_data["rtt"]), scale=st.sem(testing_data["rtt"]))
    int_conf_cubic_rtt = st.norm.interval(int_conf_perc, loc=np.mean(cubic_data["smoothed_rtt"]), scale=st.sem(cubic_data["smoothed_rtt"]))
    int_conf_reno_rtt = st.norm.interval(int_conf_perc, loc=np.mean(reno_data["smoothed_rtt"]), scale=st.sem(reno_data["smoothed_rtt"]))

    #ci_testing_bw = bootstrap_ci(testing_bw, alpha=alpha)
    #ci_cubic_bw = bootstrap_ci(cubic_bw, alpha=alpha)
    #ci_reno_bw = bootstrap_ci(reno_bw, alpha=alpha)

    #ci_testing_rtt = bootstrap_ci(testing_data["rtt"], alpha=alpha)
    #ci_cubic_rtt = bootstrap_ci(cubic_data["smoothed_rtt"], alpha=alpha)
    #ci_reno_rtt = bootstrap_ci(reno_data["smoothed_rtt"], alpha=alpha)

    #print("Bootstrap Confidence Intervals (Bandwidth):")
    #print("Testing:", ci_testing_bw)
    #print("Cubic:", ci_cubic_bw)
    #print("Reno:", ci_reno_bw)

    #print("\nBootstrap Confidence Intervals (RTT):")
    #print("Testing:", ci_testing_rtt)
    #print("Cubic:", ci_cubic_rtt)
    #print("Reno:", ci_reno_rtt)

    plot_bw_vs_rtt(rtt_data=[testing_rtt_avg, cubic_rtt_avg, reno_rtt_avg], bw_data=[testing_bw_avg, cubic_bw_avg, reno_bw_avg],
                   bw_err=[int_conf_testing_bw, int_conf_cubic_bw, int_conf_reno_bw],
                   rtt_err=[int_conf_testing_rtt, int_conf_cubic_rtt, int_conf_reno_rtt],
                    feature_name=["Flecto", "Cubic", "New Reno"], markers=["o", "*", "D"], plot_name=f"bw_vs_rtt_{int_conf_perc*100}%",
                    colors=["red", "green", "blue"])