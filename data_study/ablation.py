import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_final = os.path.normpath(os.path.join(script_dir, "..", "stats_cubic_quiche.csv"))
    data = pd.read_csv(file_path_final, on_bad_lines='skip')
    data = data.apply(pd.to_numeric, errors='coerce')

    #Calculate the mean and standard deviation of each column
    column_stats = data.agg(['mean', 'std'])

    for column, stats in column_stats.items():
        mean, std = stats
        print(f"{column}: Mean={mean:.2f}, Std Dev={std:.2f}")


    zero_std_columns = column_stats.loc['std'][column_stats.loc['std'] == 0].index
    data.drop(columns=zero_std_columns, inplace=True)

    threshold = 0.5

    correlation_matrix = data.corr()
    uncorrelated_metrics = correlation_matrix[((correlation_matrix.abs() < threshold) & (correlation_matrix.abs() != 1.0))]
    uncorrelated_metrics.dropna(inplace=True)

    plt.figure(figsize=(10, 10))
    if not uncorrelated_metrics.empty:
        sns.heatmap(uncorrelated_metrics, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Uncorrelated Metrics Correlation Matrix")
        plots_dir = os.path.normpath(os.path.join(script_dir, "..", "plots"))
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "ablation_heatmap.png"))
        print("Ablation correlation matrix saved to plots/ablation_heatmap.png")
    else:
        print("No uncorrelated metrics to display.")

