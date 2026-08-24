import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

datasets = ["A6-A567", "B9-A44B", "HE-A5NH", "QG-A5YV"]

# Read CSV files for the results
df_single = pd.read_csv("preds_single.csv")
df_ft_njt = pd.read_csv("preds_ft_njt.csv")
df_ft_pjt = pd.read_csv("preds_ft_pjt.csv")

# Convert to lists of results per method per patient
data_single = [
    df_single[dataset].tolist()
    for dataset in datasets
]

data_ft_njt = [
    df_ft_njt[dataset].tolist()
    for dataset in datasets
]

data_ft_pjt = [
    df_ft_pjt[dataset].tolist()
    for dataset in datasets
]

# Compute means and standard deviations
means_single = [np.mean(d) for d in data_single]
stds_single = [np.std(d) for d in data_single]

means_ft_njt = [np.mean(d) for d in data_ft_njt]
stds_ft_njt = [np.std(d) for d in data_ft_njt]

means_ft_pjt = [np.mean(d) for d in data_ft_pjt]
stds_ft_pjt = [np.std(d) for d in data_ft_pjt]


# Plotting
x = np.arange(len(datasets))
width = 0.35 

plt.figure(figsize=(12, 6))

# Plot the points with error bars
plt.errorbar(x - width/3, means_single, yerr=stds_single,
             fmt='o', capsize=5, label='Independent Models', color='#ff7f0e')

plt.errorbar(x, means_ft_njt, yerr=stds_ft_njt,
             fmt='o', capsize=5, label='Fine-tuned NJT', color='green')

plt.errorbar(x + width/3, means_ft_pjt, yerr=stds_ft_pjt,
             fmt='o', capsize=5, label='Fine-tuned PJT', color='red')

def get_sig_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

def get_p_values(dataset1, dataset2):
    p_values = []
    for s, h in zip(dataset1, dataset2):
        _, p = stats.wilcoxon(s, h)
        p_values.append(p)
    return p_values


p_values_njt = get_p_values(data_single, data_ft_njt)
p_values_pjt = get_p_values(data_single, data_ft_pjt)

# 2. Add Significance Brackets
for i in range(len(datasets)):

    y_max = max(
        means_single[i] + stds_single[i],
        means_ft_njt[i] + stds_ft_njt[i],
        means_ft_pjt[i] + stds_ft_pjt[i]
    )

    # Independent vs NJT
    y_line_njt = y_max - 0.005
    h = 0.003
    x_normal = x[i] - width/3
    x_njt = x[i]

    plt.plot(
        [x_normal, x_normal, x_njt, x_njt],
        [y_line_njt, y_line_njt + h, y_line_njt + h, y_line_njt],
        lw=1,
        c='black'
    )

    plt.text(
        (x_normal + x_njt) / 2,
        y_line_njt + h,
        f"{get_sig_star(p_values_njt[i])}",
        ha='center',
        va='bottom',
        fontsize=12
    )

    # Independent vs PJT
    y_line_pjt = y_max + 0.01
    x_pjt = x[i] + width/3

    plt.plot(
        [x_normal, x_normal, x_pjt, x_pjt],
        [y_line_pjt, y_line_pjt + h, y_line_pjt + h, y_line_pjt],
        lw=1,
        c='black'
    )

    plt.text(
        (x_normal + x_pjt) / 2,
        y_line_pjt + h,
        f"{get_sig_star(p_values_pjt[i])}",
        ha='center',
        va='bottom',
        fontsize=12
    )

# Labels and formatting
plt.xticks(x, datasets, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("TCGA Dataset", fontsize=13)
plt.ylabel("Pearson's R", fontsize=13)
plt.title("Model Comparison on TCGA data (Average Test Chromosome Results and Statistical Significance During 5-fold CV)", fontsize=14, pad=20)
plt.legend(frameon=False)
plt.ylim(0.7, 0.82) # Adjusted to fit brackets
plt.tight_layout()

plt.savefig("tcga-pred-comparison.png")
plt.show()
