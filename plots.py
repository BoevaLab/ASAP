import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

test_chroms = [1, 11, 20, 13]
datasets = ["GM12878", "K562"]

# 4 models
GM12878_single = [0.701, 0.695, 0.692, 0.687] # 11 epochs
K562_single = [0.711, 0.713, 0.723, 0.698] # 11 epochs
HepG2_single = [0.704, 0.704, 0.682, 0.701] # 12 epochs
IMR90_single = [0.739, 0.734, 0.731, 0.714] # 11 epochs

# one model, 2 heads
# 8 epochs
GM12878_2head = [0.706, 0.699, 0.695, 0.688]
K562_2head = [0.715, 0.718, 0.722, 0.699]

# one model, 4 heads
# 7 epochs
GM12878_4head = [0.703,  0.702, 0.692,  0.685]
K562_4head = [0.720, 0.723, 0.730,  0.705]
HepG2_4head = [0.716, 0.718, 0.69, 0.713]
IMR90_4head = [0.742, 0.733, 0.729, 0.708]




data_single = [GM12878_single, K562_single]
data_2head = [GM12878_2head, K562_2head]

# Compute means and stds
means_single = [np.mean(d) for d in data_single]
stds_single = [np.std(d) for d in data_single]
means_2head = [np.mean(d) for d in data_2head]
stds_2head = [np.std(d) for d in data_2head]

def get_sig_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

p_values = []
for s, h in zip(data_single, data_2head):
    _, p = stats.wilcoxon(s, h) # Wilcox
    p_values.append(p)

# Plotting
x = np.arange(len(datasets))
width = 0.35 

plt.figure(figsize=(8, 6))

# Plot the points with error bars
plt.errorbar(x - width/4, means_2head, yerr=stds_2head,
             fmt='o', capsize=5, label='1 model (2 heads)', color='#1f77b4')

plt.errorbar(x + width/4, means_single, yerr=stds_single,
             fmt='o', capsize=5, label='Independent Models', color='#ff7f0e')

# 2. Add Significance Brackets
for i in range(len(datasets)):
    # Determine height of the bracket
    y_max = max(means_2head[i] + stds_2head[i], means_single[i] + stds_single[i])
    y_line = y_max + 0.005  # Bracket baseline
    h = 0.003              # Bracket tick height
    
    # Draw bracket line
    # (x_start, x_end), (y_start, y_end)
    plt.plot([x[i] - width/4, x[i] - width/4, x[i] + width/4, x[i] + width/4], 
             [y_line, y_line + h, y_line + h, y_line], lw=1, c='black')
    
    # Add star/ns text
    star = get_sig_star(p_values[i])
    plt.text(x[i], y_line + h, star, ha='center', va='bottom', fontsize=12)

# Labels and formatting
plt.xticks(x, datasets, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dataset", fontsize=13)
plt.ylabel("Pearson's R", fontsize=13)
plt.title("Model Comparison Across Datasets", fontsize=14, pad=20)
plt.legend(frameon=False)
plt.ylim(0.67, 0.74) # Adjusted to fit brackets
plt.tight_layout()

plt.savefig("2head-comparison-with-stats.png")
plt.show()

# Print values for verification
for i, ds in enumerate(datasets):
    print(f"{ds}: p-value = {p_values[i]:.4f} ({get_sig_star(p_values[i])})")