# request rtx4090
# change order and rerun 19heads
# lp on all new heads
# record runtimes
# keep finetuning 13 to 14

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

datasets = ["T_cell_f_21", 
                "nk_cell_f_41", 
               "t_helper_17_m_50",
               "td_CD8_ab_T_m_30",
               "a_B_cell_m_22_treated",
               "foreski_ker_m"]

T_cell_f_21_lp = [0.681,  0.675, 0.666, 0.656] #0.669
T_cell_f_21_separate = [0.7057,  0.701, 0.699,  0.692] #0.699

nk_cell_f_41_lp = [0.702, 0.709, 0.691, 0.683]
nk_cell_f_41_separated = [0.721,  0.726, 0.716, 0.713]

t_helper_17_m_50_lp = [0.698, 0.703, 0.684, 0.689]
t_helper_17_m_50_separated = [0.0,0,0,0]

td_CD8_ab_T_m_30_lp = [0.695, 0.679, 0.671, 0.678]
td_CD8_ab_T_m_30_separated = [0.719, 0.703,  0.708, 0.696]

a_B_cell_m_22_treated_lp = [0.723, 0.712, 0.706, 0.730] #0.71775
a_B_cell_m_22_treated_separated = [0.702, 0.706, 0.703, 0.707]  #0.704

foreski_ker_m_lp = [ 0.706, 0.701, 0.685, 0.708]
foreski_ker_m_separated = [0.744, 0.742, 0.731,  0.738]


data_single = [T_cell_f_21_separate, 
               nk_cell_f_41_separated, 
               t_helper_17_m_50_separated,
               td_CD8_ab_T_m_30_separated,
               a_B_cell_m_22_treated_separated,
               foreski_ker_m_separated]
data_lp = [T_cell_f_21_lp, 
                nk_cell_f_41_lp, 
               t_helper_17_m_50_lp,
               td_CD8_ab_T_m_30_lp,
               a_B_cell_m_22_treated_lp,
               foreski_ker_m_lp]

# Compute means and stds6
means_single = [np.mean(d) for d in data_single]
stds_single = [np.std(d) for d in data_single]
means_lp = [np.mean(d) for d in data_lp]
stds_lp = [np.std(d) for d in data_lp]

def get_sig_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

p_values = []
for s, h in zip(data_single, data_lp):
    _, p = stats.wilcoxon(s, h) # Wilcox
    p_values.append(p)

# Plotting
x = np.arange(len(datasets))
width = 0.35 

plt.figure(figsize=(12, 6))

# Plot the points with error bars
plt.errorbar(x - width/4, means_lp, yerr=stds_lp,
             fmt='o', capsize=5, label='Linear Probing', color='#1f77b4')

plt.errorbar(x + width/4, means_single, yerr=stds_single,
             fmt='o', capsize=5, label='Independent Models', color='#ff7f0e')

# # 2. Add Significance Brackets
# for i in range(len(datasets)):
#     # Determine height of the bracket
#     y_max = max(means_lp[i] + stds_lp[i], means_single[i] + stds_single[i])
#     y_line = y_max + 0.005  # Bracket baseline
#     h = 0.003              # Bracket tick height
    
#     # Draw bracket line
#     # (x_start, x_end), (y_start, y_end)
#     plt.plot([x[i] - width/4, x[i] - width/4, x[i] + width/4, x[i] + width/4], 
#              [y_line, y_line + h, y_line + h, y_line], lw=1, c='black')
    
#     # Add star/ns text
#     star = get_sig_star(p_values[i])
#     plt.text(x[i], y_line + h, star, ha='center', va='bottom', fontsize=12)

# Labels and formatting
plt.xticks(x, datasets, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Dataset", fontsize=13)
plt.ylabel("Pearson's R", fontsize=13)
plt.title("Model Comparison Across Datasets", fontsize=14, pad=20)
plt.legend(frameon=False)
plt.ylim(0.65, 0.78) # Adjusted to fit brackets
plt.tight_layout()

plt.savefig("13head-lp_on_primary.png")
plt.show()

# Print values for verification
for i, ds in enumerate(datasets):
    print(f"{ds}: p-value = {p_values[i]:.4f} ({get_sig_star(p_values[i])})")
