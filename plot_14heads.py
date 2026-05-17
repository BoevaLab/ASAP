import numpy as np
import matplotlib.pyplot as plt

datasets = ['HCT116', 'A549_RGS', 'WTC11', 'GM23338', 'HG03432', 'MCF7', 'PC3', 'Panc1', 'RWPE2', 'GM12878_XSC', 'HEPG2_GJU', 'K562_FGK', 'IMR90', "T_cell_f_21_lp"]
chroms = [1, 11, 20, 13]

# combined, 13 epochs (time limit)
HCT116_13head = [0.748, 0.747, 0.731, 0.742]
A549_RGS_13head = [0.736, 0.733, 0.707, 0.718]
WTC11_13head = [0.791, 0.783, 0.777, 0.783]
GM23338_13head = [0.796, 0.786, 0.782, 0.799]
HG03432_13head = [0.74, 0.735, 0.738, 0.74]
MCF7_13head = [0.738, 0.734, 0.695, 0.742]
PC3_13head = [0.714, 0.721, 0.699, 0.714]
Panc1_13head = [0.766, 0.759, 0.746, 0.745]
RWPE2_13head = [0.717, 0.716, 0.692, 0.705]
GM12878_XSC_13head = [0.703, 0.699, 0.692, 0.69]
HEPG2_GJU_13head = [0.712, 0.711, 0.685, 0.708]
K562_FGK_13head = [0.714, 0.715, 0.722, 0.701]
IMR90_13head = [0.738, 0.726, 0.73, 0.709]
T_cell_f_21_lp = [0.681,  0.675, 0.666, 0.656]

data_13heads_plus_lp = [
    HCT116_13head,
    A549_RGS_13head,
    WTC11_13head,
    GM23338_13head,
    HG03432_13head,
    MCF7_13head,
    PC3_13head,
    Panc1_13head,
    RWPE2_13head,
    GM12878_XSC_13head,
    HEPG2_GJU_13head,
    K562_FGK_13head,
    IMR90_13head,
    T_cell_f_21_lp
]

# 13 separate models
HCT116_single = [0.758]
A549_RGS_single = [0.737]
WTC11_single = [0.8]
GM23338_single = [0.821]
HG03432_single = [0.715]
MCF7_single = [0.738]
PC3_single = [0.7]
Panc1_single = [0.744]
RWPE2_single = [0.745]
GM12878_XSC_single = [0.701, 0.695, 0.692, 0.687]
HEPG2_GJU_single = [0.704, 0.704, 0.682, 0.701]
K562_FGK_single = [0.711, 0.713, 0.723, 0.698]
IMR90_single = [0.739, 0.734, 0.731, 0.714]
T_cell_f_21_separate = [0.7057,  0.701, 0.699,  0.692]

data_single = [
    HCT116_single,
    A549_RGS_single,
    WTC11_single,
    GM23338_single,
    HG03432_single,
    MCF7_single,
    PC3_single,
    Panc1_single,
    RWPE2_single,
    GM12878_XSC_single,
    HEPG2_GJU_single,
    K562_FGK_single,
    IMR90_single,
    T_cell_f_21_separate
]


data_14head_cl = [[0.77388, 0.77218, 0.76423, 0.76750],
[0.75115, 0.74464, 0.72093, 0.73558],
[0.80259, 0.79624, 0.79078, 0.79235],
[0.80860, 0.79816, 0.79233, 0.80874],
[0.75988, 0.75415, 0.75297, 0.75616],
[0.76864, 0.76478, 0.72526, 0.77316],
[0.73899, 0.74466, 0.72559, 0.74090],
[0.77968, 0.76998, 0.76067, 0.75639],
[0.74593, 0.74494, 0.71737, 0.73883],
[0.72286, 0.71904, 0.70844, 0.70955],
[0.72074, 0.71941, 0.69779, 0.71925],
[0.72675, 0.72961, 0.73622, 0.71564],
[0.75554, 0.74607, 0.74561, 0.72812],
[0.74056, 0.73343, 0.72759, 0.71919]]


# Compute mean and std across chroms
means_14head = [np.mean(x) for x in data_13heads_plus_lp]
stds_14head = [np.std(x) for x in data_13heads_plus_lp]
means_single = [np.mean(x) for x in data_single]
stds_single = [np.std(x) for x in data_single]
means_cl = [np.mean(x) for x in data_14head_cl]
stds_cl = [np.std(x) for x in data_14head_cl]

# Plot
plt.figure(figsize=(12, 6))

width = 0.35 
x = np.arange(len(datasets))

plt.errorbar((x + width/4)[-5:], means_single[-5:], yerr=stds_single[-5:],
             fmt='o', capsize=5, label='Independent Models', color='#ff7f0e')

plt.errorbar(x , means_14head, yerr=stds_14head,
             fmt='o', capsize=5, label='1 model (13 heads, 1 head LP)', color='#1577b4')

plt.errorbar(x - width/4, means_cl, yerr=stds_cl,
             fmt='o', capsize=5, label='1 model (CL)', color='green')
for i in range(len(datasets) - 1):
    plt.axvline(
        x=i + 0.5,
        color='gray',
        linestyle=':',
        linewidth=1,
        alpha=0.7
    )

plt.xlabel("Dataset", fontsize=13)
plt.ylabel("Pearson's R", fontsize=13)
plt.title("Model Comparison Across Datasets (14 headmodel vs single models)")
plt.xticks(x, datasets, fontsize=8)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("14heads_eval.png", dpi=300, bbox_inches='tight')