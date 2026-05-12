import numpy as np
import matplotlib.pyplot as plt

datasets = ['HCT116', 'A549_RGS', 'WTC11', 'GM23338', 'HG03432', 'MCF7', 'PC3', 'Panc1', 'RWPE2', 'GM12878_XSC', 'HEPG2_GJU', 'K562_FGK', 'IMR90']
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

data_13heads = [
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
    IMR90_13head
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
    IMR90_single
]


# Compute mean and std across chroms
means_13head = [np.mean(x) for x in data_13heads]
stds_13head = [np.std(x) for x in data_13heads]
means_single = [np.mean(x) for x in data_single]
stds_single = [np.std(x) for x in data_single]

# Plot
plt.figure(figsize=(12, 6))

width = 0.35 
x = np.arange(len(datasets))
plt.errorbar(x - width/4, means_13head, yerr=stds_13head,
             fmt='o', capsize=5, label='1 model (13 heads)', color='#1f77b4')

plt.errorbar((x + width/4)[-4:], means_single[-4:], yerr=stds_single[-4:],
             fmt='o', capsize=5, label='Independent Models', color='#ff7f0e')

plt.xlabel("Dataset", fontsize=13)
plt.ylabel("Pearson's R", fontsize=13)
plt.title("Model Comparison Across Datasets (13 headmodel vs single models)")
plt.xticks(x, datasets, fontsize=8)
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig("13heads_eval.png", dpi=300, bbox_inches='tight')