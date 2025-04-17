#!/bin/bash

source ~/.bashrc
conda activate py210


cell_type=GM12878
#cell_type=TCGA-B9-A44B
#cell_type=TCGA-A6-A567
#cell_type=TCGA-HE-A5NH
#cell_type=TCGA-QG-A5YV
fold=1  # Corresponding to chr9 (2) and chr17 (1)
chrom=17

#declare -a arr=("cnn" "lstm" "dcnn")
declare -a arr=("dcnn")

for model in "${arr[@]}"
do
    if [[ "$model" == "lstm" ]]; then
        python ~/open-chromatin-snv-benchmark/sbatch.py --gpu 1 --time 48:00:00 -p run_script=export_predictions model=$model experiment_name=$model-$cell_type-$fold +cell_line=$cell_type export_predictions.chrom=$chrom 
    else
        python ~/open-chromatin-snv-benchmark/sbatch.py --gpu 1 --time 48:00:00 -p run_script=export_predictions model=$model experiment_name=$model-$cell_type-$fold +cell_line=$cell_type export_predictions.chrom=$chrom 
    fi
done
