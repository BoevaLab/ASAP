#!/bin/bash

source ~/.bashrc
conda activate py210

cell_type=TCGA-A6-A567
#cell_type=TCGA-B9-A44B
#cell_type=TCGA-QG-A5YV
#cell_type=TCGA-HE-A5NH
#cell_type=GM12878
model=dcnn
use_map=0

n_folds=5
fold=0
while (( fold < n_folds ))
do
    if [[ $use_map -eq 0 ]]
    then
        python ~/open-chromatin-snv-benchmark/sbatch.py --time 96:00:00 -p run_script=eval model=$model experiment_name=$model-upgrade-$cell_type-$fold +cell_line=$cell_type eval.fold=$fold  
    else
        python ~/open-chromatin-snv-benchmark/sbatch.py --gpu 1 --time 96:00:00 -p run_script=eval model=$model experiment_name=$model-upgrade-$cell_type-$fold-map +cell_line=$cell_type eval.fold=$fold model.use_map=True   
    fi
    
    fold=$(($fold+1))
done
