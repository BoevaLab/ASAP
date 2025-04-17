#!/bin/bash

source ~/.bashrc
conda activate py210

#cell_type=TCGA-A6-A567
#cell_type=TCGA-QG-A5YV
#cell_type=TCGA-B9-A44B
#cell_type=TCGA-HE-A5NH
cell_type=HepG2
model=cnn
use_map=0

if [[ "$model" = "dcnn" ]] || [[ "$model" = "convnexttransformer" ]] || [[ "$model" = "xlstm" ]]
then
    n_gpu=4
    mem=128
else
    n_gpu=1
    mem=64
fi


n_folds=2
fold=1
while (( fold < n_folds ))
do
    if [[ $use_map -eq 0 ]]
    then
        python ~/open-chromatin-snv-benchmark/sbatch.py --gpu $n_gpu --cpu 8 --mem $mem --time 96:00:00 -p run_script=train model=$model experiment_name=$model-$cell_type-$fold +cell_line=$cell_type train.fold=$fold  
    else
        python ~/open-chromatin-snv-benchmark/sbatch.py --gpu $n_gpu --cpu 8 --mem $mem --time 96:00:00 -p run_script=train model=$model experiment_name=$model-$cell_type-$fold-map +cell_line=$cell_type train.fold=$fold model.use_map=True   
    fi
    
    fold=$(($fold+1))
done
