#!/bin/bash

source ~/.bashrc
conda activate py210


model=roformer

if [[ "$model" = "dcnn" ]] || [[ "$model" = "roformer" ]]
then
    n_gpu=4
    mem=256
else
    n_gpu=1
    mem=64
fi

#declare -a arr=("TCGA-A6-A567" "TCGA-A6-A56B" "TCGA-AA-A01S" "TCGA-AA-A01X" "TCGA-AO-A124" "TCGA-AP-A051" "TCGA-AY-A54L" "TCGA-B1-A47N" "TCGA-B5-A0JN" "TCGA-B9-A44B" "TCGA-BA-A4IH" "TCGA-BJ-A0Z2" "TCGA-DU-6407" "TCGA-HE-A5NH" "TCGA-HE-A5NJ" "TCGA-IA-A40X" "TCGA-QG-A5YV" "TCGA-QG-A5YX")
declare -a arr=("TCGA-A6-A567" "TCGA-QG-A5YV" "TCGA-HE-A5NH" "TCGA-B9-A44B")
#declare -a arr=("TCGA-A6-A567" "TCGA-HE-A5NH" "TCGA-B9-A44B")
#declare -a arr=("TCGA-QG-A5YV")


for cell_type in "${arr[@]}"
do
    # python ~/open-chromatin-snv-benchmark/sbatch.py --gpu $n_gpu --cpu 8 --mem $mem --time 96:00:00 -p run_script=train_all model=$model experiment_name=$model-$cell_type +cell_line=$cell_type model.use_map=True  
    python ~/open-chromatin-snv-benchmark/sbatch.py --gpu $n_gpu --cpu 8 --mem $mem --time 96:00:00 -p run_script=train_all model=$model experiment_name=$model-$cell_type-m0 +cell_line=$cell_type
done
