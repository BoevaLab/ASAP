#!/bin/bash

source ~/.bashrc
conda activate py210


model=cnn

#declare -a arr=("TCGA-A6-A567" "TCGA-A6-A56B" "TCGA-AA-A01S" "TCGA-AA-A01X" "TCGA-AO-A124" "TCGA-AY-A54L" "TCGA-B1-A47N" "TCGA-B5-A0JN" "TCGA-B9-A44B" "TCGA-BA-A4IH" "TCGA-BJ-A0Z2" "TCGA-DU-6407" "TCGA-HE-A5NH" "TCGA-HE-A5NJ" "TCGA-IA-A40X" "TCGA-QG-A5YV" "TCGA-QG-A5YX")
declare -a arr=("TCGA-A6-A567" "TCGA-QG-A5YV" "TCGA-HE-A5NH" "TCGA-B9-A44B")
#declare -a arr=("TCGA-QG-A5YV")
#declare -a arr=("TCGA-A6-A567" "TCGA-HE-A5NH" "TCGA-B9-A44B")


for cell_type in "${arr[@]}"
do
    #python ~/open-chromatin-snv-benchmark/sbatch.py --gpu 1 --cpu 8 --time 24:00:00 -p run_script=predict_snv model=$model experiment_name=$model-$cell_type +cell_line=$cell_type model.use_map=True
    python ~/open-chromatin-snv-benchmark/sbatch.py --cpu 8 --time 24:00:00 -p run_script=predict_snv model=$model experiment_name=$model-$cell_type-m0 +cell_line=$cell_type
done
