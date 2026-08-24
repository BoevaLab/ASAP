import sys
import os
import numpy as np
import pyBigWig

sys.path.append(os.path.abspath("src"))

import asap

# The below code is a complete script that sets up the fine tuning of a model using the ASAP library. 
# It includes data paths, model parameters, fine tuning parameters, and the creation of training and validation datasets. 
# The script then fine tunes and evaluates the model using the specified parameters.


def main():

    # data paths
    genome = "../data/hg38.fa"
    blacklist_file = ["../data/basenji_blacklist.bed", "../data/example_snv.vcf"]
    unmap_file = "../data/basenji_unmappable.bed"
    generated = "tmp"
    logs_dir = "tmp/logs"

    # Base model parameters
    model_name = "convnext_dcnn"
    # the name of the checkpoint for the base model
    # on which we plan to add a new head via fine-tuning
    base_experiment_name = "asap-multihead-pjt-19-heads"

    # Training parameters
    test_chroms = [2, 10, 14, 19, 21]
    val_chroms = [1, 11, 20, 13]
    train_chroms = [x for x in range(1, 23) if x not in test_chroms and x not in val_chroms]
    n_gpus = 1

   
    # Fine tuning parameters
    experiment_name_new_head = f"{base_experiment_name}-finetune-TCGA-QG-A5YV" # the checkpoint name of the new, fine-tuned model
    signal_file_new_head = "../data/TCGA-QG-A5YV.nodup.no_chrM_MT.tn5.pval.signal.bigwig"
    peak_file_new_head = "../data/TCGA-QG-A5YV.nodup.no_chrM_MT.tn5.pval0.01.300K.narrowPeak"

    train_ft, val_ft = asap.training_datasets(
        signal_file=signal_file_new_head,
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    print("Add and train a new head with fine tuning...")
    asap.train_new_head_ft( 
        base_experiment_name=base_experiment_name,
        new_experiment_name=experiment_name_new_head,
        model=model_name,
        train_dataset=train_ft,
        val_dataset=val_ft,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
        num_heads=20, # should be the number of heads of the base model + 1
        max_epochs=20
    )

    print("Create an evaluation dataset for the new head...")
    peak_new_head = asap.peak_dataset(
                signal_file=signal_file_new_head,
                peak_file=peak_file_new_head,
                genome=genome,
                chroms=test_chroms,
                generated=generated,
                blacklist_file=blacklist_file,
                unmap_file=unmap_file,
            )
    
    print("Evaluate the new head...")
    peak_scores_last_head = asap.eval_multihead_model(
        experiment_name=experiment_name_new_head,
        model=model_name,
        eval_dataset=peak_new_head,
        logs_dir=logs_dir,
        num_heads=20,
        target_head=19,
    )
    print(f"Peak scores new head:", peak_scores_last_head)


if __name__ == "__main__":
    print("hi")
    main()
