import sys
import os

sys.path.append(os.path.abspath("src"))

import asap

# The below code is a complete script that sets up the training of a model using the ASAP library. 
# It includes data paths, model parameters, training parameters, and the creation of training and validation datasets. 
# The script then trains the model using the specified parameters.

def main():

    # Data paths
    signal_file = "data/ENCFF667MDI.bigWig" # Path to the ATAC-seq signal file (bigwig)
    genome = "data/hg38.fa" # Path to the genome file
    blacklist_file = ["data/basenji_blacklist.bed", "data/example_snv.vcf"]
    unmap_file = "data/basenji_unmappable.bed" # (str): Path to the unmapped regions file.
    generated = "tmp" # (str): Path to save the processed data.
    logs_dir = "tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "TCGA-A6-A567_convnext_dcnn"

    # Training parameters
    test_chroms = [2, 10, 14, 19, 21]
    val_chroms = [1, 11, 20, 13]
    train_chroms = [x for x in range(1, 23) if x not in test_chroms and x not in val_chroms]
    n_gpus = 4

    # Create the training and validation datasets
    train, val = asap.training_datasets(
        signal_file=signal_file,
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    # Train the model
    asap.train_model(
        experiment_name=experiment_name,
        model=model_name,
        train_dataset=train,
        val_dataset=val,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
    )

if __name__ == "__main__":
    main()
