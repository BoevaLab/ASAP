import sys
import os
import numpy as np
import pyBigWig

sys.path.append(os.path.abspath("src"))

import asap

# The below code is a complete script that sets up the training of a multiheaded model using the ASAP library. 
# It includes data paths, model parameters, training parameters, and the creation of training and validation datasets. 
# The script then trains the model using the specified parameters.
# It then evaluates the model on each cell type using the corresponding head


def main():

    # data paths
    data_path = "../data"
    cells = [
        "HCT116",
        "A549_RGS",
        "WTC11",
        "GM23338",
        "HG03432",
        "MCF-7",
        "PC-3", 
        "Panc1",
        "RWPE2",
        "GM12878_XSC", 
        "HEPG2_GJU",
        "K562_FGK",
        "IMR90",
        "T_cell_f_21",
        "nk_cell_f_41",
        "t_helper_17_m_50",
        "td_CD8_ab_T_m_30",
        "a_B_cell_m_22_treated",
        "foreski_ker_m"
    ]
    signal_files = [f"{data_path}/{cell}.bigWig" for cell in cells]
    peak_files = [f"{data_path}/{cell}.bed" for cell in cells]

    genome = "data/hg38.fa"
    blacklist_file = ["../data/basenji_blacklist.bed"]
    unmap_file = "../data/basenji_unmappable.bed"
    generated = "tmp"
    logs_dir = "tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "asap-multihead-njt-19-heads"

    # Training parameters
    test_chroms = [2, 10, 14, 19, 21]
    val_chroms = [1, 11, 20, 13]
    train_chroms = [x for x in range(1, 23) if x not in test_chroms and x not in val_chroms]
    n_gpus = 1

   
    # Create the training and validation datasets
    print("Creating the datasets...")
    train_comb, val_comb = asap.training_datasets(
        signal_file=signal_files,
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    
    print("Starting training...\n")

    # Train the multiheaded model
    asap.train_multiheaded_model(
        experiment_name=experiment_name,
        model=model_name,
        num_heads=len(signal_files),
        train_dataset=train_comb,
        val_dataset=val_comb,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
    )

    print("Training done.")
    print("Evaluate the model on each dataset using its corresponding head...\n")

    for cell_index in range(len(cells)):
        print("Creating the peak dataset for ", cells[cell_index])
        peak = asap.peak_dataset(
                signal_file=signal_files[cell_index],
                peak_file=peak_files[cell_index],
                genome=genome,
                chroms=test_chroms,
                generated=generated,
                blacklist_file=blacklist_file,
                unmap_file=unmap_file,
            )

        # Evaluate the model
        print("Evaluating ", cells[cell_index])
        peak_scores_head = asap.eval_multihead_model(
            experiment_name=experiment_name,
            model=model_name,
            eval_dataset=peak,
            logs_dir=logs_dir,
            num_heads=len(signal_files),
            target_head=cell_index,
        )
        print(f"Peak scores head {cell_index}:", peak_scores_head)
        print()

if __name__ == "__main__":
    print("hi")
    main()
