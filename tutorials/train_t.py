import sys
import os
import numpy as np
import pyBigWig

sys.path.append(os.path.abspath("src"))

import asap

# The below code is a complete script that sets up the training of a model using the ASAP library. 
# It includes data paths, model parameters, training parameters, and the creation of training and validation datasets. 
# The script then trains the model using the specified parameters.


def main():
    genome = "data/hg38.fa"
    blacklist_file = ["data/basenji_blacklist.bed"]
    unmap_file = "data/basenji_unmappable.bed"
    generated = "tmp"
    logs_dir = "tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "a_B_cell_m_22_treated"

    # Training parameters
    val_chroms = [1, 11, 20, 13]
    train_chroms  = [2, 10, 14, 19, 21]
    test_chroms = [x for x in range(1, 23) if x not in val_chroms and x not in train_chroms]
    n_gpus = 1

    
    signal_file = "/cluster/work/boeva/mindilewitsc/UniversalEPI/data/atac/raw/a_B_cell_m_22_treated.bigWig"
    peak_file = "/cluster/work/boeva/mindilewitsc/UniversalEPI/data/atac/raw/a_B_cell_m_22_treated.bed"

    train, val = asap.training_datasets(
        signal_file=signal_file,
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    print("training a new head")
    asap.train_model(
        experiment_name=experiment_name,
        model=model_name,
        train_dataset=train,
        val_dataset=val,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
        max_epochs=20
    )
    print()
    print("Create new eval dataset")
    peak = asap.peak_dataset(
                signal_file=signal_file,
                peak_file=peak_file,
                genome=genome,
                chroms=test_chroms,
                generated=generated,
                blacklist_file=blacklist_file,
                unmap_file=unmap_file,
            )
    print()
    print("Eval")
    peak_scores = asap.eval_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak,
        logs_dir=logs_dir,
    )
    print(f"Peak scores: ", peak_scores)


    

if __name__ == "__main__":
    print("hi")
    # sudo mount -a
    main()

    # bigwig (signl)

    # import pyBigWig
    # bw = pyBigWig.open(signal_file)
    # print(bw.values("chr1", 100000, 100100)) [0.1,0.9,...]
    # print(bw.chroms()) [chr1:463772] {chrom:len}

    # fasta (genome): AGGGCAAAA...

    # bed (blacklist): chr1	10468 11447 
    #     (unmapabble region): if 65% overlap, remove 2046 window


# send means for 2 and 4
# Train 13 to validate
# Do Linear Probing On Primary cells for the 13 cells , compare to Alan?


# send means for 2 and 4
# Train 13 to validate
# Do Linear Probing On Primary cells for the 13 cells , compare to Alan?
