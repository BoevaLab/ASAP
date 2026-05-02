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

    leomed_path = "/cluster/work/boeva/mindilewitsc/UniversalEPI/data/atac/raw"

    datasets = [
        ["HCT116", "ENCFF624HRW.bigWig", "ENCFF296ZZB.bed"],
        ["A549_RGS", "ENCFF399KCR.bigWig", "ENCFF899OMR.bed"],
        ["WTC11", "ENCFF123YPY.bigWig", "ENCFF321VDH.bed"],
        ["GM23338", "ENCFF234AYB.bigWig", "ENCFF567ZCX.bed"],
        ["HG03432", "ENCFF993BIL.bigWig", "ENCFF831FGS.bed"],
        ["MCF-7", "ENCFF976UNK.bigWig", "ENCFF821OEF.bed"],
        ["PC-3", "ENCFF145UAD.bigWig", "ENCFF811MOZ.bed"],
        ["Panc1", "ENCFF794CNJ.bigWig", "ENCFF182SSP.bed"],
        ["RWPE2", "ENCFF881UWW.bigWig", "ENCFF729MMJ.bed"],
        ["GM12878", "ENCFF667MDI.bigWig", "ENCFF748UZH.bed"],
        ["HEPG2_GJU", "ENCFF262URW.bigWig", "ENCFF439EIO.bed"],
        ["K562", "ENCFF357GNC.bigWig", "ENCFF333TAT.bed"],
        ["IMR90", "ENCFF770EAV.bigWig", "ENCFF243NTP.bed"]
    ]

    print(len(datasets))
    signal_files = [f"{leomed_path}/{dataset[0]}.bigWig" for dataset in datasets]
    signal_files[0] = "/cluster/work/boeva/mindilewitsc/UniversalEPI/data/atac/raw/HCT116.bigwig"
    peak_files = [f"{leomed_path}/{dataset[0]}.bed" for dataset in datasets]

    print(signal_files, peak_files)
    

    genome = "data/hg38.fa"
    blacklist_file = ["data/basenji_blacklist.bed", "data/example_snv.vcf"]
    unmap_file = "data/basenji_unmappable.bed"
    generated = "tmp"
    logs_dir = "tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "allCellLines"

    # Training parameters
    test_chroms = [1, 11, 20, 13]
    train_chroms  = [2, 10, 14, 19, 21]
    val_chroms = [x for x in range(1, 23) if x not in test_chroms and x not in train_chroms]
    n_gpus = 2

    # Create the training and validation datasets
    print("create the dataset")
    train_comb, val_comb = asap.training_datasets(
        signal_file=signal_files,
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    
    print("start to train!!!\n")

    # Train the model
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
    print("Create eval ds")
    print()

    peak = []
    for i in range(len(signal_files)):
        print("Evaluating ", datasets[i][0])
        peak.append(asap.peak_dataset(
                signal_file=signal_files[i],
                peak_file=peak_files[i],
                genome=genome,
                chroms=test_chroms,
                generated=generated,
                blacklist_file=blacklist_file,
                unmap_file=unmap_file,
            )
        )

        # Evaluate the model
        peak_scores_head = asap.eval_multihead_model(
            experiment_name=experiment_name,
            model=model_name,
            eval_dataset=peak[i],
            logs_dir=logs_dir,
            num_heads=len(signal_files),
            target_head=i,
        )
        print(f"Peak scores head {i}:", peak_scores_head)
        print()
        print()


    peak_scores_bad = asap.eval_multihead_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak,
        logs_dir=logs_dir,
        num_heads=len(signal_files),
        target_head=1,
    )
    print("Peak scores bad:", peak_scores_bad)
    print()

    print("Finished")


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
