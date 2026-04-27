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

    # Data paths
    signal_file_1 =  "data/GM12878/ENCFF667MDI-signal.bigWig"
    peak_file_1 = "data/GM12878/ENCFF748UZH-peak.bed"

    signal_file_2 =  "data/K562/ENCFF357GNC-signal.bigWig"
    peak_file_2 = "data/K562/ENCFF333TAT-peak.bed.gz"

    signal_file_3 =  "data/HepG2/ENCFF262URW-signal.bigWig"
    peak_file_3 = "data/HepG2/ENCFF439EIO-peak.bed.gz"

    signal_file_4 =  "data/IMR90/ENCFF770EAV-signal.bigWig"
    peak_file_4 = "data/IMR90/ENCFF243NTP-peak.bed.gz"
    
    genome = "data/hg38.fa"
    blacklist_file = ["data/basenji_blacklist.bed", "data/example_snv.vcf"]
    unmap_file = "data/basenji_unmappable.bed"
    generated = "tmp"
    logs_dir = "tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "all4"

    # Training parameters
    test_chroms = [1, 11, 20, 13]
    train_chroms  = [2, 10, 14, 19, 21]
    val_chroms = [x for x in range(1, 23) if x not in test_chroms and x not in train_chroms]
    n_gpus = 2

    # Create the training and validation datasets
    print("Create the dataset")
    train_comb, val_comb = asap.training_datasets(
        signal_file=[signal_file_1, signal_file_2, signal_file_3],
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    # Create the new datasets 
    print("Create the cl dataset")
    train_cl, val_cl = asap.training_datasets(
        signal_file=[signal_file_4],
        genome=genome,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    # TODO creating the buffers 
    # BUFFER_EVERY_K_SAMPLES = 64 # buffer one sample every K samples (here around one per batch)

    # size_train_buffer = train_comb.__len__() // BUFFER_EVERY_K_SAMPLES
    # size_val_buffer = val_comb.__len__() // BUFFER_EVERY_K_SAMPLES

    # TODO create buffer dataset class 
    
    print("Start to train!!!\n")

    # Train the model
    asap.train_multiheaded_model(
        experiment_name=experiment_name,
        model=model_name,
        num_heads=3,
        train_dataset=train_comb,
        val_dataset=val_comb,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
    )

    print("Training done.")
    print("Create eval ds")
    print()


    peak1 = asap.peak_dataset(
        signal_file=signal_file_1,
        peak_file=peak_file_1,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    peak2 = asap.peak_dataset(
        signal_file=signal_file_2,
        peak_file=peak_file_2,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    peak3 = asap.peak_dataset(
        signal_file=signal_file_3,
        peak_file=peak_file_3,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    # peak4 = asap.peak_dataset(
    #     signal_file=signal_file_4,
    #     peak_file=peak_file_4,
    #     genome=genome,
    #     chroms=test_chroms,
    #     generated=generated,
    #     blacklist_file=blacklist_file,
    #     unmap_file=unmap_file,
    # )

    # Evaluate the model
    peak_scores_head1 = asap.eval_multihead_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak1,
        logs_dir=logs_dir,
        num_heads=3,
        target_head=0,
    )
    print("Peak scores head1: ", peak_scores_head1)
    print()
    print()


    peak_scores_head2 = asap.eval_multihead_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak2,
        logs_dir=logs_dir,
        num_heads=3,
        target_head=1,
    )
    print("Peak scores head2: ", peak_scores_head2)
    print()
    print()

    peak_scores_head3 = asap.eval_multihead_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak3,
        logs_dir=logs_dir,
        num_heads=3,
        target_head=2,
    )
    print("Peak scores head3: ", peak_scores_head3)
    print()
    print()

    # peak_scores_head4 = asap.eval_multihead_model(
    #     experiment_name=experiment_name,
    #     model=model_name,
    #     eval_dataset=peak4,
    #     logs_dir=logs_dir,
    #     num_heads=4,
    #     target_head=3,
    # )
    # print("Peak scores head4:", peak_scores_head4)
    # print()
    # print()


    peak_scores_bad = asap.eval_multihead_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak1,
        logs_dir=logs_dir,
        num_heads=3,
        target_head=1,
    )
    print("Peak scores bad: ", peak_scores_bad)
    print()
    print() 

    # TODO TODO TODO THIS IS ALMOST CERTAINLY TOO LARGE TO WORK, REPLACE WITH BUFFER LATER
    print("Start CL training held-out dataset")
    asap.train_new_head_continually(
        base_experiment_name=experiment_name,
        new_experiment_name=experiment_name + "_replay",
        model=model_name,
        buffer_train_dataset=train_comb,
        buffer_val_dataset=val_comb,
        train_dataset=train_cl,
        val_dataset=val_cl,
        logs_dir=logs_dir,
        n_gpus=n_gpus,
        max_epochs=10,
    )

    # TODO evaluate new CL head 



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
