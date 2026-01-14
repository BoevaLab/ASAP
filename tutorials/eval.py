import asap

# The below code is a complete script that evaluates a model using the ASAP library.
# It includes data paths, model parameters, and the creation of evaluation datasets.
# It then evaluates the model on both peak and whole genome datasets, as well as robustness datasets.

def main():

    # Data paths
    signal_file = "../data/TCGA-A6-A567/TCGA-A6-A567.nodup.no_chrM_MT.tn5.pval.signal.bigwig"
    peak_file = "../data/TCGA-A6-A567/TCGA-A6-A567.nodup.no_chrM_MT.tn5.pval0.01.300K.narrowPeak.gz"
    genome = "../data/hg38.fa"
    blacklist_file = ["../data/basenji_blacklist.bed", "../data/example_snv.vcf"]
    unmap_file = "../data/basenji_unmappable.bed"
    generated = "../tmp"
    logs_dir = "../tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "TCGA-A6-A567_convnext_dcnn"

    # Training parameters
    test_chroms = [2, 10, 14, 19, 21]


    # Create the evaluation datasets
    peak = asap.peak_dataset(
        signal_file=signal_file,
        peak_file=peak_file,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )

    wg = asap.wg_dataset(
        signal_file=signal_file,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    
    peak_robustness = asap.robustness_peak_dataset(
        signal_file=signal_file,
        peak_file=peak_file,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )
    wg_robustness = asap.robustness_wg_dataset(
        signal_file=signal_file,
        genome=genome,
        chroms=test_chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=unmap_file,
    )


    # Evaluate the model
    peak_scores = asap.eval_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak,
        logs_dir=logs_dir,
    )
    print("Peak scores:", peak_scores)
    wg_scores = asap.eval_model(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=wg,
        logs_dir=logs_dir,
    )
    print("WG scores:", wg_scores)
    
    peak_robustness = asap.eval_robustness(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=peak_robustness,
        logs_dir=logs_dir,
    )
    print("Peak robustness scores:", peak_robustness)
    wg_robustness = asap.eval_robustness(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=wg_robustness,
        logs_dir=logs_dir,
    )
    print("WG robustness scores:", wg_robustness)



if __name__ == "__main__":
    main()
