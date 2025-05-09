import asap

# The below code is to be used for exporting predictions from a trained model using the ASAP library.
# It includes data paths, model parameters, and the creation of datasets for prediction.

def main():

    # Data paths
    signal_file = "/cluster/work/boeva/lkasak/data/TCGA-A6-A567/TCGA-A6-A567.nodup.no_chrM_MT.tn5.pval.signal.bigwig"
    genome = "/cluster/work/boeva/lkasak/data/hg38.fa"
    blacklist_file = ["/cluster/work/boeva/lkasak/data/basenji_blacklist.bed", "/cluster/work/boeva/lkasak/data/TCGA-A6-A567/TCGA-A6-A567_pcawg_hg38.vcf.gz"]
    generated = "/cluster/work/boeva/lkasak/tmp"
    logs_dir = "/cluster/work/boeva/lkasak/tmp/logs"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "TCGA-A6-A567_convnext_dcnn"

    # Prediction parameters
    chroms = [*range(1,23)]
    
    # Create the evaluation dataset
    wg = asap.wg_dataset(
        signal_file=signal_file,
        genome=genome,
        chroms=chroms,
        generated=generated,
        blacklist_file=blacklist_file,
        unmap_file=None,
    )

    # Export predictions
    asap.export_predictions(
        experiment_name=experiment_name,
        model=model_name,
        eval_dataset=wg,
        logs_dir=logs_dir,
        out_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
