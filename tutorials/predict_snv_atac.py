import asap

# The below code is to be used for predicting SNV effects on ATAC-seq data using a trained model.
# It includes data paths, model parameters, and the creation of datasets for prediction.

def main():

    # Data paths
    signal_file = "/cluster/work/boeva/lkasak/data/TCGA-A6-A567/TCGA-A6-A567.nodup.no_chrM_MT.tn5.pval.signal.bigwig"
    genome = "/cluster/work/boeva/lkasak/data/hg38.fa"
    logs_dir = "/cluster/work/boeva/lkasak/tmp/logs"
    snv_file = "/cluster/work/boeva/lkasak/data/TCGA-A6-A567/TCGA-A6-A567_pcawg_hg38.vcf.gz"

    # Model parameters
    model_name = "convnext_dcnn"
    experiment_name = "TCGA-A6-A567_convnext_dcnn"

    # Prediction parameters
    chroms = [*range(1,23)]

    # Generate predictions for each SNV
    asap.predict_snv_atac(
        experiment_name=experiment_name,
        model=model_name,
        snv_file=snv_file,
        signal_file=signal_file,
        logs_dir=logs_dir,
        out_dir=logs_dir,
        genome=genome,
        chroms=chroms,
    )


if __name__ == "__main__":
    main()
