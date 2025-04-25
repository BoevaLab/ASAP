# ASAP: Allele-specifc ATAC-seq Prediction
### Early feature extraction determines performance: systematic evaluation of deep learning models for high-resolution chromatin accessibility prediction

This repository provides a framework for fine-grained prediction of chromatin accessibility from DNA sequence, using ConvNeXt V2 blocks as powerful feature extractors. By integrating these blocks into diverse model architectures—including CNNs, LSTMs, dilated CNNs, and transformers—we demonstrate consistent performance gains, with the ConvNeXt-based dilated CNN achieving the most robust and shape-preserving predictions of ATAC-seq signal at 4 bp resolution. Our codebase includes benchmarks and tools for cell type-specific chromatin modeling at high resolution.

##  Setting Up the Environment

* Use the `environment.yml` file to set up the necessary dependencies.

   ```bash
   $ conda env create -f environment.yml
   ```

* This will create a conda environment named `convnext-atac`.

##   Configuration

* The `conf/` directory contains YAML configuration files. 
* `config.yaml` is the main configuration file. 
* `conf/dataset/` contains dataset-specific configurations.
* `conf/paths/` contains path configurations for different environments. 
* `conf/cell_line/` contains cell-line-specific configurations. To add a new cell type, create a new configuration here.
* `conf/model/` contains model-specific configuration. To train a new model architecture, add a new configuration here.

## Training Models

* To train a model for a specific cell type and a specific cross-validation fold on your local machine

   ```bash
   $ python run_hydra.py run_script=train model=convnextdcnn experiment_name=train_convnextdcnn_0 +cell_line=GM12878 train.fold=0 train.trainer.max_epochs=50 paths=local
   ```
   where, 
   * `model` can be [cnn, lstm, dcnn, convnextcnn, convnextlstm, convnextdcnn, convnexttransformer, xlstm]
   * `train.fold` is in [0,4]
   * `cell_line` can be [GM12878, K562, IMR90, HepG2, TCGA-A6-A567, TCGA-B9-A44B, TCGA-HE-A5NH, TCGA-QG-A5YV]
   * `model.use_map=True` can be additionally used to include additional mappability information

* To train a model on all chromosomes omitting the single-nucleotide-variant (SNV) positions
   
    ```bash
   $ python run_hydra.py run_script=train_all model=convnextdcnn experiment_name=convnextdcnn_TCGA-A6-A567 +cell_line=TCGA-A6-A567 paths=local
    ```

* The following command can be used to submit training jobs to the SLURM workload manager.  It takes various arguments to configure the job, including:

    * `-g` or `--gpu`:  Number of GPUs to use.
    * `-c` or `--cpu`: Number of CPUs to use.
    * `-m` or `--mem`:  Memory to allocate (in GB).
    * `-t` or `--time`:  Maximum runtime for the job.
    * `-p` or `--params`:  Hydra parameters to override configuration.

    ```bash
   $ python sbatch.py --gpu 4 --cpu 8 --mem 128 --time 72:00:00 -p run_script=train model=convnextdcnn experiment_name=convnextdcnn_GM12878_0 +cell_line=GM12878 train.fold=0 train.trainer.max_epochs=50
    ```

   **This can be extended to all the following tasks -- evaluations, SNV predictions, and exporting predictions as bigwigs.**


## Generate Evaluations

* To obtain Pearson's correlation and variation scores on peaks and whole genome of unseen chromosomes
    ```bash
   $ python run_hydra.py run_script=eval model=convnextdcnn experiment_name=convnextdcnn_GM12878_0 +cell_line=GM12878 eval.fold=0 paths=local
    ```
   This generates `data/results.csv` and `data/robustness.csv` containing the required scores.

## Predict SNVs

* One can predict the effect of SNVs using a pre-trained model.

    ```bash
   $ python run_hydra.py run_script=predict_snv model=convnextdcnn experiment_name=convnextdcnn_TCGA-A6-A567 +cell_line=TCGA-A6-A567 paths=local
    ```
   Here the SNVs are defined in `conf/cell_line/TCGA-A6-A567`. 

   This generates a csv with allele-specific predicted signals in `data/generated/predictions`.

## Exporting Predictions

* To export model predictions of a specific chromosome as a BigWig file

    ```bash
   $ python hydra.py run_script=export_predictions model=convnextdcnn experiment_name=convnextdcnn_GM12878_1 +cell_line=$cell_type export_predictions.chrom=17 paths=local
    ```
   The above command uses model trainined with fold=1 where chromsome 17 is a test chromosome.
