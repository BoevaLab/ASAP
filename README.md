# ASAP: Allele-Specific ATAC-seq Prediction
### Early feature extraction determines performance: systematic evaluation of deep learning models for high-resolution chromatin accessibility prediction

[![Preprint](https://img.shields.io/badge/preprint-available-green)](https://doi.org/10.1101/2025.03.01.641000) &nbsp;

ASAP is a python package for fine-grained prediction of chromatin accessibility from DNA sequence, using ConvNeXt V2 blocks as powerful feature extractors. By integrating these blocks into diverse model architectures, including CNNs, LSTMs, dilated CNNs, and transformers, we demonstrate consistent performance gains, with the ConvNeXt-based dilated CNN achieving the most robust and shape-preserving predictions of ATAC-seq signal at 4 bp resolution. 

## Features

* Data Processing: Create datasets for whole-genome and peak-based analysis with support for blacklists and unmappable regions.
* Model Training: Train convolutional neural networks (e.g., ConvNext DCNN) on genomic data with customizable chromosome splits for training and validation.
* Model Evaluation: Evaluate model performance on peak and whole-genome datasets, including robustness testing.
* Prediction: Export model predictions and predict SNV effects on ATAC-seq data.
* Scalability: Support for multi-GPU training and efficient data handling.

## Installation

ASAP is available on PyPI and can be installed using pip:
```bash
pip install atac-asap
```
Ensure you have the required dependencies installed, including Python 3.8+.

## Usage

### 1. Model Training
An example script to train a model using asap has been defined in `tutorials/train.py`.

#### 1.1 Create training and validation dataset

```python
asap.training_datasets(signal_file, genome, train_chroms, val_chroms, generated, blacklist_file=None, unmap_file=None)
```
Creates training and validation datasets using genomic sequence as input and ATAC-seq signal as output.

Args:
   * `signal_file` (str): Path to the ATAC-seq signal file.
   * `genome` (str): Path to the genome file.
   * `train_chroms` (List[int]): List of chromosomes for training.
   * `val_chroms` (List[int]): List of chromosomes for validation.
   * `generated` (str): Path to save the processed data.
   * `blacklist_file` (List[str]): List of paths to blacklist files (including SNVs).
   * `unmap_file` (str): Path to the unmapped regions file.

Returns:
   * `train_dataset` (asap.dataloader.WGDataset): Training dataset
   * `val_dataset` (asap.dataloader.WGDataset): Validation dataset

#### 1.2 Train a model

```python
asap.train_model(experiment_name, model, train_dataset, val_dataset, logs_dir, n_gpus=0, max_epochs=70, learning_rate=1e-3, batch_size=64, use_map=False)
```
Trains the selected model using training and validation datasets.

Args:
   * `experiment_name` (str): The name of the experiment. This will be used to save model checkpoints.
   * `model` (nn.Module): The name of the model to train. Choose from [cnn, lstm, dcnn, convnext_cnn, convnext_lstm, convnext_dcnn, convnext_transformer].
   * `train_dataset` (asap.dataloader.WGDataset): The training dataset.
   * `val_dataset` (asap.dataloader.WGDataset): The validation dataset.
   * `logs_dir` (str): The directory where model is saved. 
   * `n_gpus` (int): The number of GPUs to use for training. Set 0 for CPU.
   * `max_epochs` (int): The maximum number of epochs to train the model.
   * `learning_rate` (float): The learning rate for the optimizer.
   * `batch_size` (int): The batch size for training.
   * `use_map` (bool): Whether to additionally use mappability information for training.

Returns:
   * None

**Note:** The Pearson's R for the trained model on the validation dataset is expected to be ~0.7.

### 2. Model Evaluation 
An example script to evaluate a model using asap has been defined in `tutorials/eval.py`.

#### 2.1 Create a peak or whole-genome dataset for evaluation on generalizability

```python
asap.peak_dataset(signal_file, peak_file, genome, chroms, generated, blacklist_file=None, unmap_file=None)
```
or 
```python
asap.wg_dataset(signal_file, genome, chroms, generated, blacklist_file=None, unmap_file=None)
```
Creates a peak or whole-genome dataset for evaluation. 

Args:
   * `signal_file` (str): Path to the signal file.
   * `peak_file` (str): Path to the peak file.
   * `genome` (str): Path to the genome file.
   * `chroms` (List[int]): List of chromosomes for evaluation.
   * `generated` (str): Path to the generated data.
   * `blacklist_file` (List[str]): List of paths to blacklist files (including SNV VCFs).
   * `unmap_file` (str): Path to the unmapped regions file.

Returns:
   * `test_dataset` (asap.dataloader.BaseDataset): Test dataset (either peak or whole-genome)

#### 2.2 Evaluate a pre-trained model on generalizability

```python
asap.eval_model(experiment_name, model, eval_dataset, logs_dir, batch_size=64, use_map=False)
```

Evaluates the pre-trained model on peak or whole-genome dataset.

Args:
   * `experiment_name` (str): The name of the experiment. This will be used to load model checkpoints.
   * `model` (nn.Module): The model name to evaluate. Choose from [cnn, lstm, dcnn, convnext_cnn, convnext_lstm, convnext_dcnn, convnext_transformer].
   * `eval_dataset` (asap.dataloader.BaseDataset): The test dataset used for model evaluation.
   * `logs_dir` (str): The directory to load model checkpoints from.
   * `batch_size` (int): The batch size for evaluation.
   * `use_map` (bool): If mappability information was used during training.

Returns:
   * `scores` (Dict[Dict]): For each test chromosome, a dictionary with Pearson's correlation (pearson_r), Mean squared error (mse), Poisson negative log-likelihood (poisson_nll), Spearman's correlation (spearman_r), and Kendall's Tau (kendall_tau).

####  2.3 Create a peak or whole-genome dataset for evaluation on robustness

```python
asap.robustness_peak_dataset(signal_file, peak_file, genome, chroms, generated, blacklist_file=None, unmap_file=None)
```
or 
```python
asap.robustness_wg_dataset(signal_file, genome, chroms, generated, blacklist_file=None, unmap_file=None)
```
Creates a peak or whole-genome dataset for evaluation on robustness. 

Args:
   * `signal_file` (str): Path to the signal file.
   * `peak_file` (str): Path to the peak file.
   * `genome` (str): Path to the genome file.
   * `chroms` (List[int]): List of chromosomes for evaluation.
   * `generated` (str): Path to the generated data.
   * `blacklist_file` (List[str]): List of paths to blacklist files (including SNV VCFs).
   * `unmap_file` (str): Path to the unmapped regions file.

Returns:
   * `test_dataset` (asap.dataloader.BaseDataset): Test dataset for robustness (either peak or whole-genome)

#### 2.4 Evaluate a pre-trained model on robustness

```python
asap.eval_robustness(experiment_name, model, eval_dataset, logs_dir, batch_size=64, use_map=False, nr_samples_for_var=17)
```

Evaluates the pre-trained model for robustness on peak or whole-genome dataset.

Args:
   * `experiment_name` (str): The name of the experiment. This will be used to load model checkpoints.
   * `model` (nn.Module): The model name to evaluate. Choose from [cnn, lstm, dcnn, convnext_cnn, convnext_lstm, convnext_dcnn, convnext_transformer].
   * `eval_dataset` (asap.dataloader.BaseDataset): The test dataset used for model evaluation.
   * `logs_dir` (str): The directory to load model checkpoints from.
   * `batch_size` (int): The batch size for evaluation.
   * `use_map` (bool): If mappability information was used during training.
   * `nr_samples_for_var` (int): The number of samples for variance calculation.

Returns:
   * `scores` (Dict[Dict]): For each test chromosome, a dictionary with average coefficient of variation (cov) and average coefficient of variation stratified by position (cov_per_bin). 

### 3. Predict SNV Effects on ATAC-seq
An example script to predict SNV effects using asap has been defined in `tutorials/predict_snv_atac.py`.

```python
asap.predict_snv_atac(experiment_name, model, snv_file, signal_file, logs_dir, out_dir, genome, chroms=[*range(1,23)], use_map=False)
```
Predict ATAC-seq for SNVs using a pre-trained model. The results are stored in a csv with allele-specific predictions for each SNV.

Args:
   * `experiment_name` (str): The name of the experiment. This will be used to load model checkpoints.
   * `model` (nn.Module): The model name to evaluate. Choose from [cnn, lstm, dcnn, convnext_cnn, convnext_lstm, convnext_dcnn, convnext_transformer].
   * `snv_file` (str): The path to the SNV VCF file.
   * `signal_file` (str): The path to the signal file.
   * `logs_dir` (str): The directory to save logs.
   * `out_path` (str): The output path for results.
   * `genome` (str): The reference genome.
   * `chroms` (List[int]): List of chromosomes to consider for prediction.
   * `use_map` (bool): Whether to use mappability for model.

Returns:
   * None

### 4. Export Predictions
An example script to export model predictions using asap has been defined in `tutorials/export_predictions.py`.

```python
asap.export_predictions(experiment_name, model, eval_dataset, logs_dir, out_dir, batch_size=64, use_map=False)
```
Export a pre-trained model's predictions as bigwig. One file will be generated for each chromosome in the eval_dataset.

Args:
   * `experiment_name` (str): The name of the experiment. This will be used to load model checkpoints.
   * `model` (nn.Module): The model name to evaluate. Choose from [cnn, lstm, dcnn, convnext_cnn, convnext_lstm, convnext_dcnn, convnext_transformer].
   * `eval_dataset` (asap.dataloader.WGDataset): The whole-genome dataset corresponding to which predictions will be generated.
   * `logs_dir` (str): The directory to load model checkpoints from.
   * `out_dir` (str): The directory to save bigwigs. 
   * `batch_size` (int): The batch size for prediction.
   * `use_map` (bool): If mappability information was used during training.

Returns:
   * None

## Citing ASAP
If you use ASAP in your work, you can cite it using
```BibTex
@article{grover2025evaluation,
  title={Evaluation of deep learning approaches for high-resolution chromatin accessibility prediction from genomic sequence},
  author={Grover, Aayush and Muser, Till and Kasak, Liine and Zhang, Lin and Krymova, Ekaterina and Boeva, Valentina},
  journal={bioRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License
ASAP is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions to ASAP are welcome! Please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.
For major changes, please open an issue first to discuss the proposed changes.
