# Adaptive Calibrated Doubly Robust Estimators for Debiased Recommendation

This is the official PyTorch implementation of "Adaptive Calibrated Doubly Robust Estimators for Debiased Recommendation" paper.

## Overview

This repository implements an adaptive calibrated doubly robust (ACDR) method for debiased recommendation systems. The method combines prediction, imputation, and propensity models to address selection bias in recommendation data using doubly robust estimation with adaptive calibration.

## Features

- **Adaptive Calibration**: Uses expected calibration error (ECE) to improve propensity estimation
- **Doubly Robust Estimation**: Combines inverse propensity scoring (IPS) and direct method for robust estimation
- **Multiple Datasets**: Supports Coat, Yahoo! R3, and KuaiRec datasets
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning

## Installation

### Requirements

- Python >= 3.8
- PyTorch == 2.0.0
- NumPy == 1.24.2
- SciPy == 1.10.1
- Pandas == 2.0.0
- Optuna == 4.1.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd icdm_adpt_cdr

# Install the package (uses setup.py)
pip install -e .
```

**Note**: The `pip install -e .` command reads the `setup.py` file in the project root directory and automatically:
- Installs the `adpt_cdr` package into your Python environment
- Installs all dependencies listed in `requirements.txt`
- Installs in editable mode, so code changes don't require reinstallation

Alternatively, install dependencies only (without installing the package itself):

```bash
pip install -r requirements.txt
```

**Note**: If you only install dependencies without installing the package, you'll need to use absolute import paths in your code or manually set `PYTHONPATH`.

## Project Structure

```
icdm_adpt_cdr/
├── src/
│   └── adpt_cdr/          # Main package
│       ├── __init__.py
│       ├── models.py      # Model implementations
│       ├── datasets.py    # Dataset loading utilities
│       └── utils.py       # Evaluation and utility functions
├── experiments/           # Experiment scripts
│   ├── coat_experiment.py
│   ├── yahoo_experiment.py
│   └── kuai_experiment.py
├── data/                  # Dataset files (not included in repo)
│   ├── coat/
│   ├── yahoo/
│   └── kuai/
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Quick Start

### Basic Usage

```python
from adpt_cdr import MF_adpt_cdr, load_data, rating_mat_to_sample, binarize

# Load data
train_mat, test_mat = load_data("coat")
x_train, y_train = rating_mat_to_sample(train_mat)
x_test, y_test = rating_mat_to_sample(test_mat)

# Binarize ratings
y_train = binarize(y_train)
y_test = binarize(y_test)

# Initialize model
model = MF_adpt_cdr(
    num_users=train_mat.shape[0],
    num_items=train_mat.shape[1],
    batch_size=128,
    embedding_k=4
)

# Move to GPU if available
if torch.cuda.is_available():
    model.cuda()

# Train model
model.fit(
    x_train, y_train,
    num_epoch=1000,
    lr1=0.05, lr2=0.05, lr3=0.05,
    n_bins=15,
    calib_lamb=1.0
)

# Predict
predictions = model.predict(x_test)
```

### Running Experiments

Run hyperparameter optimization experiments:

```bash
# Coat dataset
python experiments/coat_experiment.py

# Yahoo! R3 dataset
python experiments/yahoo_experiment.py

# KuaiRec dataset
python experiments/kuai_experiment.py
```

## Datasets

### Coat Dataset

- **Format**: Rating matrices (train.ascii, test.ascii)
- **Location**: `data/coat/`
- **Usage**: `load_data("coat")` returns `(train_matrix, test_matrix)`

### Yahoo! R3 Dataset

- **Format**: User-item-rating triplets
- **Location**: `data/yahoo/`
- **Files**: `ydata-ymusic-rating-study-v1_0-train.txt`, `ydata-ymusic-rating-study-v1_0-test.txt`
- **Usage**: `load_data("yahoo")` returns `(x_train, y_train, x_test, y_test)`

### KuaiRec Dataset

- **Format**: User-item-rating triplets
- **Location**: `data/kuai/`
- **Files**: `user.txt` (biased), `random.txt` (unbiased)
- **Usage**: `load_data("kuai")` returns `(x_train, y_train, x_test, y_test)`

**Note**: Dataset files are not included in this repository. Please download them separately and place them in the `data/` directory.

## API Reference

### Models

#### `MF_adpt_cdr`

Main model class implementing adaptive calibrated doubly robust estimation.

**Parameters:**
- `num_users` (int): Number of users
- `num_items` (int): Number of items
- `batch_size` (int): Batch size for training
- `embedding_k` (int): Embedding dimension (default: 4)

**Methods:**
- `fit()`: Train the model
- `predict()`: Make predictions

### Datasets

#### `load_data(name, data_dir="./data")`

Load dataset by name.

**Parameters:**
- `name` (str): Dataset name ('coat', 'yahoo', or 'kuai')
- `data_dir` (str): Root directory containing data folder

**Returns:**
- For 'coat': `(train_matrix, test_matrix)`
- For 'yahoo'/'kuai': `(x_train, y_train, x_test, y_test)`

### Evaluation Functions

- `ndcg_func(model, x_test, y_test, top_k_list=[5, 10])`: Calculate nDCG@K
- `recall_func(model, x_test, y_test, top_k_list=[5, 10])`: Calculate Recall@K
- `precision_func(model, x_test, y_test, top_k_list=[5, 10])`: Calculate Precision@K
- `expected_calibration_error(y_true, y_prob, n_bins=15)`: Calculate ECE

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Adaptive Calibrated Doubly Robust Estimators for Debiased Recommendation},
  author={...},
  journal={...},
  year={...}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
