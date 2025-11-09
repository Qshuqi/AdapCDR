"""README for experiments directory."""

# Experiments

This directory contains experiment scripts for different datasets:

- `coat_experiment.py`: Experiment script for Coat dataset
- `yahoo_experiment.py`: Experiment script for Yahoo! R3 dataset  
- `kuai_experiment.py`: Experiment script for KuaiRec dataset

## Usage

To run an experiment, execute:

```bash
# For Coat dataset
python experiments/coat_experiment.py

# For Yahoo! R3 dataset
python experiments/yahoo_experiment.py

# For KuaiRec dataset
python experiments/kuai_experiment.py
```

Make sure to install the package first:

```bash
pip install -e .
```

Or add the src directory to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

