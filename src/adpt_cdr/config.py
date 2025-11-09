"""Configuration file for experiments."""

# Dataset configurations
DATASET_CONFIGS = {
    "coat": {
        "batch_size": 128,
        "n_trials": 750,
        "output_file": "coat_results.txt",
        "binarize_threshold": 3,
    },
    "yahoo": {
        "batch_size": 2048,
        "n_trials": 500,
        "output_file": "yahoo_results.txt",
        "binarize_threshold": 3,
    },
    "kuai": {
        "batch_size": 2048,
        "n_trials": 500,
        "output_file": "kuai_results.txt",
        "binarize_threshold": 2,
    },
}

# Hyperparameter search spaces
HYPERPARAMETER_SPACES = {
    "lamb1": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "lamb2": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "lamb3": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    "G": {"coat": (1, 5), "yahoo": (2, 5), "kuai": (2, 5)},
    "emb": [4, 8, 16],
    "calib_lamb": (0.1, 10),
    "gamma": (0.01, 0.05),
    "n_bins": [5, 10, 15, 20, 25],
    "lr1": [0.01, 0.05],
    "lr2": [0.01, 0.05],
    "lr3": [0.01, 0.05],
    "option": ["ce", "mse"],
    "calib_epoch_freq": {
        "coat": [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1],
        "yahoo": [0.1, 0.2, 0.4, 0.6, 0.8, 1],
        "kuai": [0.1, 0.2, 0.4, 0.6, 0.8, 1],
    },
}

# Evaluation metrics
EVALUATION_METRICS = {
    "coat": {"top_k_list": [5, 10]},
    "yahoo": {"top_k_list": [5, 10]},
    "kuai": {"top_k_list": [20, 50]},
}

# Training parameters
TRAINING_PARAMS = {
    "num_epoch": 1000,
    "tol": 1e-5,
    "stop": 5,
}

