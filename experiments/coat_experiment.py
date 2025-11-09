"""Experiment script for Coat dataset."""

import numpy as np
import torch
import optuna
from sklearn.metrics import roc_auc_score

from adpt_cdr import (
    MF_adpt_cdr,
    load_data,
    rating_mat_to_sample,
    binarize,
    ndcg_func,
    recall_func,
    precision_func,
)

# Set random seeds for reproducibility
np.random.seed(2020)
torch.manual_seed(2020)

# Configuration
DATASET_NAME = "coat"
BATCH_SIZE = 128
N_TRIALS = 750
OUTPUT_FILE = "coat_results.txt"


def mse_func(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((x - y) ** 2)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object.
        
    Returns:
        AUC score to maximize.
    """
    # Load data
    train_mat, test_mat = load_data("coat")
    x_train, y_train = rating_mat_to_sample(train_mat)
    x_test, y_test = rating_mat_to_sample(test_mat)
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]

    y_train = binarize(y_train)
    y_test = binarize(y_test)

    # Suggest hyperparameters
    lamb1 = trial.suggest_categorical('lamb1', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    lamb2 = trial.suggest_categorical('lamb2', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    lamb3 = trial.suggest_categorical('lamb3', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
    G = trial.suggest_int('G', 1, 5)
    emb = trial.suggest_categorical('emb', [4, 8, 16])
    calib_lamb = trial.suggest_float('calib_lamb', 0.1, 10)
    gamma = trial.suggest_float('gamma', 0.01, 0.05)
    n_bins = trial.suggest_categorical('n_bins', [5, 10, 15, 20, 25])
    lr1 = trial.suggest_categorical('lr1', [0.01, 0.05])
    lr2 = trial.suggest_categorical('lr2', [0.01, 0.05])
    lr3 = trial.suggest_categorical('lr3', [0.01, 0.05])
    option = trial.suggest_categorical('option', ['ce', 'mse'])
    calib_epoch_freq = trial.suggest_categorical('calib_epoch_freq', [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1])

    # Initialize and train model
    model = MF_adpt_cdr(num_user, num_item, batch_size=BATCH_SIZE, embedding_k=emb)
    # Model automatically detects and uses available device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.fit(
        x_train, y_train,
        n_bins=n_bins,
        calib_lamb=calib_lamb,
        lr1=lr1, lr2=lr2, lr3=lr3,
        calib_epoch_freq=calib_epoch_freq,
        gamma=gamma,
        lamb_prop=lamb1,
        lamb_pred=lamb2,
        lamb_imp=lamb3,
        G=G,
        option=option,
        tol=1e-5
    )

    # Evaluate
    test_pred = model.predict(x_test)
    mse_score = mse_func(y_test, test_pred)
    auc_score = roc_auc_score(y_test, test_pred)
    ndcg_res = ndcg_func(model, x_test, y_test)
    recall_res = recall_func(model, x_test, y_test)
    precision_res = precision_func(model, x_test, y_test)

    f1_5 = 2 * np.mean(recall_res["recall_5"]) * np.mean(precision_res["precision_5"]) / (
        np.mean(recall_res["recall_5"]) + np.mean(precision_res["precision_5"])
    )
    f1_10 = 2 * np.mean(recall_res["recall_10"]) * np.mean(precision_res["precision_10"]) / (
        np.mean(recall_res["recall_10"]) + np.mean(precision_res["precision_10"])
    )

    # Save results
    results = {
        'm': mse_score,
        'a': auc_score,
        'n5': np.mean(ndcg_res["ndcg_5"]),
        'r5': np.mean(recall_res["recall_5"]),
        'p5': np.mean(precision_res["precision_5"]),
        'f1_5': f1_5,
        'f1_10': f1_10,
    }

    with open(OUTPUT_FILE, 'a') as f:
        f.write(f'lamb1: {lamb1}, lamb2: {lamb2}, lamb3: {lamb3}, G: {G}, emb: {emb}, '
                f'n_bins: {n_bins}, lr1: {lr1}, lr2: {lr2}, lr3: {lr3}, '
                f'cali: {calib_lamb}, gamma: {gamma}, calib_epoch_freq: {calib_epoch_freq}, '
                f'option: {option}\n')
        for k, v in results.items():
            f.write(f'{k}: {v:.3f}\n')
        f.write('-----------------------------------\n')
        f.write(f'{auc_score:.3f}\n\n')

    return auc_score


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    trial = study.best_trial
    print(f'Value: {trial.value}')
    print('Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

