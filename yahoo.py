# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb
import optuna
import time
import os
from dataset import load_data
from matrix_factorization_adpt_cdr import MF_adpt_cdr
from itertools import product
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU,recall_func, precision_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 指定使用编号为1的GPU
dataset_name = "yahoo"

if dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    x_train[:, 0] -= 1
    x_train[:, 1] -= 1
    x_test[:, 0] -= 1
    x_test[:, 1] -= 1
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

# y_train = binarize(y_train)
# y_test = binarize(y_test)

y_train = binarize(y_train)
y_test = binarize(y_test)

def objective(trial):
        lamb1 = trial.suggest_categorical('lamb1', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        lamb2 = trial.suggest_categorical('lamb2', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        lamb3 = trial.suggest_categorical('lamb3', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        G = trial.suggest_int('G', 2, 5)  
        emb = trial.suggest_categorical('emb', [4,8,16]) 
        calib_lamb = trial.suggest_float('calib_lamb', 0.1, 10)
        gamma = trial.suggest_float('gamma', 0.01, 0.05)
        n_bins = trial.suggest_categorical('n_bins', [5,10,15,20,25])
        lr1 = trial.suggest_categorical('lr1', [0.01, 0.05])
        lr2 = trial.suggest_categorical('lr2', [0.01, 0.05])
        lr3 = trial.suggest_categorical('lr3', [0.01, 0.05])
        option = trial.suggest_categorical('option', ['ce', 'mse'])
        calib_epoch_freq = trial.suggest_categorical('calib_epoch_freq', [0.1, 0.2, 0.4, 0.6, 0.8, 1])

        model = MF_adpt_cdr(num_user, num_item, batch_size=2048, embedding_k = emb)
        model.cuda()

        model.fit(x_train, y_train, n_bins = n_bins, calib_lamb = calib_lamb, lr1 = lr1, lr2 = lr2, lr3 = lr3, calib_epoch_freq=calib_epoch_freq,
        gamma = gamma, lamb_prop=lamb1,
        lamb_pred=lamb2,
        lamb_imp=lamb3,
        G = G,
        option = option,
        tol=1e-5)

        test_pred = model.predict(x_test)
        mse_mfdrmc = mse_func(y_test, test_pred)
        auc_mfdrmc = roc_auc_score(y_test, test_pred)
        ndcg_res = ndcg_func(model, x_test, y_test)
        recall_res = recall_func(model, x_test, y_test)
        precision_res = precision_func(model, x_test, y_test)

        f1_5 = 2 * np.mean(recall_res["recall_5"]) * np.mean(precision_res["precision_5"]) / (np.mean(recall_res["recall_5"]) + np.mean(precision_res["precision_5"]))
        f1_10 = 2 * np.mean(recall_res["recall_10"]) * np.mean(precision_res["precision_10"]) / (np.mean(recall_res["recall_10"]) + np.mean(precision_res["precision_10"]))

        results = []
        results.append({'m': mse_mfdrmc})
        results.append({'a': auc_mfdrmc})
        results.append({'n5': np.mean(ndcg_res["ndcg_5"])})
        results.append({'r5': np.mean(recall_res["recall_5"])})
        results.append({'p5': np.mean(precision_res["precision_5"])})
        results.append({'f1_5': f1_5})
        results.append({'f1_10': f1_10})

        with open('yahoo_icdmw26_ours_new.txt', 'a') as file:
                file.write(f'lamb1: {lamb1}, lamb2: {lamb2}, lamb3: {lamb3}, G: {G}, emb: {emb}, n_bins: {n_bins}, lr1: {lr1}, lr2: {lr2}, lr3: {lr3}, cali: {calib_lamb}, gamma: {gamma}, calib_epoch_freq: {calib_epoch_freq}, option: {option}' + '\n')
                for result in results:
                        for k, v in result.items():
                                file.write(k + ': ' + str(v.round(3)) + '\n')
                file.write('-----------------------------------' + '\n')
                file.write(str(auc_mfdrmc.round(3) + np.mean(ndcg_res["ndcg_5"]).round(3)))
                file.write('\n')
                file.write('\n')
        return auc_mfdrmc + np.mean(ndcg_res["ndcg_5"])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')  